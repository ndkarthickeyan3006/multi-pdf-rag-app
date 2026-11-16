
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# ---------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDF documents and ask questions "
    "across all documents."
)


# ---------------------------------------
# 2. CHECK API KEY
# ---------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY not found. "
        "Please configure your environment variable."
    )
    st.stop()


# ---------------------------------------
# 3. INITIALIZE SESSION STATE
# ---------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------
# 4. INITIALIZE MODELS
# ---------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)


# ---------------------------------------
# 5. MULTIPLE PDF UPLOAD
# ---------------------------------------

st.header("Step 1: Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------------
# 6. PROCESS AND INDEX ALL PDFs
# ---------------------------------------

if st.button("Build Vector Database"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:

        all_documents = []

        with st.spinner("Reading and processing PDFs..."):

            for uploaded_file in uploaded_files:

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    # Load PDF pages
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()

                    # Preserve original filename and page number
                    for document in documents:
                        document.metadata["source"] = (
                            uploaded_file.name
                        )

                        # PyPDFLoader page numbers are zero-based
                        document.metadata["page"] = (
                            int(document.metadata.get("page", 0)) + 1
                        )

                    all_documents.extend(documents)

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # --------------------------------
            # 7. SPLIT TEXT INTO CHUNKS
            # --------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(
                all_documents
            )

            if not chunks:
                st.error("No readable text found in the PDFs.")
                st.stop()

            # --------------------------------
            # 8. CREATE ONE FAISS DATABASE
            # --------------------------------

            with st.spinner("Generating embeddings..."):

                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

            # Save database in session state
            st.session_state.store = vector_store

            st.session_state.indexed_files = [
                file.name for file in uploaded_files
            ]

            st.session_state.messages = []

        # --------------------------------
        # 9. DISPLAY INDEXING RESULTS
        # --------------------------------

        st.success("PDF indexing completed successfully!")

        st.metric(
            "Total chunks indexed",
            len(chunks)
        )

        st.write("### Uploaded Documents")

        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")


# ---------------------------------------
# 10. QUESTION AND ANSWER SECTION
# ---------------------------------------

st.divider()

st.header("Step 2: Ask Questions")

if st.session_state.store is None:

    st.info("Please upload PDFs and build the vector database first.")

else:

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your uploaded PDFs..."
    )

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                # --------------------------------
                # 11. RETRIEVE RELEVANT CHUNKS
                # --------------------------------

                retriever = st.session_state.store.as_retriever(
                    search_kwargs={"k": 8}
                )

                docs = retriever.invoke(question)

                if not docs:

                    answer = (
                        "I could not find relevant information "
                        "in the uploaded documents."
                    )

                    sources = []

                else:

                    # --------------------------------
                    # 12. PREPARE CONTEXT
                    # --------------------------------

                    context = ""

                    for doc in docs:

                        filename = doc.metadata.get(
                            "source", "Unknown"
                        )

                        page = doc.metadata.get(
                            "page", "Unknown"
                        )

                        context += (
                            f"\n\nSource: {filename}, "
                            f"Page: {page}\n"
                            f"Content:\n{doc.page_content}"
                        )

                    # --------------------------------
                    # 13. GENERATE ANSWER
                    # --------------------------------

                    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

Instructions:
1. Give a clear and accurate answer.
2. Do not invent facts that are not in the documents.
3. If the answer is not available, clearly say so.
4. If two or more documents disagree, explain the
   disagreement instead of choosing one silently.
5. Identify which documents support each conflicting
   statement.
6. Do not assume that missing information means
   the information is false.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    # --------------------------------
                    # 14. EXTRACT UNIQUE SOURCES
                    # --------------------------------

                    sources = []

                    seen_sources = set()

                    for doc in docs:

                        filename = doc.metadata.get(
                            "source", "Unknown"
                        )

                        page = doc.metadata.get(
                            "page", "Unknown"
                        )

                        source_key = (filename, page)

                        if source_key not in seen_sources:

                            seen_sources.add(source_key)

                            sources.append(
                                (filename, page)
                            )

            # --------------------------------
            # 15. DISPLAY ANSWER
            # --------------------------------

            st.markdown("### Answer")
            st.markdown(answer)

            # --------------------------------
            # 16. DISPLAY SOURCES
            # --------------------------------

            st.markdown("### 📌 Sources")

            if sources:

                for filename, page in sources:

                    st.caption(
                        f"📄 {filename} — Page {page}"
                    )

            else:

                st.caption("No supporting sources found.")

        # Save assistant response
        source_text = "\n".join(
            [
                f"- {filename} — Page {page}"
                for filename, page in sources
            ]
        )

        complete_answer = (
            f"{answer}\n\n### Sources\n{source_text}"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": complete_answer
        })


# ---------------------------------------
# 17. RESET APPLICATION
# ---------------------------------------

if st.button("Clear Database and Chat"):

    st.session_state.store = None
    st.session_state.messages = []
    st.session_state.indexed_files = []

    st.rerun()
    