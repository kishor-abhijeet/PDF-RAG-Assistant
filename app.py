import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.answer-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a book or PDF and ask questions about its content.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# API Keys
# --------------------------------------------------

gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY is missing from your .env file.")
    st.stop()

if not groq_api_key:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload Book")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown("""
    1. Upload a PDF/book
    2. The PDF is split into chunks
    3. Gemini creates embeddings
    4. ChromaDB stores the embeddings
    5. Ask questions about the PDF
    6. Groq generates the answer
    """)


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# --------------------------------------------------
# Process uploaded PDF
# --------------------------------------------------

if uploaded_file is not None:

    # Process only when a new file is uploaded
    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):

        with st.spinner("Processing your PDF..."):

            # Create temporary directory
            temp_dir = tempfile.mkdtemp()

            pdf_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            # Save uploaded PDF
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ------------------------------------------
            # Load PDF
            # ------------------------------------------

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            # ------------------------------------------
            # Split PDF into chunks
            # ------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # ------------------------------------------
            # Create embeddings
            # ------------------------------------------

            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=gemini_api_key
            )

            # ------------------------------------------
            # Create ChromaDB
            # ------------------------------------------

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model
            )

            # Store vectorstore in session
            st.session_state.vectorstore = vectorstore

            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )

            # Clear previous conversation
            st.session_state.messages = []

        st.success(
            f"✅ {uploaded_file.name} is ready!"
        )


# --------------------------------------------------
# Display uploaded file
# --------------------------------------------------

if st.session_state.uploaded_file_name:

    st.info(
        f"📄 Current document: "
        f"**{st.session_state.uploaded_file_name}**"
    )


# --------------------------------------------------
# Chat interface
# --------------------------------------------------

st.subheader("💬 Ask Questions")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# User question
# --------------------------------------------------

query = st.chat_input(
    "Ask something about your PDF..."
)


if query:

    # Make sure PDF is uploaded
    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload a PDF first."
        )

        st.stop()

    # Display user message
    with st.chat_message("user"):

        st.markdown(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    # --------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------

    with st.spinner("Searching the document..."):

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
        )

        docs = retriever.invoke(query)


    # --------------------------------------------------
    # Create context
    # --------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # --------------------------------------------------
    # Create prompt
    # --------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the
            provided context.

            If the answer is not present in the context,
            say:

            "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context:

            {context}

            Question:

            {question}
            """
        )
    ])


    # --------------------------------------------------
    # Create Groq LLM
    # --------------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key
    )


    # --------------------------------------------------
    # Generate final prompt
    # --------------------------------------------------

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            response = llm.invoke(
                final_prompt
            )

            answer = response.content

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })