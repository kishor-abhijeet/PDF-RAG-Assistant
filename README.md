# RAG Project

This project is a Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, Google Gemini, and Groq. It allows users to upload PDF documents, split them into chunks, generate embeddings, store them in a vector database, and ask natural-language questions about the document content.

The project includes both a Streamlit web interface and a command-line RAG workflow for experimentation and learning.

## Features

- PDF ingestion and chunking
- Semantic search using embeddings
- Vector storage with ChromaDB
- Question answering grounded in document context
- Streamlit UI for quick interaction
- CLI-based prototype for retriever + LLM workflow
- Support for document loading and retrieval experiments

## Tech Stack

- Python 3.13+
- LangChain
- ChromaDB
- Google Generative AI embeddings
- Groq LLMs
- Streamlit
- PyPDF
- Python-dotenv

## Project Structure

```text
RAG PROJECT/
├── app.py                  # Streamlit PDF RAG assistant
├── main.py                 # CLI-based retrieval + generation example
├── create_database.py      # Creates the vector database from a PDF
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and Python packaging config
├── .env                    # Local environment variables (not committed)
├── chroma_db/              # Persisted vector database
├── document loaders/       # Sample PDFs and loader experiments
├── retrievers/             # Retrieval experiments and examples
├── src/                    # Source package structure
├── vector store/           # DB-related assets and helpers
├── README.md               # Project documentation
└── uv.lock                 # Lockfile for uv package manager
```

## Prerequisites

- Python 3.13 or newer
- A virtual environment (recommended)
- API keys for:
  - `GEMINI_API_KEY`
  - `GROQ_API_KEY`

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd "RAG PROJECT"
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the required keys:

```env
GEMINI_API_KEY=your_google_gemini_key
GROQ_API_KEY=your_groq_key
```

## Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in the browser, upload a PDF, and ask questions about it.

### Build the vector database

```bash
python create_database.py
```

This loads the PDF in `document loaders/Cos.pdf`, splits it into chunks, creates embeddings, and stores them in the `chroma_db` directory.

### Run the CLI version

```bash
python main.py
```

This starts an interactive terminal session where you can ask questions against the stored vector database.

## How the RAG Pipeline Works

1. A PDF is loaded using a document loader.
2. The document is split into smaller chunks.
3. Embeddings are generated using Gemini embeddings.
4. Chunks are stored in ChromaDB.
5. A user query is converted into embeddings and matched against stored chunks.
6. Relevant context is retrieved and passed to the Groq LLM.
7. The model responds using only the retrieved document context.

## Notes

- This project is a learning and experimentation project for building RAG pipelines.
- The app is designed around PDF-based knowledge retrieval and grounded question answering.
- The `document loaders/` and `retrievers/` folders contain examples and experiments that may be extended for other document types or retrieval strategies.

## Future Improvements

- Add support for multiple file types (TXT, DOCX, Markdown)
- Add document metadata and source tracking
- Improve chunking strategy for better retrieval quality
- Add chat history and multi-turn memory
- Add API/backend deployment with FastAPI
- Add authentication and user management
- Add Dockerization for easier deployment

## License

This project currently does not include a custom license file. Please check with the repository owner before reusing or distributing it commercially.
