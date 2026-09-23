import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

docs = [
    Document(
        page_content="This is a test document for vector store.",
        metadata={"source": "test document"},
    ),
    Document(
        page_content="Vector stores help applications search documents using embeddings.",
        metadata={"source": "vector store document"},
    ),
    Document(
        page_content="LangChain provides tools for loading, splitting, and retrieving documents.",
        metadata={"source": "langchain document"},
    ),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db",
)

result = vector_store.similarity_search("what are the functions of Kernel?", k =2)
for r in result:
    print(r.page_content)
    print(r.metadata)
retriver = vector_store.as_retriever()

docs = retriver.invoke("functions of kernel")

for d in docs:
    print(d.page_content)
