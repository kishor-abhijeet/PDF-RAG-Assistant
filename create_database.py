#load the pdf
#split into chunks
#create the embeddings
#store into chroma
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

#Load the PDF file
data = PyPDFLoader("document loaders/Cos.pdf")
#convert the file into Langchain Document objects
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks  = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

vector_store = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

