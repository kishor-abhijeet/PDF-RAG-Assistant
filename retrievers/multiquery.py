from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv



load_dotenv()

docs = [
    Document(page_content="A kernel is a part of the operating system that interacts directly with the hardware and performs the most crucial tasks."),

Document(page_content="The kernel is the heart or core component of an operating system and is the very first part of the OS to load during startup."),

Document(page_content="User space is where application software runs. Applications do not have privileged access to the underlying hardware and interact with the kernel."),

Document(page_content="A shell, also known as a command interpreter, receives commands from users and gets them executed."),

Document(page_content="The kernel performs process management, including scheduling processes and threads on CPUs, creating and deleting processes, suspending and resuming processes, and providing mechanisms for process synchronization and communication."),
]


embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()


llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)