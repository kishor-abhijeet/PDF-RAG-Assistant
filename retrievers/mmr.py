from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



docs = [
    Document(page_content="A kernel is a part of the operating system that interacts directly with the hardware and performs the most crucial tasks."),

Document(page_content="The kernel is the heart or core component of an operating system and is the very first part of the OS to load during startup."),

Document(page_content="User space is where application software runs. Applications do not have privileged access to the underlying hardware and interact with the kernel."),

Document(page_content="A shell, also known as a command interpreter, receives commands from users and gets them executed."),

Document(page_content="The kernel performs process management, including scheduling processes and threads on CPUs, creating and deleting processes, suspending and resuming processes, and providing mechanisms for process synchronization and communication."),
]


embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(docs, embeddings)


similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)