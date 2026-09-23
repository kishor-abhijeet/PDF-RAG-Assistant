from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",#The character that should be used to split the text. If None, the text will be split on whitespace.
    chunk_size = 10,#The maximum number of characters in each chunk
    chunk_overlap = 1#the number of overlapping characters between chunks
)
data = TextLoader("document loaders/notes.txt")


docs = data.load()

chunks = splitter.split_documents(docs)
# print(len(chunks))
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
