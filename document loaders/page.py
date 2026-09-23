#web page loader
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/shop/buy-iphone/iphone-18-pro/6.9%22-display-256gb-glacier"

data = WebBaseLoader(url)
docs =  data.load()
print(docs[0].page_content)