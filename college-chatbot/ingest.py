import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

docs = []
for file in os.listdir("data"):
    loader = TextLoader(f"data/{file}", encoding="utf-8")
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="college_db"
)

db.persist()
print("Data indexed successfully")
