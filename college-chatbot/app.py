from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

llm = Ollama(
    model="llama3:8b",
    system="""
    You are a college assistant chatbot.
    Answer only using the given college data.
    If information is not available, say you do not know.
    """
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="college_db",
    embedding_function=embeddings
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

print("College chatbot is ready")

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("Bot:", qa.run(q))
