from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load LLM
llm = OllamaLLM(model="llama3:8b")

# IMPORTANT: must match ingestion embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# If you re-indexed with bge-base-en, use that here instead

# Load vector DB
db = Chroma(
    persist_directory="college_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 8})

SYSTEM_PROMPT = """
You are a college information assistant.
Answer ONLY using the provided context.
If the context does not contain relevant information, reply:
"Information not found in the college database."
Do not add fallback text if partial information exists.

"""

print("College chatbot ready. Type 'exit' to quit.")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(query)

    if not docs:
        print("Bot: Information not found in the college database.")
        continue

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)
    print("Bot:", answer)
