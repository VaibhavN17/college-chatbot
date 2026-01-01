import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Page setup
st.set_page_config(page_title="College Chatbot", layout="centered")
st.title("🎓 College Chatbot")

# Load models
llm = OllamaLLM(model="llama3:8b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="college_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 8})

SYSTEM_PROMPT = """
You are a college information assistant.
Answer ONLY using the provided context.
If information is not found, say:
"Information not found in the college database."
"""

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask something about the college...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    docs = retriever.invoke(query)

    if not docs:
        answer = "Information not found in the college database."
    else:
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""
        answer = llm.invoke(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
