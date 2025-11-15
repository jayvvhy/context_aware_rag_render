import os
import streamlit as st
from dotenv import load_dotenv

from document_indexer import DocumentIndexer
from rag_engine import ContextAwareRAG

# ---------------------------------------------
# Load environment variables
# ---------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 OPENAI_API_KEY is missing! Add it in Render → Environment Variables.")
    st.stop()


# ---------------------------------------------
# Load FAISS indexer (cached once per machine)
# ---------------------------------------------
@st.cache_resource
def load_indexer():
    indexer = DocumentIndexer(base_dir="artefacts", api_key=OPENAI_API_KEY)
    indexer.load_vector_store_from_disk()
    return indexer


# ---------------------------------------------
# Get a per-session RAG engine
# ---------------------------------------------
def get_rag():
    if "rag" not in st.session_state:
        indexer = load_indexer()
        st.session_state["rag"] = ContextAwareRAG(
            indexer=indexer,
            api_key=OPENAI_API_KEY
        )
        st.session_state["messages"] = []
    return st.session_state["rag"]


# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="Stanford Admin Guide RAG", page_icon="📘")
st.title("📘 Stanford Admin Guide Chatbot (Context-Aware RAG)")

rag = get_rag()

# ---- Clear chat history ----
if st.button("🧽 Clear History"):
    rag.history.clear()
    st.session_state["messages"] = []
    st.success("Chat history cleared.")


# ---- Show past messages ----
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---- User writes a prompt ----
prompt = st.chat_input("Ask a question about the Stanford Admin Guide")

if prompt:
    # User message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.query(prompt, k=3, return_context=True)
            answer = result["answer"]
            st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
