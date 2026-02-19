import streamlit as st
import os
import shutil
from rag_engine import RAGEngine

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# Initialize session state for engines
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # API Key handling
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Document (PDF/TXT/MD)", type=["pdf", "txt", "md"])
    if uploaded_file:
        # Save file locally
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            num_chunks = st.session_state.rag.ingest_file(file_path, uploaded_file.name)
            st.success(f"Ingested {num_chunks} chunks!")

# Main Content Placeholder
st.title("Agentic RAG Assistant")
st.write("Upload a file in the sidebar to get started.")
