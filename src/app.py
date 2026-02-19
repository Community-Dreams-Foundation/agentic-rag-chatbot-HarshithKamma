import streamlit as st
import os
import shutil
from rag_engine import RAGEngine
from llm_engine import LLMEngine

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# Initialize session state for engines
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "llm" not in st.session_state:
    st.session_state.llm = LLMEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # API Key handling
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        # Re-configure engines if key changes (simple reload equivalent)
    
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

# Main Chat Interface
st.title("Agentic RAG Assistant")

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Retrieve
    with st.spinner("Thinking..."):
        results = st.session_state.rag.retrieve(prompt)
        
        # Extract meaningful context from Chroma results
        # results['documents'] is List[List[str]]
        context_chunks = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        # Prepare context with citations
        context_for_llm = []
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):
            context_for_llm.append(f"Source: {meta['source']}\nContent: {chunk}")

        # Generate
        if not context_for_llm:
             response = "I couldn't find any relevant information in the uploaded documents."
        else:
             try:
                response = st.session_state.llm.generate_response(prompt, context_for_llm)
             except Exception as e:
                response = f"Error generating response: {e}. Please check your API Key."
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
