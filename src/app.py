import streamlit as st
import os
import shutil
from rag_engine import RAGEngine
from llm_engine import LLMEngine

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")


if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "llm" not in st.session_state:
    st.session_state.llm = LLMEngine()
if "memory" not in st.session_state:
    from memory_manager import MemoryManager
    st.session_state.memory = MemoryManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.title("Settings")

    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

        st.session_state.rag = RAGEngine(api_key=api_key)
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Document (PDF/TXT/MD)", type=["pdf", "txt", "md"])
    if uploaded_file:
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            num_chunks = st.session_state.rag.ingest_file(file_path, uploaded_file.name)
            st.success(f"Ingested {num_chunks} chunks!")

    st.divider()
    enable_memory = st.checkbox("Enable Memory Extraction", value=False, help="Enable this to extract facts (Uses more API quota)")


st.title("Agentic RAG Assistant")


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    

    with st.spinner("Thinking..."):
        results = st.session_state.rag.retrieve(prompt)
        

        context_chunks = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        

        context_for_llm = []
        for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas)):
            context_for_llm.append(f"Source: {meta['source']}\nContent: {chunk}")

        if not context_for_llm:
             response = "I couldn't find any relevant information in the uploaded documents."
        else:
             try:
                response = st.session_state.llm.generate_response(prompt, context_for_llm)
             except Exception as e:
                response = f"Error generating response: {e}. Please check your API Key."
        

        with st.chat_message("assistant"):
            st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

     
        if enable_memory:
            with st.status("Updating Memory...", expanded=False) as status:
                try:
                   
                    import time
                    time.sleep(2) 
                    
                    memories = st.session_state.llm.extract_memory(
                        prompt, response
                    )
                    
                    if memories.get("user_memory"):
                        st.session_state.memory.update_user_memory(memories["user_memory"])
                        st.write(f"💾 Saved User Fact: {memories['user_memory']}")
                        
                    if memories.get("company_memory"):
                        st.session_state.memory.update_company_memory(memories["company_memory"])
                        st.write(f"🏢 Saved Company Learning: {memories['company_memory']}")
                    
                    if not memories.get("user_memory") and not memories.get("company_memory"):
                        st.write("No new memories extracted.")
                    
                    status.update(label="Memory Updated", state="complete")
                except Exception as e:
                    status.update(label="Memory Update Skipped (Rate Limit)", state="error")
                    st.write(f"Memory update skipped: {e}")
