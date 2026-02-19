import os
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from typing import List, Dict, Any
import hashlib

# Configure Gemini
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class RAGEngine:
    def __init__(self, api_key: str = None, persist_directory: str = "chroma_db"):
        if api_key:
            genai.configure(api_key=api_key)
            
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Use a custom embedding function adapter for Gemini
        self.embedding_fn = self._create_gemini_embedding_fn()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_docs",
            embedding_function=self.embedding_fn
        )

    def _create_gemini_embedding_fn(self):
        """Creates a ChromaDB-compatible embedding function using Gemini."""
       
        class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
               
                model = "models/gemini-embedding-001"
                
              
                result = genai.embed_content(
                    model=model,
                    content=input,
                    task_type="retrieval_document"
                )
                return result['embedding']
        
        return GeminiEmbeddingFunction()

    def _get_query_embedding(self, text: str) -> List[float]:
        model = "models/gemini-embedding-001"
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']

    def ingest_file(self, file_path: str, original_filename: str) -> int:
        """Reads file, chunks it, and adds to ChromaDB. Returns number of chunks."""
        text = ""
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else: # txt, md
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        chunks = self._chunk_text(text)
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
           
            chunk_id = hashlib.md5(f"{original_filename}_{i}".encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": original_filename,
                "chunk_id": i
            })
            
        if documents:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        return len(documents)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple helper to chunk text with overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
            
        return chunks

    def retrieve(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Retrieves relevant documents for a query."""

        
        query_embedding = self._get_query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
