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
    def __init__(self, persist_directory: str = "chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Use a custom embedding function adapter for Gemini
        self.embedding_fn = self._create_gemini_embedding_fn()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_docs",
            embedding_function=self.embedding_fn
        )

    def _create_gemini_embedding_fn(self):
        """Creates a ChromaDB-compatible embedding function using Gemini."""
        # Define a wrapper class that matches Chroma's expectation
        class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                # Gemini embedding model
                model = "models/text-embedding-004"
                
                # Batch embedding if needed, but for simplicity here we loop or use batch support if available
                # genai.embed_content supports batching
                result = genai.embed_content(
                    model=model,
                    content=input,
                    task_type="retrieval_document"
                )
                return result['embedding']
        
        return GeminiEmbeddingFunction()

    def _get_query_embedding(self, text: str) -> List[float]:
        model = "models/text-embedding-004"
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
            # Create a deterministic ID
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

    def retrieve(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieves relevant documents for a query."""
        # We need to manually embed the query if using the custom class in a specific way,
        # but Chroma's collection.query automatically uses the embedding_function defined on the collection.
        # However, for Gemini, task_type differs for document vs query. 
        # So we might need to override query_embeddings.
        
        query_embedding = self._get_query_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
