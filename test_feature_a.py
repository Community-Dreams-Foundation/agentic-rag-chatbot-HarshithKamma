import os
from src.rag_engine import RAGEngine
from src.llm_engine import LLMEngine
from dotenv import load_dotenv


load_dotenv()

def test_feature_a():
    print("Testing Feature A...")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("SKIPPING TEST: No GOOGLE_API_KEY found in environment.")
        return


    rag = RAGEngine(persist_directory="test_chroma_db")
    llm = LLMEngine()

   
    print("\n[1/3] Ingesting test document...")
    with open("test_doc.txt", "w") as f:
        f.write("The capital of Mars is not defined because nobody lives there yet. But Elon wants to build a city there.")
    
    rag.ingest_file("test_doc.txt", "test_doc.txt")
    print("Ingestion complete.")

    print("\n[2/3] Retrieving context...")
    query = "What is the capital of Mars?"
    results = rag.retrieve(query)
    docs = results['documents'][0]
    print(f"Retrieved: {docs}")
    
    if not docs:
        print("FAIL: No documents retrieved.")
        return

    # 4. Generate
    print("\n[3/3] Generating answer...")
    context_list = [f"Content: {d}" for d in docs]
    answer = llm.generate_response(query, context_list)
    print(f"Answer: {answer}")
    
    # Cleanup
    if os.path.exists("test_doc.txt"):
        os.remove("test_doc.txt")
    
    print("\nFeature A Test Completed.")

if __name__ == "__main__":
    test_feature_a()
