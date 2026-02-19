# Architecture Overview

## Goal
An AI-first chatbot that provides file-grounded Q&A (RAG) and maintains durable memory of high-signal facts.

## High-Level Flow

### 1. Ingestion (Upload → Parse → Chunk)
- **Inputs**: PDF, TXT, MD files via Streamlit UI.
- **Parsing**: `pypdf` for PDFs, standard file reading for text/markdown.
- **Chunking**: Text is split recursively into 1000-character chunks with 200-character overlap to maintain context.
- **Metadata**: Each chunk is tagged with its `source` filename and a `chunk_id`.

### 2. Indexing / Storage
- **Vector Database**: `ChromaDB` (Persistent Client) stores embeddings locally in `./chroma_db`.
- **Embeddings**: Uses Google's `models/gemini-embedding-001` via the `google-generativeai` library.
- **Persistence**: Data survives app restarts.

### 3. Retrieval + Grounded Answering
- **Retrieval**: Uses semantic search (cosine similarity) to find the top-3 most relevant chunks. Reduced from top-5 to optimize for Free Tier token limits.
- **Generation**:
    - **Model**: `models/gemini-2.0-flash-lite-001` (chosen for stability and free tier availability).
    - **Prompting**: System prompt instructs the model to answer *only* based on the retrieved context.
    - **Robustness**: Implements `tenacity` retry logic with exponential backoff to handle `429 Resource Exhausted` errors gracefully.
    - **Fallback**: If the API is completely overwhelmed, valuable error messages are displayed without crashing the UI.

### 4. Memory System (Selective)
- **Extraction**: A secondary LLM call analyzes every User/Assistant exchange.
- **High-Signal Protocol**:
    - Extracts `user_memory` (facts about the user).
    - Extracts `company_memory` (organizational learnings).
- **Storage**: Appends bullet points to `USER_MEMORY.md` and `COMPANY_MEMORY.md`.
- **User Control**: A "Enable Memory Extraction" toggle allows users to opt-in/out to manage API quota usage.

## Technical Decisions & Tradeoffs
- **Gemini Free Tier**: We chose `flash-lite` and reduced context size to work within strict rate limits (`15 RPM`, `1M TPM`).
- **Synchronous Memory**: Memory extraction happens after the answer. In a production app, this would be an async background task to improve latency.
- **Local Vectors**: ChromaDB is running in-process for simplicity. For scale, a client/server setup would be preferred.