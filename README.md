# Agentic AI RAG Chatbot (LangGraph + Pinecone)

A Retrieval-Augmented Generation chatbot that answers **only** from the
*Agentic AI for Executives* eBook (`data/Ebook-Agentic-AI.pdf`), built in custom Python.

## Architecture

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter (1000 / 200 overlap)
    -> Gemini embeddings (gemini-embedding-001) -> Pinecone (cosine, metadata: text + page)

Query -> [retrieve] -> [generate] -> [grade] -> response
              |            |
       top-k chunks   refuses if context is irrelevant (skips grading)
```

| Component | Tech | Responsibility |
| --- | --- | --- |
| Ingestion | PyPDF, LangChain splitter | Load PDF, chunk with overlap, keep page numbers |
| Embeddings + Vector DB | Google Gemini embeddings + Pinecone | Dense vectors and semantic search |
| Orchestration | LangGraph `StateGraph` | `retrieve` -> `generate` -> `grade` with a conditional edge |
| API | FastAPI | `POST /chat` returns the structured payload |

**Groundedness / confidence:**
1. Out-of-scope guard: if the best chunk's cosine similarity is below `MIN_RELEVANCE` (0.30), the bot refuses.
2. The generate prompt forces answers from context only.
3. The `grade` node asks the LLM to rate how well the answer is supported by the context (0-1).
4. `confidence_score = 0.4 * retrieval_similarity + 0.6 * groundedness`. Refusals get `0.0`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GOOGLE_API_KEY (free Gemini key) and PINECONE_API_KEY
```

Models are configurable via `EMBEDDING_MODEL` and `LLM_MODEL` in `.env` (defaults: `models/gemini-embedding-001`, `gemini-2.5-flash`).

## Run

```bash
python -m src.ingestion                 # 1. build the Pinecone index (once)
uvicorn app:app --reload                # 2. start the API on http://127.0.0.1:8000
python tests_sample_queries.py          # 3. run the 6 sample queries
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
     -d '{"query": "What is Agentic AI?"}'
```

Response format:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["chunk 1...", "chunk 2..."],
  "confidence_score": 0.87
}
```

## Project structure

```
rag-agentic-ai/
├── data/Ebook-Agentic-AI.pdf
├── src/{config.py, ingestion.py, graph.py}
├── app.py
├── tests_sample_queries.py
├── requirements.txt
└── .env.example
```
