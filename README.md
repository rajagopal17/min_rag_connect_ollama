# min_rag_connect_ollama

A minimal RAG (Retrieval-Augmented Generation) pipeline for querying SAP domain documents, using PostgreSQL + pgvector as the vector store and OpenAI for embeddings and generation.

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Embedding | `text-embedding-3-small` | OpenAI |
| Chat / Generation | `gpt-4o-mini` | OpenAI |
| Reranker | `ms-marco-MiniLM-L-12-v2` | FlashRank (local ONNX) |
| Alt embed (configured, unused in connector) | `nomic-embed-text:latest` | Ollama (local) |

---

## Connections

| System | Details |
|--------|---------|
| PostgreSQL | `localhost:5434` — database `business_partner` |
| pgvector table | `sap_cash_management` |
| OpenAI API | Embeddings + Chat completions |
| Ollama | `http://localhost:11434` (local LLM server) |

---

## Vector Index

**Index type:** HNSW (Hierarchical Navigable Small World) — provided by the `pgvector` extension.

**Distance metric:** Cosine distance (`<=>` operator)

**Vector dimension:** `1536` (matches `text-embedding-3-small` output)

**Runtime parameters set at query time:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `hnsw.ef_search` | `50` | Candidate list size during search — higher = more accurate, slower |
| `enable_seqscan` | `off` | Forces the planner to use the HNSW index instead of a sequential scan |

---

## Flow

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  OpenAI text-embedding-3-small  │  ← 1536-dim vector
└─────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  PostgreSQL + pgvector (HNSW index)      │
│  Table: sap_cash_management              │
│  Fetch TOP_K × 3 chunks (cosine search)  │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  FlashRank Reranker                      │
│  ms-marco-MiniLM-L-12-v2 (cross-encoder) │
│  Re-scores and selects TOP_K chunks      │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  System prompt built from ranked chunks  │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  OpenAI gpt-4o-mini                      │
│  temperature=0.2                         │
└──────────────────────────────────────────┘
     │
     ▼
   Answer
```

---

## Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_K` | `5` | Final number of chunks returned to the LLM |
| fetch multiplier | `TOP_K × 3` | Over-fetch before dedup + rerank |

---

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys and DB credentials.

3. Ensure PostgreSQL is running with the `pgvector` extension enabled and the `sap_cash_management` table populated with 1536-dim embeddings.

4. Run:
   ```bash
   python connector.py
   ```

---

## Stack

- **LangChain** — document loaders, text splitters, metadata handling
- **pgvector** — vector similarity search inside PostgreSQL
- **FlashRank** — lightweight local reranking (no GPU required)
- **FastAPI + Uvicorn** — REST API layer
- **python-dotenv** — environment configuration
