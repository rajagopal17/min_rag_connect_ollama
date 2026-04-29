import os
import sys
import psycopg2
from typing import List
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest # pyright: ignore[reportMissingImports]
import ollama

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# ----------------------------
# CONFIG
# ----------------------------
DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", "5434")),
    "dbname":   os.getenv("PG_DB"),
    "user":     os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
}

TABLE        = os.getenv("PG_TABLE")
TOP_K        = int(os.getenv("TOP_K", "5"))
OLLAMA_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL   = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# EMBEDDING  (OpenAI — must match 1536-dim vectors stored in DB)
# ----------------------------
def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

# ----------------------------
# RETRIEVE
# ----------------------------
def retrieve(query_vec):
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET hnsw.ef_search = 50;")
    # fetch extra rows to absorb duplicates after dedup
    fetch_k = TOP_K * 3
    cur.execute(f"""
        SELECT content, source, file_name, doc_category, chunk_index,
               embedding <=> %s::vector AS score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT {fetch_k};
    """, (query_vec, query_vec))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    # deduplicate by content; build (content, metadata, score) tuples
    seen, results = set(), []
    for content, source, file_name, doc_category, chunk_index, score in rows:
        if content not in seen:
            seen.add(content)
            metadata = {"source": source or file_name or "", "page": chunk_index, "category": doc_category}
            results.append((content, metadata, score))
        if len(results) == TOP_K:
            break
    return results

# ----------------------------
# RERANK
# ----------------------------
def rerank(query: str, chunks: list) -> list:
    ranker = Ranker(model_name=RERANK_MODEL)
    passages = [{"id": i, "text": chunk[0], "meta": chunk[1]} for i, chunk in enumerate(chunks)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    return [(r["text"], r["meta"], r["score"]) for r in results]

# ----------------------------
# PROMPT
# ----------------------------
def build_prompt(chunks):
    context = "\n\n".join(
        f"[Chunk {i+1} | page={c[1].get('page','?')}]\n{c[0]}"
        for i, c in enumerate(chunks)
    )
    return f"""You are an expert SAP consultant. Always answer using the reference material below.
Synthesize information across all chunks. Do not say "I don't know" if the answer can be inferred from the material.
Only say "I don't know" if the topic is genuinely absent from the context.

Reference material:
{context}
"""

# ----------------------------
# GENERATE  (OpenAI LLM)
# ----------------------------
def generate(prompt, question):
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ----------------------------
def main():
    question = "explain how asset under construction is accounted for in SAP?"
    query_vec = get_embedding(question)
    chunks = retrieve(query_vec)
    chunks = rerank(question, chunks)
    prompt = build_prompt(chunks)
    answer = generate(prompt, question)

    print("\n" + "="*60, flush=True)
    print("ANSWER", flush=True)
    print("="*60, flush=True)
    print(answer, flush=True)

    print("\n" + "="*60, flush=True)
    print("RERANKED CHUNKS", flush=True)
    print("="*60, flush=True)
    for i, (text, metadata, score) in enumerate(chunks):
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        page   = metadata.get("page", "?")          if isinstance(metadata, dict) else "?"
        source = metadata.get("source", "?").split("\\")[-1] if isinstance(metadata, dict) else "?"
        print(f"\n[{i+1}] score={score:.4f} | page={page} | {source}", flush=True)
        print("-"*50, flush=True)
        print(safe_text.strip(), flush=True)

if __name__ == "__main__":
    main()
