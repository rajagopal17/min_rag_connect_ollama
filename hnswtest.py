import os
import sys
import psycopg2
from typing import List
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest # pyright: ignore[reportMissingImports]
import ollama
from pgvector import Vector

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
    query_vec = response.data[0].embedding
    return query_vec

# ----------------------------
# RETrIEVE  
# ----------------------------
def run_hnsw_search(query_vec: List[float], top_k: int = TOP_K, ef_search: int = 100) -> List[tuple]:
    """
    Perform HNSW-based vector similarity search using pgvector.

    Args:
        query_embedding (List[float]): The query embedding vector.
        ef_search (int): HNSW search parameter controlling recall vs speed.
        top_k (int): Number of nearest neighbors to retrieve.

    Returns:
        List[Tuple]: Query results containing (id, content, similarity_score).
    """
       
    # Convert embedding to pgvector format  
    query_vec_pg = Vector(query_vec)  
    # establish database connection
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)  # Ensure pgvector is registered  
    cur = conn.cursor()
    # Set HNSW search parameters
    #cur.execute("SET enable_seqscan = off;")  # Force index usage
    cur.execute(f"SET hnsw.ef_search = {ef_search};")  # Adjust for recall/speed tradeoff
    # Execute similarity search query   
    cur.execute(f"""
        
        SELECT id, 
               content, 
               file_name,
               embedding <=> %s::vector AS similarity_score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_vec_pg, query_vec_pg, top_k))
    results = cur.fetchall()
    # Clean up database connection
    cur.close()
    conn.close()
    return results

# function to join content and metadata into a single string to pass it to llm
def format_results_for_llm(results: List[tuple]) -> str:
    formatted = []
    for idx, content, file_name, score in results:
        formatted.append(f"Content: {content}\nFile Name: {file_name}\nSimilarity Score: {score:.4f}\n{'-'*40}")
    return "\n".join(formatted)

# function to pass above formatted string to reranker and get back the top_k results sorted by relevance to the query
def rerank_results(query: str, formatted_results: str) -> List[tuple]:
    ranker = Ranker(model_name=RERANK_MODEL)
    passages = [{"id": i, "text": res, "meta": {}} for i, res in enumerate(formatted_results.split("\n" + "-"*40 + "\n"))]
    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)
    return [(r["text"], r["score"]) for r in ranked]
# function to generate prompt for llm using the top_k reranked results

def build_prompt(reranked_results: List[tuple]) -> str:
    prompt = "Answer the following question STRICTLY based on the provided context:\n\n"
    prompt += "Question: {query}\n\n"
    prompt += "Context:\n"
    for i, (content, score) in enumerate(reranked_results):
        prompt += f"[Passage {i+1} | Relevance Score: {score:.4f}]\n{content}\n\n"
    prompt += "Answer:"
    return prompt

# function to generate answer using openai llm given the prompt and the chat model
def generate_answer(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": query}]
    )
    return response.choices[0].message.content.strip()






#create a function to run explain analyze on the hnsw search query to verify index usage and understand query performance
""" def explain_hnsw_search(query_vec: List[float], top_k: int = TOP_K, ef_search: int = 50) -> str:
    
    Run EXPLAIN ANALYZE on the HNSW search query to verify index usage and understand performance.

    Args:
        query_embedding (List[float]): The query embedding vector.
        ef_search (int): HNSW search parameter controlling recall vs speed.
        top_k (int): Number of nearest neighbors to retrieve.

    Returns:
        str: The EXPLAIN ANALYZE output.
   
    query_vec_pg = Vector(query_vec)
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    cur.execute(f
        EXPLAIN ANALYZE
        SELECT id, content, embedding <=> %s::vector AS similarity_score
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    , (query_vec_pg, query_vec_pg, top_k))
    explain_output = "\n".join(row[0] for row in cur.fetchall())
    cur.close()
    conn.close()
    return explain_output """


# Run a test search
if __name__ == "__main__":
    query = "how asset under construction is transformed to final asset?"
    query_embedding = get_embedding(query)

    search_results = run_hnsw_search(query_embedding)
    context_for_llm = format_results_for_llm(search_results)
    reranked_results = rerank_results(query, context_for_llm)
    prompt = build_prompt(reranked_results)
    answer = generate_answer(prompt)

    print("Generated Answer:")
    print(answer)

    # print only content and similarity score for each result
    #for idx, content,file_name, score in search_results:
    #    print(f"Result ID: {idx}, File Name: {file_name}, Similarity Score: {score:.4f}\nContent: {content}\n{'-'*80}\n")

    # Run EXPLAIN ANALYZE to verify index usage
    #explain_output = explain_hnsw_search(query_embedding)
    #print("EXPLAIN ANALYZE Output:")
    #print(explain_output) """
