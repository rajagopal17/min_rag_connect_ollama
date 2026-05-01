"""
eval_harness.py — RAG Evaluation Harness for min_rag_connect_ollama
====================================================================
Evaluates your REAL pipeline (pgvector → rerank → OpenAI) against a
golden dataset of SAP Treasury questions.

HOW TO RUN:
    python eval_harness.py               # full eval
    python eval_harness.py --list-files  # show file_names in your DB
    python eval_harness.py --sample      # print 3 sample chunks to help
                                         # you extend the golden dataset

CUSTOMIZE:
    Edit GOLDEN_DATASET below to add/remove questions that match your
    actual documents. Use --list-files to find your real file_name values.
"""

import os
import sys
import argparse
import psycopg2
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

# ── Import the real pipeline ───────────────────────────────────────────────
from connector import (
    get_embedding, retrieve, rerank,
    build_prompt, generate,
    openai_client, CHAT_MODEL, DB_CONFIG, TABLE,
)
from pgvector.psycopg2 import register_vector


# ─────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class GoldenItem:
    question:        str
    ground_truth:    str
    # Substrings that must appear in a truly relevant chunk's content.
    # A retrieved chunk counts as "relevant" if it contains ANY of these
    # keywords (case-insensitive). Keep them specific enough to avoid
    # false positives. Use --sample to see real chunk content.
    relevant_keywords: List[str]
    category:        str   # factual | procedural | comparative | analytical | adversarial


# ─────────────────────────────────────────────────────────────────────────
#  GOLDEN DATASET  — SAP Treasury & Risk Management (S4F50 / SAPTRM.pdf)
#  Edit these to match questions real users ask your system.
# ─────────────────────────────────────────────────────────────────────────
GOLDEN_DATASET: List[GoldenItem] = [

    # PROCEDURAL
  
    GoldenItem(
    "How do you change time-dependent data for an asset in SAP?",
    "Go to Asset → Change → Asset, choose the Time-dependent data tab, then choose Further Intervals to select or define a new time interval, and save.",
    ["time-dependent", "further intervals", "tab", "interval"],
    "procedural",
),
GoldenItem(
    "What are the two methods for changing a balance-sheet-relevant organizational unit on an asset?",
    "Method 1: Change directly in the master data transaction (only if business area and cost center are not time-dependent — the system auto-creates a transfer document). Method 2: Post an asset transfer to a new master record (required when those fields are time-dependent).",
    ["master data transaction", "transfer", "time-dependent", "business area", "cost center"],
    "comparative",
),
GoldenItem(
    "How do you change the asset class of an asset in SAP?",
    "Changing the asset class is only possible by posting a transfer to a new asset master record. Copy the original record as a reference, assign the new asset class, then carry out a complete transfer using transaction type 300/320 for prior-year and current-year acquisitions separately.",
    ["asset class", "transfer", "master record", "transaction type 300"],
    "procedural",
),
GoldenItem(
    "What is the purpose of the blocking indicator in the SAP asset master record?",
    "The blocking indicator prevents future acquisition postings to the asset. It is typically used when an asset under construction project is complete. Transfer and retirement postings to blocked assets are still allowed.",
    ["blocking indicator", "acquisitions", "asset under construction", "blocked"],
    "factual",

    ),

    # ANALYTICAL
    GoldenItem(
        "Why is the HNSW index used for vector search in this RAG system?",
        "UNANSWERABLE — not in the SAP knowledge base",
        [],
        "adversarial",
    ),
    GoldenItem(
        "What is the weather in Dubai today?",
        "UNANSWERABLE — not in the SAP knowledge base",
        [],
        "adversarial",
    ),
    GoldenItem(
        "Tell me about competitor treasury software pricing",
        "UNANSWERABLE — not in the SAP knowledge base",
        [],
        "adversarial",
    ),

    # FX RISK MANAGEMENT
    GoldenItem(
        "What are the four FX product types handled in SAP TRM?",
        "SAP TRM handles FX Spot (immediate currency exchange), FX Forward (future exchange at a fixed rate), FX Swap (exchange and reverse after a period), and FX Options (right but not obligation to buy/sell at a future rate).",
        ["FX spot", "FX forward", "FX swap", "FX option"],
        "factual",
    ),
    GoldenItem(
        "How is the forward rate calculated for an FX forward transaction in SAP?",
        "The forward rate is made up of the spot rate plus the swap rate. The spot rate is the current exchange rate when the trade was executed; the swap rate is a markup or markdown based on the interest rate difference between the two currencies.",
        ["forward rate", "spot rate", "swap rate", "interest rate difference"],
        "factual",
    ),
    GoldenItem(
        "What transaction code is used to create an FX Spot or Forward in SAP?",
        "Transaction code TX01 is used to create an FX Spot or Forward contract in SAP Transaction Manager.",
        ["TX01", "FX spot", "FX forward", "transaction manager"],
        "factual",
    ),
    GoldenItem(
        "What are the two settlement types available for FX transactions in SAP, and how do they differ?",
        "Physical Delivery posts both the buy and sell amounts into FI at the maturity of the trade. Cash Settlement posts only a net settlement amount into FI at maturity.",
        ["physical delivery", "cash settlement", "maturity", "settlement"],
        "comparative",
    ),
    GoldenItem(
        "What organizational roles are involved in FX trade processing in SAP Transaction Manager?",
        "Trade processing follows a Treasury organizational structure: the Front-Office executes trades, the Back-Office handles confirmations and payments, and Accounting manages postings to the SAP General Ledger including accruals and valuations.",
        ["front-office", "back-office", "accounting", "general ledger", "confirmation"],
        "factual",
    ),
]


# ─────────────────────────────────────────────────────────────────────────
#  RELEVANCE CHECK  (content-keyword based, works without exact doc IDs)
# ─────────────────────────────────────────────────────────────────────────
def is_relevant(chunk_content: str, keywords: List[str]) -> bool:
    """A chunk is relevant if its content contains ANY of the keywords."""
    text = chunk_content.lower()
    return any(kw.lower() in text for kw in keywords)


def retrieved_relevance_flags(chunks: list, keywords: List[str]) -> List[bool]:
    return [is_relevant(c[0], keywords) for c in chunks]


# ─────────────────────────────────────────────────────────────────────────
#  RETRIEVAL METRICS
# ─────────────────────────────────────────────────────────────────────────
def precision_at_k(flags: List[bool], k: int) -> float:
    return sum(flags[:k]) / k if k > 0 else 0.0

def recall_at_k(flags: List[bool], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 1.0  # adversarial — vacuously true
    return sum(flags[:k]) / total_relevant

def mrr(flags: List[bool]) -> float:
    return next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)


# ─────────────────────────────────────────────────────────────────────────
#  GENERATION METRICS  (LLM-as-judge via OpenAI)
# ─────────────────────────────────────────────────────────────────────────
def score_faithfulness(answer: str, context: str) -> float:
    """Fraction of answer claims supported by the retrieved context."""
    prompt = f"""You are an evaluation judge. Given the context below, score how faithful the answer is.
Faithfulness = fraction of answer claims that are supported by the context (0.0 to 1.0).
Reply with a single decimal number only (e.g. 0.85).

Context:
{context[:3000]}

Answer:
{answer}

Score:"""
    try:
        raw = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).choices[0].message.content.strip()
        return float(raw)
    except Exception:
        return 0.5


def score_answer_relevancy(answer: str, question: str) -> float:
    """How well does the answer address the question (0.0 to 1.0)?"""
    prompt = f"""You are an evaluation judge. Score how well the answer addresses the question.
Answer Relevancy = how directly and completely the answer responds to the question (0.0 to 1.0).
Reply with a single decimal number only (e.g. 0.90).

Question: {question}
Answer: {answer}

Score:"""
    try:
        raw = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        ).choices[0].message.content.strip()
        return float(raw)
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────
#  RAW RETRIEVE  (no distance threshold — used by --debug and --tune)
# ─────────────────────────────────────────────────────────────────────────
def raw_retrieve(query_vec, fetch_k: int = 15) -> list:
    """Return chunks with their raw vector distances, ignoring DIST_THRESHOLD."""
    from connector import DB_CONFIG, TABLE
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET hnsw.ef_search = 100;")
    cur.execute(f"""
        SELECT content, file_name, chunk_index,
               embedding <=> %s::vector AS dist
        FROM {TABLE}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_vec, query_vec, fetch_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    seen, results = set(), []
    for content, file_name, chunk_index, dist in rows:
        if content not in seen:
            seen.add(content)
            results.append((content, file_name or "", chunk_index, dist))
    return results


# ─────────────────────────────────────────────────────────────────────────
#  DEBUG MODE  — show raw scores for a single question
# ─────────────────────────────────────────────────────────────────────────
def debug_query(question: str):
    """Show vector distance + reranker score for every candidate chunk."""
    from connector import DIST_THRESHOLD, RERANK_THRESHOLD
    from flashrank import Ranker, RerankRequest
    from connector import RERANK_MODEL

    print(f"\n  Question: {question}")
    print(f"  Current DIST_THRESHOLD={DIST_THRESHOLD}  RERANK_THRESHOLD={RERANK_THRESHOLD}\n")

    query_vec = get_embedding(question)
    raw       = raw_retrieve(query_vec, fetch_k=15)

    passages = [{"id": i, "text": r[0], "meta": {}} for i, r in enumerate(raw)]
    from flashrank import RerankRequest
    ranker  = Ranker(model_name=RERANK_MODEL)
    ranked  = ranker.rerank(RerankRequest(query=question, passages=passages))
    rerank_scores = {r["id"]: r["score"] for r in ranked}

    print(f"  {'#':<3} {'VecDist':<9} {'ReRank':<9} {'Threshold':<12} {'File':<30} {'Content (first 80 chars)'}")
    print(f"  {'─' * 110}")

    for i, (content, file_name, chunk_idx, dist) in enumerate(raw):
        r_score  = rerank_scores.get(i, 0.0)
        dist_ok  = dist <= DIST_THRESHOLD
        rank_ok  = r_score >= RERANK_THRESHOLD
        if dist_ok and rank_ok:
            gate = "✅ PASS both"
        elif dist_ok:
            gate = "⚠️  fails rerank"
        elif rank_ok:
            gate = "⚠️  fails dist"
        else:
            gate = "❌ fails both"
        short_f = str(file_name).split("\\")[-1][:28]
        short_c = content.replace("\n", " ")[:78]
        print(f"  {i+1:<3} {dist:<9.4f} {r_score:<9.4f} {gate:<14} {short_f:<30} {short_c}")

    print(f"\n  HOW TO READ THIS:")
    print(f"  → Chunks you want in context should show ✅ PASS both")
    print(f"  → If good chunks show ⚠️  fails dist, RAISE DIST_THRESHOLD")
    print(f"  → If bad chunks show ✅ PASS both, LOWER DIST_THRESHOLD or RAISE RERANK_THRESHOLD")
    print(f"\n  Set thresholds in .env:  DIST_THRESHOLD=0.xx   RERANK_THRESHOLD=0.x")


# ─────────────────────────────────────────────────────────────────────────
#  TUNE MODE  — suggest optimal thresholds across the full golden dataset
# ─────────────────────────────────────────────────────────────────────────
def tune_thresholds():
    """
    For each answerable golden item:
      - relevant distances  = dist of chunks whose content matches relevant_keywords
      - irrelevant distances = dist of chunks that do NOT match
    Suggests a DIST_THRESHOLD that separates the two populations.
    """
    from flashrank import Ranker, RerankRequest
    from connector import RERANK_MODEL

    answerable = [g for g in GOLDEN_DATASET if g.category != "adversarial"]
    print(f"\n  Analysing {len(answerable)} questions (no API cost — uses raw distances only)...\n")

    rel_dists, irrel_dists = [], []
    rel_rscores, irrel_rscores = [], []

    ranker = Ranker(model_name=RERANK_MODEL)

    for item in answerable:
        query_vec = get_embedding(item.question)
        raw       = raw_retrieve(query_vec, fetch_k=15)

        passages  = [{"id": i, "text": r[0], "meta": {}} for i, r in enumerate(raw)]
        ranked    = ranker.rerank(RerankRequest(query=item.question, passages=passages))
        rscores   = {r["id"]: r["score"] for r in ranked}

        for i, (content, _, _, dist) in enumerate(raw):
            rs = rscores.get(i, 0.0)
            if is_relevant(content, item.relevant_keywords):
                rel_dists.append(dist)
                rel_rscores.append(rs)
            else:
                irrel_dists.append(dist)
                irrel_rscores.append(rs)

    if not rel_dists:
        print("  No relevant chunks found — check your relevant_keywords in GOLDEN_DATASET.")
        return

    rel_max_dist   = max(rel_dists)
    irrel_min_dist = min(irrel_dists) if irrel_dists else 1.0
    rel_min_rscore = min(rel_rscores)
    irrel_max_rscore = max(irrel_rscores) if irrel_rscores else -99

    suggested_dist   = round((rel_max_dist + irrel_min_dist) / 2, 3)
    suggested_rerank = round((rel_min_rscore + irrel_max_rscore) / 2, 3)

    print(f"  VECTOR DISTANCE ANALYSIS:")
    print(f"  {'─' * 55}")
    print(f"  Relevant chunks    — avg: {sum(rel_dists)/len(rel_dists):.3f}  max: {rel_max_dist:.3f}  min: {min(rel_dists):.3f}")
    print(f"  Irrelevant chunks  — avg: {sum(irrel_dists)/len(irrel_dists):.3f}  min: {irrel_min_dist:.3f}  max: {max(irrel_dists):.3f}")
    print(f"\n  RERANKER SCORE ANALYSIS:")
    print(f"  {'─' * 55}")
    print(f"  Relevant chunks    — avg: {sum(rel_rscores)/len(rel_rscores):.3f}  min: {rel_min_rscore:.3f}  max: {max(rel_rscores):.3f}")
    print(f"  Irrelevant chunks  — avg: {sum(irrel_rscores)/len(irrel_rscores):.3f}  max: {irrel_max_rscore:.3f}  min: {min(irrel_rscores):.3f}")

    gap_dist   = irrel_min_dist - rel_max_dist
    gap_rerank = rel_min_rscore - irrel_max_rscore

    print(f"\n  SUGGESTED THRESHOLDS:")
    print(f"  {'─' * 55}")
    if gap_dist > 0:
        print(f"  DIST_THRESHOLD   = {suggested_dist}  (gap between populations: {gap_dist:.3f})")
    else:
        print(f"  DIST_THRESHOLD   = {rel_max_dist + 0.02:.3f}  ⚠️  populations overlap — reranker is your main filter")
    if gap_rerank > 0:
        print(f"  RERANK_THRESHOLD = {suggested_rerank}  (gap between populations: {gap_rerank:.3f})")
    else:
        print(f"  RERANK_THRESHOLD = 0.0  ⚠️  reranker populations overlap — rely on distance filter")

    print(f"\n  Add these to your .env file, then re-run the full eval to confirm improvement.")


# ─────────────────────────────────────────────────────────────────────────
#  DB HELPERS
# ─────────────────────────────────────────────────────────────────────────
def list_db_files():
    """Print distinct file_name values in the vector table."""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT file_name, COUNT(*) as chunks FROM {TABLE} GROUP BY file_name ORDER BY chunks DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"\n  {'File Name':<60} {'Chunks'}")
    print(f"  {'─' * 70}")
    for file_name, count in rows:
        print(f"  {str(file_name):<60} {count}")


def sample_db_chunks(n: int = 3):
    """Print n sample chunks to help build the golden dataset."""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT content, file_name, doc_category FROM {TABLE} ORDER BY RANDOM() LIMIT {n};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    for i, (content, file_name, category) in enumerate(rows, 1):
        print(f"\n  ── Sample {i} ── file={file_name} | category={category}")
        print(f"  {content[:400]}{'...' if len(content) > 400 else ''}")


# ─────────────────────────────────────────────────────────────────────────
#  MODULE 1: RETRIEVAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────
def module1_retrieval():
    print("\n╔" + "═" * 62 + "╗")
    print("║  MODULE 1: RETRIEVAL EVALUATION (real pgvector pipeline)   ║")
    print("╚" + "═" * 62 + "╝")

    answerable = [g for g in GOLDEN_DATASET if g.category != "adversarial"]
    print(f"\n  Evaluating {len(answerable)} answerable questions against your DB...\n")

    K = 5
    p_scores, r_scores, m_scores = [], [], []
    category_results = {}

    print(f"  {'Category':<14} {'Question':<48} {'P@5':<6} {'R@5':<6} {'MRR'}")
    print(f"  {'─' * 85}")

    for item in answerable:
        query_vec = get_embedding(item.question)
        chunks    = retrieve(query_vec)
        chunks    = rerank(item.question, chunks)

        flags     = retrieved_relevance_flags(chunks, item.relevant_keywords)
        # total_relevant: max chunks in the DB that could match (cap at K for recall)
        total_rel = min(sum(1 for c in chunks if is_relevant(c[0], item.relevant_keywords)) or 1, K)

        p = precision_at_k(flags, K)
        r = recall_at_k(flags, total_rel, K)
        m = mrr(flags)

        p_scores.append(p)
        r_scores.append(r)
        m_scores.append(m)

        cat = item.category
        category_results.setdefault(cat, {"p": [], "r": [], "m": []})
        category_results[cat]["p"].append(p)
        category_results[cat]["r"].append(r)
        category_results[cat]["m"].append(m)

        short_q = item.question[:46]
        print(f"  {cat:<14} {short_q:<48} {p:.2f}   {r:.2f}   {m:.2f}")

    avg_p = sum(p_scores) / len(p_scores)
    avg_r = sum(r_scores) / len(r_scores)
    avg_m = sum(m_scores) / len(m_scores)

    print(f"\n  {'─' * 85}")
    print(f"  {'OVERALL AVERAGE':<63} {avg_p:.3f}  {avg_r:.3f}  {avg_m:.3f}")

    print(f"\n  PER-CATEGORY SUMMARY:\n")
    print(f"  {'Category':<14} {'N':<4} {'Avg P@5':<10} {'Avg R@5':<10} {'Avg MRR':<10} Status")
    print(f"  {'─' * 60}")
    for cat, vals in category_results.items():
        n   = len(vals["p"])
        ap  = sum(vals["p"]) / n
        ar  = sum(vals["r"]) / n
        am  = sum(vals["m"]) / n
        ok  = "✅ Good" if ap >= 0.6 else "⚠️  Needs work"
        print(f"  {cat:<14} {n:<4} {ap:<10.3f} {ar:<10.3f} {am:<10.3f} {ok}")

    return {"P@5": avg_p, "R@5": avg_r, "MRR": avg_m}


# ─────────────────────────────────────────────────────────────────────────
#  MODULE 2: GENERATION EVALUATION
# ─────────────────────────────────────────────────────────────────────────
def module2_generation():
    print("\n╔" + "═" * 62 + "╗")
    print("║  MODULE 2: GENERATION EVALUATION (faithfulness + relevancy) ║")
    print("╚" + "═" * 62 + "╝")

    # Run on a subset to keep API cost low (first 5 answerable items)
    answerable = [g for g in GOLDEN_DATASET if g.category != "adversarial"][:5]
    print(f"\n  Evaluating {len(answerable)} questions (full generate → judge loop)...\n")
    print(f"  {'Question':<48} {'Faithfulness':<14} {'Relevancy'}")
    print(f"  {'─' * 75}")

    faith_scores, relev_scores = [], []

    for item in answerable:
        query_vec = get_embedding(item.question)
        chunks    = retrieve(query_vec)
        chunks    = rerank(item.question, chunks)
        prompt    = build_prompt(chunks)
        answer    = generate(prompt, item.question)

        context   = "\n".join(c[0] for c in chunks)
        faith     = score_faithfulness(answer, context)
        relev     = score_answer_relevancy(answer, item.question)

        faith_scores.append(faith)
        relev_scores.append(relev)

        short_q = item.question[:46]
        f_icon  = "✅" if faith >= 0.8 else "⚠️ "
        r_icon  = "✅" if relev >= 0.8 else "⚠️ "
        print(f"  {short_q:<48} {f_icon} {faith:.2f}        {r_icon} {relev:.2f}")

    avg_faith = sum(faith_scores) / len(faith_scores)
    avg_relev = sum(relev_scores) / len(relev_scores)

    print(f"\n  {'─' * 75}")
    print(f"  {'AVERAGE':<48} {avg_faith:.3f}          {avg_relev:.3f}")

    return {"faithfulness": avg_faith, "answer_relevancy": avg_relev}


# ─────────────────────────────────────────────────────────────────────────
#  MODULE 3: ADVERSARIAL CHECK
# ─────────────────────────────────────────────────────────────────────────
def module3_adversarial():
    print("\n╔" + "═" * 62 + "╗")
    print("║  MODULE 3: ADVERSARIAL / OUT-OF-SCOPE DETECTION            ║")
    print("╚" + "═" * 62 + "╝")

    adv_items = [g for g in GOLDEN_DATASET if g.category == "adversarial"]
    print(f"\n  Checking {len(adv_items)} out-of-scope questions...\n")
    print(f"  {'Question':<50} {'Retrieved top chunk (first 60 chars)'}")
    print(f"  {'─' * 90}")

    hallucination_risk = 0
    for item in adv_items:
        query_vec = get_embedding(item.question)
        chunks    = retrieve(query_vec)
        top_text  = chunks[0][0][:60].replace("\n", " ") if chunks else "(no results)"
        score     = chunks[0][2] if chunks else 1.0
        risk      = "⚠️  LOW score — may hallucinate" if score < 0.5 else "✅ High distance — safe"
        if score < 0.5:
            hallucination_risk += 1
        short_q = item.question[:48]
        print(f"  {short_q:<50} dist={score:.3f} | {risk}")

    print(f"\n  {hallucination_risk}/{len(adv_items)} adversarial queries have low retrieval distance (hallucination risk).")
    print(f"  Fix: add a distance threshold guard in connector.py — reject if min score > 0.4.")

    return hallucination_risk / len(adv_items) if adv_items else 0.0


# ─────────────────────────────────────────────────────────────────────────
#  MODULE 4: FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────
def module4_verdict(retrieval: dict, generation: dict, adv_risk: float):
    print("\n╔" + "═" * 62 + "╗")
    print("║  MODULE 4: PASS / FAIL VERDICT                             ║")
    print("╚" + "═" * 62 + "╝\n")

    targets = [
        ("Retrieval P@5",        retrieval["P@5"],             0.60),
        ("Retrieval R@5",        retrieval["R@5"],             0.60),
        ("Retrieval MRR",        retrieval["MRR"],             0.55),
        ("Faithfulness",         generation["faithfulness"],   0.80),
        ("Answer Relevancy",     generation["answer_relevancy"],0.80),
        ("Adversarial Risk",     1.0 - adv_risk,               0.70),  # higher = safer
    ]

    DIAGNOSIS = {
        "Retrieval P@5":    "Too many irrelevant chunks returned → tune TOP_K, improve chunking, or add reranking threshold",
        "Retrieval R@5":    "Missing relevant chunks → check HNSW ef_search, try larger TOP_K, review chunking size",
        "Retrieval MRR":    "Relevant chunk exists but ranks low → increase ef_search or tune reranker",
        "Faithfulness":     "LLM hallucinating → tighten system prompt in build_prompt(), reduce context noise",
        "Answer Relevancy": "Answers off-topic → check build_prompt() instruction clarity",
        "Adversarial Risk": "Out-of-scope questions not rejected → add distance threshold in retrieve()",
    }

    all_pass = True
    print(f"  {'Metric':<25} {'Score':<10} {'Target':<10} {'Status'}")
    print(f"  {'─' * 60}")
    failed = []
    for label, score, target in targets:
        passed = score >= target
        all_pass = all_pass and passed
        icon = "✅ PASS" if passed else "❌ FAIL"
        bar  = "█" * int(score * 20)
        print(f"  {label:<25} {score:.3f}     {target:.3f}     {icon}  {bar}")
        if not passed:
            failed.append(label)

    if failed:
        print(f"\n  ROOT CAUSE & FIX:")
        print(f"  {'─' * 60}")
        for label in failed:
            print(f"\n  ❌ {label}")
            print(f"     → {DIAGNOSIS[label]}")

    print(f"\n  {'─' * 60}")
    verdict = "✅  SYSTEM READY" if all_pass else "❌  DO NOT SHIP — fix failing metrics first"
    print(f"  FINAL VERDICT: {verdict}")
    print(f"  {'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")
    parser.add_argument("--list-files", action="store_true",  help="List file_names in the DB and exit")
    parser.add_argument("--sample",     action="store_true",  help="Print sample DB chunks and exit")
    parser.add_argument("--debug",      metavar="QUESTION",   help="Show raw vector distances + reranker scores for one question")
    parser.add_argument("--tune",       action="store_true",  help="Analyse all golden questions and suggest optimal thresholds")
    args = parser.parse_args()

    print("=" * 64)
    print("  RAG EVALUATION HARNESS — min_rag_connect_ollama")
    print(f"  Golden dataset: {len(GOLDEN_DATASET)} questions "
          f"({sum(1 for g in GOLDEN_DATASET if g.category != 'adversarial')} answerable, "
          f"{sum(1 for g in GOLDEN_DATASET if g.category == 'adversarial')} adversarial)")
    print("=" * 64)

    if args.list_files:
        list_db_files()
        return

    if args.sample:
        sample_db_chunks(n=5)
        return

    if args.debug:
        debug_query(args.debug)
        return

    if args.tune:
        tune_thresholds()
        return

    # ── Full evaluation ────────────────────────────────────────────────
    retrieval_scores = module1_retrieval()
    input("\n  ▶  Press ENTER to run Module 2: Generation Evaluation...")

    generation_scores = module2_generation()
    input("\n  ▶  Press ENTER to run Module 3: Adversarial Check...")

    adv_risk = module3_adversarial()
    input("\n  ▶  Press ENTER to see Final Verdict...")

    module4_verdict(retrieval_scores, generation_scores, adv_risk)


if __name__ == "__main__":
    main()
