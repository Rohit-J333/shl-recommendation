"""
Evaluation: Mean Recall@10 on the labeled Train-Set.
Reads train_set.csv, runs the full pipeline per query, computes Recall@10.
"""
import sys
import csv
import logging
import time
from pathlib import Path
from collections import defaultdict

# Make sure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np
from app.config import CATALOG_CSV_PATH
from app.services.embedder import load_index, embed_query, configure_gemini
from app.services.jd_parser import parse_request
from app.services.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CSV = Path(__file__).parent.parent / "data" / "train_set.csv"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_catalog() -> list[dict]:
    items = []
    with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = row.get("duration_minutes", "")
            test_types = [t.strip() for t in row.get("test_type", "").split(",") if t.strip()]
            items.append({
                "id": int(row.get("id", 0)),
                "name": row["name"],
                "url": row["url"],
                "description": row.get("description", ""),
                "test_type": test_types,
                "remote_testing": row.get("remote_testing", "False").lower() in ("1", "true"),
                "adaptive_irt": row.get("adaptive_irt", "False").lower() in ("1", "true"),
                "duration_minutes": int(dur) if dur.strip().isdigit() else None,
                "embedding_text": (
                    f"{row['name']}. Test type: {row.get('test_type','')}. "
                    f"Duration: {dur or 'N/A'} mins. {row.get('description','')}"
                ).strip(),
            })
    return items


def load_train_data() -> dict[str, list[str]]:
    """Load train set. Returns {query: [relevant_url, ...]}"""
    ground_truth = defaultdict(list)
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Query"].strip()
            u = row["Assessment_url"].strip()
            if q and u:
                ground_truth[q].append(u)
    return dict(ground_truth)


def recall_at_k(predicted_urls: list[str], relevant_urls: list[str], k: int = 10) -> float:
    """Recall@K = |relevant ∩ top-k predicted| / |relevant|"""
    if not relevant_urls:
        return 0.0
    top_k = set(predicted_urls[:k])
    relevant = set(relevant_urls)
    return len(top_k & relevant) / len(relevant)


def normalize_url(url: str) -> str:
    """Normalize URL for comparison: strip trailing slash, lowercase, unify /products/ and /solutions/products/."""
    url = url.strip().rstrip("/").lower()
    # Unify old prefix (/products/) with new prefix (/solutions/products/)
    OLD = "https://www.shl.com/products/product-catalog/view/"
    NEW = "https://www.shl.com/solutions/products/product-catalog/view/"
    if url.startswith(OLD):
        url = NEW + url[len(OLD):]
    return url


def run_evaluation(use_llm_rerank: bool = True, dense_w: float = 0.6, bm25_w: float = 0.4):
    """Run full evaluation on train set."""
    logger.info("=" * 60)
    logger.info(f"Config: dense={dense_w}, bm25={bm25_w}, llm_rerank={use_llm_rerank}")
    logger.info("=" * 60)

    configure_gemini()
    catalog = load_catalog()
    logger.info(f"Catalog loaded: {len(catalog)} assessments")

    faiss_index = load_index()
    if faiss_index is None or faiss_index.ntotal != len(catalog):
        logger.error("FAISS index not found or size mismatch. Run: python backend/app/main.py first to build it.")
        sys.exit(1)

    retriever = HybridRetriever(catalog, faiss_index)
    ground_truth = load_train_data()
    logger.info(f"Train queries: {len(ground_truth)}")

    recall_scores = []
    results_table = []

    for query, relevant_urls in ground_truth.items():
        logger.info(f"\nQuery: {query[:60]}...")
        try:
            parsed = parse_request(query=query)
            query_vec = embed_query(parsed["canonical_query"])
            predicted = retriever.retrieve(
                query_vec=query_vec,
                query_text=parsed["canonical_query"],
                parsed_jd=parsed,
                use_llm_rerank=use_llm_rerank,
            )
            predicted_urls = [normalize_url(r["url"]) for r in predicted]
            normalized_relevant = [normalize_url(u) for u in relevant_urls]

            score = recall_at_k(predicted_urls, normalized_relevant, k=10)
            recall_scores.append(score)
            results_table.append({
                "query": query[:60],
                "recall@10": round(score, 4),
                "predicted": len(predicted_urls),
                "relevant": len(relevant_urls),
            })
            logger.info(f"  Recall@10 = {score:.4f} ({len(predicted_urls)} predicted, {len(relevant_urls)} relevant)")
            
            # Google Gemini free tier rate limit is 15 requests per minute.
            # evaluate_recall makes 2 requests per query (1 for JD parsing, 1 for Reranking)
            # So we must sleep for 8.5 seconds to ensure we do not exceed 15 requests/min.
            time.sleep(8.5)

        except Exception as e:
            logger.error(f"  Error: {e}")
            recall_scores.append(0.0)
            results_table.append({"query": query[:60], "recall@10": 0.0, "predicted": 0, "relevant": len(relevant_urls)})
            time.sleep(8.5)

    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print("\n" + "=" * 70)
    print(f"{'Query':<50} {'R@10':>6} {'Pred':>5} {'Rel':>5}")
    print("-" * 70)
    for r in results_table:
        print(f"{r['query']:<50} {r['recall@10']:>6.4f} {r['predicted']:>5} {r['relevant']:>5}")
    print("=" * 70)
    print(f"{'Mean Recall@10:':<50} {mean_recall:>6.4f}")
    print("=" * 70)

    return mean_recall, results_table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM reranking")
    parser.add_argument("--dense-w", type=float, default=0.6)
    parser.add_argument("--bm25-w", type=float, default=0.4)
    args = parser.parse_args()

    mean_r, _ = run_evaluation(
        use_llm_rerank=not args.no_llm,
        dense_w=args.dense_w,
        bm25_w=args.bm25_w,
    )
    print(f"\nFinal Mean Recall@10: {mean_r:.4f}")
