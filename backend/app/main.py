import logging
import csv
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.models import RecommendRequest, RecommendResponse, AssessmentRecommendation, HealthResponse
from app.services.embedder import embed_batch, build_faiss_index, save_index, load_index, embed_query, configure_gemini
from app.services.retriever import HybridRetriever
from app.services.jd_parser import parse_request
from app.config import CATALOG_CSV_PATH, EMBEDDINGS_PATH, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

retriever: HybridRetriever = None
catalog: list[dict] = []


def load_catalog_from_csv() -> list[dict]:
    items = []
    with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_types = [t.strip() for t in row.get("test_type", "").split(",") if t.strip()]
            dur = row.get("duration_minutes", "")
            items.append({
                "id": int(row.get("id", 0)),
                "name": row["name"],
                "url": row["url"],
                "description": row.get("description", ""),
                "test_type": test_types,
                "remote_testing": row.get("remote_testing", "False").lower() in ("1", "true"),
                "adaptive_irt": row.get("adaptive_irt", "False").lower() in ("1", "true"),
                "duration_minutes": int(dur) if dur.isdigit() else None,
                "embedding_text": (
                    f"{row['name']}. Test type: {row.get('test_type','')}. "
                    f"Duration: {dur or 'N/A'} minutes. {row.get('description','')}"
                ).strip(),
            })
    return items


def load_or_build_index(catalog_items: list[dict]):
    from app.services.embedder import EMBED_DIM
    index = load_index()
    if index is not None and index.ntotal == len(catalog_items) and index.d == EMBED_DIM:
        logger.info(f"Using existing FAISS index ({index.ntotal} vectors, dim={index.d})")
        return index
    if index is not None:
        logger.warning(f"Stale FAISS index (ntotal={index.ntotal}, d={index.d}) — rebuilding with dim={EMBED_DIM}")

    logger.info(f"Building embeddings for {len(catalog_items)} assessments...")
    texts = [item["embedding_text"] for item in catalog_items]
    embeddings = embed_batch(texts)
    index = build_faiss_index(embeddings)
    save_index(index, embeddings)
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, catalog
    logger.info("Starting SHL Recommendation API...")

    try:
        catalog = load_catalog_from_csv()
        logger.info(f"Loaded {len(catalog)} assessments from catalog CSV")
    except FileNotFoundError:
        logger.error(
            f"Catalog CSV not found at {CATALOG_CSV_PATH}. "
            "Run the crawler first: python crawler/scrape_shl_catalog.py"
        )
        catalog = []

    if catalog:
        configure_gemini()
        faiss_index = load_or_build_index(catalog)
        retriever = HybridRetriever(catalog, faiss_index)
        logger.info("Retriever ready")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL Individual Test Solutions from a natural language query or JD.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(status="ok")


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
async def recommend(request: Request, body: RecommendRequest):
    start_time = time.perf_counter()

    if not retriever:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Run the crawler first.",
        )

    if not body.get_input_text():
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: query, jd_text, or jd_url.",
        )

    try:
        parsed = parse_request(
            query=body.query,
            jd_text=body.jd_text,
            jd_url=body.jd_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"JD parsing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse input.")

    try:
        query_vec = embed_query(parsed["canonical_query"])
        results = retriever.retrieve(
            query_vec=query_vec,
            query_text=parsed["canonical_query"],
            parsed_jd=parsed,
            use_llm_rerank=True,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Retrieval pipeline failed.")

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    top_urls = [r["url"] for r in results[:5]]
    logger.info(f"[{latency_ms}ms] query={parsed['canonical_query'][:60]}... top5={top_urls}")

    recommendations = [
        AssessmentRecommendation(
            assessment_name=r["name"],
            assessment_url=r["url"],
            score=r.get("_score", 0.0),
            test_type=",".join(r.get("test_type", [])) if r.get("test_type") else "",
            duration_minutes=r.get("duration_minutes"),
            remote_testing=r.get("remote_testing", False),
            adaptive_irt=r.get("adaptive_irt", False),
            explanation=r.get("_explanation", ""),
        )
        for r in results
    ]

    return RecommendResponse(recommendations=recommendations)
