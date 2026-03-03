"""
Embedder service: sentence-transformers for local embeddings + FAISS index.
Uses 'all-MiniLM-L6-v2' (384-dim, fast, no API key needed).
Keeps Gemini Flash for LLM reranking only.
"""
import numpy as np
import faiss
import logging
import time
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX_PATH, EMBEDDINGS_PATH

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloaded once, cached locally)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def configure_gemini():
    """No-op for sentence-transformers. Gemini still used for LLM reranking."""
    pass


def embed_text(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> list[float]:
    """Embed a single text using sentence-transformers."""
    model = get_model()
    vec = model.encode(text[:512], normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str], task_type: str = "SEMANTIC_SIMILARITY") -> np.ndarray:
    """Embed a batch of texts efficiently. Returns shape (N, EMBED_DIM)."""
    model = get_model()
    logger.info(f"Embedding {len(texts)} texts...")
    # sentence-transformers handles batching internally
    embeddings = model.encode(
        [t[:512] for t in texts],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP (inner-product = cosine since vectors are normalized)."""
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, embeddings: np.ndarray):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(str(EMBEDDINGS_PATH), embeddings)
    logger.info(f"FAISS index saved ({index.ntotal} vectors)")


def load_index() -> Optional[faiss.Index]:
    if FAISS_INDEX_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index
    return None


def embed_query(query: str) -> np.ndarray:
    """Embed and normalize a query vector for cosine similarity lookup."""
    model = get_model()
    vec = model.encode(query[:512], normalize_embeddings=True)
    return np.array([vec], dtype=np.float32)
