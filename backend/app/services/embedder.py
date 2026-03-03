import numpy as np
import faiss
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX_PATH, EMBEDDINGS_PATH

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def configure_gemini():
    pass


def embed_text(text: str) -> list[float]:
    model = get_model()
    vec = model.encode(text[:512], normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> np.ndarray:
    model = get_model()
    logger.info(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        [t[:512] for t in texts],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
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
    model = get_model()
    vec = model.encode(query[:512], normalize_embeddings=True)
    return np.array([vec], dtype=np.float32)
