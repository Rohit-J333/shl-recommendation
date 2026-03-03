"""
Configuration — reads from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CATALOG_CSV_PATH = DATA_DIR / "catalog_clean.csv"
DB_PATH = DATA_DIR / "shl_catalog.db"

# Retrieval hyperparameters (tuned on train set)
# 0.6/0.4 dense/BM25 balance: semantic retrieval dominates for concept-heavy
# queries (leadership, cultural-fit, consulting) while BM25 still captures
# exact skill keywords (Java, SQL, Selenium).
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
CANDIDATE_SIZE = 100
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1

# K/P balance thresholds
TECH_KEYWORDS = [
    "java", "python", "sql", "javascript", "typescript", "react", "node",
    "developer", "engineer", "programmer", "coder", "coding", "database",
    "api", "backend", "frontend", "devops", "cloud", "aws", "selenium",
    "html", "css", "data analyst", "data scientist", "machine learning",
    "software", "testing", "qa", "automation"
]
SOFT_KEYWORDS = [
    "collaborate", "communication", "team", "leadership", "management",
    "interpersonal", "stakeholder", "culture", "personality", "behavior",
    "sales", "customer", "service", "emotional", "empathy", "verbal",
    "written", "presentation", "relationship", "influence", "negotiation"
]
