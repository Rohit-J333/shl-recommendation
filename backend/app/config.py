import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CATALOG_CSV_PATH = DATA_DIR / "catalog_clean.csv"
DB_PATH = DATA_DIR / "shl_catalog.db"

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
CANDIDATE_SIZE = 100
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1

TECH_KEYWORDS = [
    "java", "python", "sql", "javascript", "typescript", "react", "node",
    "html", "css", "c#", "c++", "kotlin", "swift", "golang", "rust",
    "developer", "engineer", "coder", "coding", "software",
    "database", "api", "backend", "frontend", "devops", "cloud", "aws",
    "docker", "kubernetes", "ci/cd",
    "selenium", "testing", "qa", "automation", "playwright",
    "data analyst", "data scientist", "machine learning", "deep learning",
    "tableau", "power bi", "powerbi", "excel", "pandas", "numpy",
    "jira", "confluence", "salesforce", "sap", "oracle", "erp", "crm",
    "sdlc", "agile", "scrum",
]

SOFT_KEYWORDS = [
    "collaborate", "communication", "team", "leadership", "management",
    "interpersonal", "stakeholder", "culture", "personality", "behavior",
    "sales", "customer", "service", "emotional", "empathy", "verbal",
    "written", "presentation", "relationship", "influence", "negotiation",
    "media", "content", "editorial", "brand", "marketing", "creative",
    "community", "radio", "broadcast", "journalism",
]
