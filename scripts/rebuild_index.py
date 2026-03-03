"""
Rebuilds the FAISS index and embeddings for the updated catalog.
Run this after adding new assessments to catalog_clean.csv.
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.config import CATALOG_CSV_PATH
from app.services.embedder import embed_batch, build_faiss_index, save_index

items = []
with open(CATALOG_CSV_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        dur = row.get("duration_minutes", "")
        items.append({
            "name": row["name"],
            "embedding_text": (
                f"{row['name']}. Test type: {row.get('test_type','')}. "
                f"Duration: {dur or 'N/A'} minutes. {row.get('description','')}"
            ).strip()
        })

print(f"Building embeddings for {len(items)} assessments...")
texts = [item["embedding_text"] for item in items]
embeddings = embed_batch(texts)
index = build_faiss_index(embeddings)
save_index(index, embeddings)
print(f"Done. FAISS index: ntotal={index.ntotal}, dim={index.d}")
