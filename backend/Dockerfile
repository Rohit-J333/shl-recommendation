# Build context must be the repo root: `docker build -f backend/Dockerfile .`
FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformer model so it's baked into the image
# (avoids HuggingFace download on first request which causes Render timeout)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app code preserving backend/ directory so config.py path resolution works:
# config.py is at /app/backend/app/config.py → .parent.parent.parent = /app → /app/data/ ✓
COPY backend/app/ ./backend/app/
COPY data/catalog_clean.csv ./data/catalog_clean.csv
COPY data/faiss.index ./data/faiss.index
COPY data/embeddings.npy ./data/embeddings.npy

# Expose port
EXPOSE 8000

# Run from /app/backend so uvicorn finds the `app` package
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
