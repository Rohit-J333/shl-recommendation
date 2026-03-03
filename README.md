# SHL Assessment Recommendation System

> AI-powered system that recommends relevant SHL Individual Test Solutions from a natural language query or job description (text or URL).

## 📁 Project Structure

```
shl-recommendation-system/
├── crawler/              # SHL catalog scraper
├── backend/
│   ├── app/              # FastAPI application
│   │   ├── main.py       # API endpoints
│   │   ├── models.py     # Pydantic schemas
│   │   ├── config.py     # Configuration
│   │   └── services/     # embedder, retriever, jd_parser
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # React + Vite UI
├── data/                 # catalog CSV, train/test data, FAISS index
├── eval/                 # evaluation + test prediction scripts
└── docs/                 # 2-page approach document
```

## 🚀 Quickstart

### 1. Environment Setup

```bash
cd "d:\code\STL assignment\shl-recommendation-system"

# Create and activate venv
python -m venv .venv
.venv\Scripts\activate      # Windows

# Install backend deps
pip install -r backend/requirements.txt

# Set your Gemini API key
copy .env.example .env
# Edit .env and replace YOUR_GEMINI_API_KEY with your actual key
# Get free key at: https://ai.google.dev/
```

### 2. Crawl SHL Catalog (run once)

```bash
python crawler/scrape_shl_catalog.py
# Saves data/catalog_clean.csv (≥377 Individual Test Solutions)
```

### 3. Prepare Train Data

```bash
python data/prepare_train_data.py
# Converts Gen_AI Dataset.xlsx → data/train_set.csv
```

### 4. Start Backend API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
# First run builds FAISS embeddings (~5-10 min for 400 items)
# Subsequent runs load from cache (~3 sec)
```

API is live at: `http://localhost:8000`

Test it:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query":"Java developer who can collaborate with business teams, max 40 mins"}'
```

### 5. Start React Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 6. Run Evaluation

```bash
python eval/evaluate_recall.py
# Prints per-query Recall@10 and Mean Recall@10 on train set
```

### 7. Generate Test Predictions CSV

```bash
python eval/generate_test_predictions.py
# Saves data/test_predictions.csv in the required format:
# columns: Query, Assessment_url
```

## 🔌 API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /recommend`

**Request** (any one of):

```json
{ "query": "Natural language hiring query" }
{ "jd_text": "Full job description text..." }
{ "jd_url": "https://example.com/jobs/engineer" }
```

**Response**:

```json
{
  "recommendations": [
    {
      "assessment_name": "Core Java (Entry Level)",
      "assessment_url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "score": 0.82,
      "test_type": "K",
      "duration_minutes": 30,
      "remote_testing": true,
      "adaptive_irt": false,
      "explanation": "Directly tests Java programming skills required for the role."
    },
    ...
  ]
}
```

## 🏗️ Architecture

```
Query/JD Text or URL
        ↓
  jd_parser.py          ← LLM (Gemini Flash) extracts role, skills, constraints
        ↓
  embedder.py           ← Gemini embedding-001 (768-dim)
        ↓
  retriever.py          ← FAISS (dense, 0.6w) + BM25 (sparse, 0.4w)
        ↓
  K/P Balance           ← Enforces tech+soft mix when both required
        ↓
  LLM Reranker          ← Gemini Flash prompt over top-20 candidates
        ↓
  Top 1–10 results      → JSON via /recommend
```

## ☁️ Deployment

### Backend (Render.com free tier)

1. Push repo to GitHub
2. Create new **Web Service** on Render
3. Set **Root Directory** to `backend/`
4. Set env var `GEMINI_API_KEY=<your-key>`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel)

1. Set **Root Directory** to `frontend/`
2. Set env var `VITE_API_URL=https://your-render-url.onrender.com`
3. Deploy

## 📊 Evaluation

Run to see per-query Recall@10 and Mean Recall@10 on the labeled train set:

```bash
python eval/evaluate_recall.py
python eval/evaluate_recall.py --no-llm    # without LLM reranking
python eval/evaluate_recall.py --dense-w 0.7 --bm25-w 0.3
```
