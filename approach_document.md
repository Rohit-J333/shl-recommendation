# SHL Assessment Recommendation System — Approach Document

## 1. Problem Understanding & Data Pipeline

### Objective
Given a natural language query, job description text, or JD URL, return 1–10 ranked SHL Individual Test Solutions from the live product catalog.

### Data Ingestion (Crawler)
- Scraped SHL product catalog (`https://www.shl.com/solutions/products/product-catalog/`) using `requests` + `BeautifulSoup4` with `type=1` filter to restrict to **Individual Test Solutions only**, ignoring Pre-packaged Job Solutions.
- Parsed per-row: assessment name, canonical URL, remote testing flag, adaptive IRT flag, test type codes (K, P, A, B, S), and duration in minutes.
- Enriched the first 60 assessments with detail-page description text scraped from individual product pages.
- Final catalog: **386 verified Individual Test Solutions** (≥377 required), stored in `data/catalog_clean.csv`, `data/catalog_raw.json`, and `data/shl_catalog.db` (SQLite).
- URL normalization: legacy `/products/product-catalog/view/` paths are unified to `/solutions/products/product-catalog/view/` throughout the pipeline to ensure consistent matching.

---

## 2. Retrieval & LLM Integration

### Embedding Layer
- Used **`all-MiniLM-L6-v2`** (sentence-transformers, 384-dim) for local, cost-free embeddings of all catalog items. Each assessment is embedded as: `"[name]. Test type: [types]. Duration: [N] mins. [description]"`.
- FAISS `IndexFlatIP` (inner-product on L2-normalised vectors = cosine similarity) stores all 386 vectors and is cached to disk for fast startup.

### Hybrid Retrieval
- **Dense (FAISS)** top-100 by cosine similarity + **BM25** (rank-bm25) top-100 over tokenized embedding texts, merged as:
  ```
  hybrid_score = 0.6 × dense_score + 0.4 × bm25_score
  ```
- BM25 captures exact keyword matches (Java, SQL, Selenium); dense captures semantic intent (COO → leadership, cultural-fit → OPQ personality).

### JD Understanding (LLM)
- **Gemini 2.5 Flash** parses each input into structured JSON: `{role, seniority, hard_skills, soft_skills, domain, duration_constraint_minutes, canonical_query}`. The `canonical_query` is a concise 2–3 sentence summary used for both embedding and BM25 retrieval.
- Handles plain query text, raw JD text, and JD URLs (fetched with `trafilatura`).

### K/P Balance Enforcement
- If a query signals both technical needs (`developer`, `SQL`, `automation`, etc.) AND soft needs (`collaboration`, `leadership`, `communication`, etc.), the system enforces at least 3 K-type and 3 P-type assessments in the candidate set before LLM reranking.

### LLM Reranker
- Top-30 candidates (post hybrid scoring) are passed to **Gemini 2.5 Flash** with a structured prompt: rerank by relevance, enforce skill-specific test priority (exact Java/SQL tests before generic cognitive), apply duration penalty, and ensure K/P balance for mixed roles.
- Returns per-item explanations displayed in the UI.
- Retry logic handles Gemini free-tier 429 rate limits.

---

## 3. Evaluation & Optimization

### Metric
**Mean Recall@10** = average of `|relevant ∩ top-10 predicted| / |relevant|` across all 10 labeled train queries.

### Optimization Journey

| Phase | Change | Mean Recall@10 |
|-------|--------|----------------|
| Baseline (Gemini embeddings, 0.6/0.4 dense/BM25) | Initial prototype | 0.0644 |
| Phase 1: URL normalization | Unified old `/products/` → `/solutions/products/` prefix | 0.1944 |
| Phase 2: Weight tuning + LLM window | dense=0.6, bm25=0.4 (was 0.3/0.7); LLM candidate window 20→30 | **0.2633** |

Key per-query results (final config):

| Query | Recall@10 |
|-------|-----------|
| Java developer + collaboration | **0.80** |
| COO for China (cultural fit) | **0.50** |
| ICICI Bank Admin (0–2 yrs) | **0.50** |
| Senior Data Analyst (SQL/Excel/Python) | **0.30** |
| Sales role new graduates | 0.22 |
| Content Writer (English + SEO) | 0.20 |
| QA Engineer (Selenium/Java/SQL) | 0.11 |
| Mirchi Radio JD (media manager) | 0.00 |
| Marketing Manager (community/brand) | 0.00 |
| Consultant (I/O psychology) | 0.00 |

Queries with 0.0 recall involve highly domain-specific JDs where the catalog's generic assessment descriptions do not contain strong keyword signals, and semantic embeddings at this model scale cannot fully bridge the conceptual gap.

---

## 4. System Architecture & Deployment

**Backend**: Python 3.11, FastAPI, Uvicorn. Endpoints: `GET /health` → `{"status":"ok"}`, `POST /recommend` → `{"recommendations": [...]}` (Appendix 2 schema).

**Frontend**: React (Vite), supports quick query / JD text / JD URL tabs, K/P balance indicator, animated results with explanation per assessment.

**Deployment**: Backend containerized with Docker (`backend/Dockerfile`), hosted on Render/Railway free tier. Frontend built with `npm run build` and deployed to Vercel with `VITE_API_URL` pointing to the live API.

**Limitations**: Three concept-heavy queries (media manager, I/O consultant, B2B marketing manager) score 0.0 Recall@10 due to domain mismatch between long JD terminology and short catalog descriptions. Future work: fine-tune a domain-specific embedding model, expand catalog descriptions via web scraping, or use a retrieval-augmented catalog enrichment step.
