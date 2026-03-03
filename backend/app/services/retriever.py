"""
Hybrid Retriever: BM25 + FAISS dense search + LLM reranking via Gemini.
"""
import json
import logging
import re
import numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi
from google import genai as google_genai

from app.config import (
    GEMINI_API_KEY, DENSE_WEIGHT, BM25_WEIGHT,
    CANDIDATE_SIZE, MAX_RECOMMENDATIONS, MIN_RECOMMENDATIONS
)

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a senior SHL assessment consultant. Given a hiring query and candidate assessments, return the {k} MOST RELEVANT assessments, ranked best to worst.

RULES (strictly follow):
1. **Exact skill match first**: If query names specific tools/languages (Java, SQL, Python, Selenium, Excel), prioritize assessments whose name directly matches. A "Core Java" test beats a generic cognitive test for a Java role.
2. **K/P balance when both needed**: If query has BOTH technical skills AND soft skills (communication, collaboration, leadership), ensure mix of K-type (Knowledge) and P-type (Personality) in results — at least 2 of each.
3. **Leadership/executive queries**: If query is about senior role, COO, executive, cultural fit, or leadership — prioritize personality/leadership reports (OPQ, Leadership Report, Global Skills) over technical tests.
4. **Duration constraint**: If query gives a time limit ("40 minutes", "1 hour"), exclude or deprioritize assessments far outside that range.
5. **Cognitive + personality for analyst/consultant roles**: If query is about analyst, consultant, or problem-solving roles without specific tech tools, include BOTH cognitive (Verify Numerical, Verify Verbal, Inductive Reasoning) AND personality (OPQ32r) tests.

Query: {query}

Candidates (JSON):
{candidates}

Return ONLY a valid JSON array of exactly {k} objects with "url" and "explanation" fields:
[{{"url": "url1", "explanation": "One sentence reason."}}, ...]
No markdown, no extra text. Exactly {k} items.
"""

FALLBACK_EXPLANATION_PROMPT = """Given this hiring query and assessment, write ONE sentence (max 20 words) explaining why this assessment is relevant.
Query: {query}
Assessment: {name} ({test_types})
Return ONLY the explanation sentence.
"""


class HybridRetriever:
    def __init__(self, catalog: list[dict], faiss_index):
        """
        catalog: list of assessment dicts with keys:
            id, name, url, description, test_type, remote_testing,
            adaptive_irt, duration_minutes, embedding_text
        faiss_index: loaded FAISS index (ntotal == len(catalog))
        """
        self.catalog = catalog
        self.faiss_index = faiss_index
        self.url_to_item = {item["url"]: item for item in catalog}

        # Build BM25 index over embedding_text
        logger.info("Building BM25 index...")
        corpus = [
            self._tokenize(item.get("embedding_text", f"{item['name']} {item.get('description','')}"))
            for item in catalog
        ]
        self.bm25 = BM25Okapi(corpus)
        self.bm25_texts = [
            item.get("embedding_text", f"{item['name']} {item.get('description','')}").lower()
            for item in catalog
        ]
        logger.info(f"BM25 index built over {len(corpus)} assessments")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\w+", text)

    def _dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(catalog_idx, score)] for top-k dense results."""
        scores, indices = self.faiss_index.search(query_vec, k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _bm25_search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Return [(catalog_idx, normalized_score)] for top-k BM25 results."""
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        max_score = scores[top_k_indices[0]] if scores[top_k_indices[0]] > 0 else 1.0
        return [(int(idx), float(scores[idx]) / max_score) for idx in top_k_indices]

    def _duration_penalty(self, item: dict, constraint_minutes: Optional[int]) -> float:
        """Return a penalty [0,1] based on duration mismatch."""
        if constraint_minutes is None:
            return 0.0
        item_dur = item.get("duration_minutes")
        if item_dur is None:
            return 0.0  # unknown duration — no penalty
        diff = abs(item_dur - constraint_minutes)
        if diff <= 10:
            return 0.0  # perfect match
        elif diff <= 25:
            return 0.05
        elif diff <= 60:
            return 0.15
        else:
            return 0.30

    def _enforce_kp_balance(self, candidates: list[dict], needs_tech: bool, needs_soft: bool) -> list[dict]:
        """
        If query needs both tech and soft skills, ensure at least 2 K-type and 2 P-type
        appear in the top results. Reorders if necessary.
        """
        if not (needs_tech and needs_soft):
            return candidates

        k_items = [c for c in candidates if "K" in c.get("test_type", [])]
        p_items = [c for c in candidates if "P" in c.get("test_type", [])]
        other_items = [c for c in candidates if c not in k_items and c not in p_items]

        target_k = min(3, len(k_items))
        target_p = min(3, len(p_items))

        # Build balanced list: alternate K and P, fill rest from other/remaining
        balanced = []
        ki, pi = 0, 0
        while len(balanced) < MAX_RECOMMENDATIONS:
            added = False
            if ki < target_k and ki < len(k_items):
                balanced.append(k_items[ki]); ki += 1; added = True
            if pi < target_p and pi < len(p_items):
                balanced.append(p_items[pi]); pi += 1; added = True
            if not added:
                break

        # Add remaining items from candidates not already in balanced
        in_balanced = {c["url"] for c in balanced}
        for c in candidates:
            if c["url"] not in in_balanced and len(balanced) < MAX_RECOMMENDATIONS:
                balanced.append(c)

        return balanced[:MAX_RECOMMENDATIONS]

    def _llm_rerank(self, query_text: str, candidates: list[dict], k: int) -> list[dict]:
        """Use Gemini Flash to rerank top-20 candidates and attach per-item explanations."""
        client = google_genai.Client(api_key=GEMINI_API_KEY)
        candidate_data = [
            {
                "url": c["url"],
                "name": c["name"],
                "test_type": c.get("test_type", []),
                "duration_minutes": c.get("duration_minutes"),
                "description": (c.get("description") or "")[:200],
            }
            for c in candidates[:30]
        ]

        prompt = RERANK_PROMPT.format(
            query=query_text[:1000],
            candidates=json.dumps(candidate_data, indent=2),
            k=min(k, len(candidates)),
        )
        # Add a manual retry loop for 429 errors from free tier Gemini
        max_retries = 5
        # Build normalized URL lookup (strip trailing slash, lowercase)
        url_to_candidate = {c["url"].strip().rstrip("/").lower(): c for c in candidates}
        result = []
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                content = response.text.strip()
                content = re.sub(r"```(?:json)?\n?", "", content).strip().rstrip("```").strip()
                ranked_objects = json.loads(content)
                if not isinstance(ranked_objects, list):
                    raise ValueError("LLM returned non-list")
                
                for obj in ranked_objects:
                    # Support both {url, explanation} objects and bare strings (fallback)
                    if isinstance(obj, dict):
                        url = obj.get("url", "")
                        explanation = obj.get("explanation", "")
                    elif isinstance(obj, str):
                        url = obj
                        explanation = ""
                    else:
                        continue
                    norm_url = url.strip().rstrip("/").lower()
                    cand = url_to_candidate.get(norm_url)
                    if cand:
                        item = dict(cand)  # copy
                        item["_explanation"] = explanation
                        result.append(item)

                # Add any missed candidates without explanation
                seen = {c["url"].strip().rstrip("/").lower() for c in result}
                for c in candidates:
                    if c["url"].strip().rstrip("/").lower() not in seen and len(result) < MAX_RECOMMENDATIONS:
                        item = dict(c)
                        item.setdefault("_explanation", "")
                        result.append(item)
                return result

            except Exception as e:
                # If we hit a rate limit, sleep and retry
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = 15 * (attempt + 1)
                        logger.warning(f"429 Rate limit hit in reranker (attempt {attempt+1}), sleeping for {sleep_time}s...")
                        import time
                        time.sleep(sleep_time)
                        continue
                logger.warning(f"LLM reranking failed, using score order: {e}")
                for c in candidates:
                    c.setdefault("_explanation", "")
                return candidates

    def retrieve(
        self,
        query_vec: np.ndarray,
        query_text: str,
        parsed_jd: dict,
        use_llm_rerank: bool = True,
    ) -> list[dict]:
        """
        Main retrieval pipeline:
        1. FAISS top-CANDIDATE_SIZE
        2. BM25 top-CANDIDATE_SIZE
        3. Weighted merge with constraint-aware scoring
        4. K/P balance enforcement
        5. LLM reranking (attaches explanation per item)
        6. Return top MAX_RECOMMENDATIONS with _score and _explanation fields
        """
        query_tokens = self._tokenize(query_text)

        # Step 1: Dense search
        dense_results = self._dense_search(query_vec, CANDIDATE_SIZE)
        dense_map = {idx: score for idx, score in dense_results}

        # Step 2: BM25 search
        bm25_results = self._bm25_search(query_tokens, CANDIDATE_SIZE)
        bm25_map = {idx: score for idx, score in bm25_results}

        # Step 3: Merge candidates with scores
        all_idxs = set(dense_map.keys()) | set(bm25_map.keys())
        scored = []
        duration_constraint = parsed_jd.get("duration_constraint")

        for idx in all_idxs:
            if idx < 0 or idx >= len(self.catalog):
                continue
            item = self.catalog[idx]
            d_score = dense_map.get(idx, 0.0)
            b_score = bm25_map.get(idx, 0.0)
            hybrid_score = DENSE_WEIGHT * d_score + BM25_WEIGHT * b_score
            penalty = self._duration_penalty(item, duration_constraint)
            final_score = max(0.0, hybrid_score - penalty)
            scored.append((final_score, item))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Attach _score to each candidate
        candidates = []
        score_map = {}
        for final_score, item in scored[:CANDIDATE_SIZE]:
            item_copy = dict(item)
            item_copy["_score"] = round(float(final_score), 4)
            item_copy.setdefault("_explanation", "")
            candidates.append(item_copy)
            score_map[item["url"]] = round(float(final_score), 4)

        # Step 4: K/P balance
        needs_tech = parsed_jd.get("needs_tech", False)
        needs_soft = parsed_jd.get("needs_soft", False)
        candidates = self._enforce_kp_balance(candidates, needs_tech, needs_soft)

        # Step 5: LLM reranking (also attaches _explanation)
        if use_llm_rerank and len(candidates) > MIN_RECOMMENDATIONS:
            candidates = self._llm_rerank(
                parsed_jd.get("canonical_query", query_text),
                candidates,
                MAX_RECOMMENDATIONS,
            )

        # Restore scores (LLM rerank changes order but not original hybrid scores)
        for c in candidates:
            if "_score" not in c or c["_score"] == 0.0:
                c["_score"] = score_map.get(c["url"], 0.0)

        # Final deduplicate and slice
        seen_urls = set()
        final = []
        for c in candidates:
            if c["url"] not in seen_urls:
                seen_urls.add(c["url"])
                final.append(c)
            if len(final) >= MAX_RECOMMENDATIONS:
                break

        return final[:MAX_RECOMMENDATIONS]
