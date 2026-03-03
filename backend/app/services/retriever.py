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

RULES (apply ALL that are relevant):

1. **Exact skill match first**: If query names specific tools/languages (Java, SQL, Python, JavaScript, Selenium, Excel, Tableau, SDLC, Jira), prioritize assessments whose name DIRECTLY includes that technology. "Core Java" beats generic cognitive for a Java role; "SQL Server" beats verbal reasoning for a SQL role.

2. **K/P balance for mixed roles**: If query mentions BOTH technical skills AND people skills (communication, collaboration, leadership, team), include a MIX — at least 2 Knowledge/Skills (K-type) AND at least 2 Personality (P-type) assessments.

3. **Leadership / executive roles**: If query mentions COO, VP, Director, CXO, Head of, senior leadership, cultural fit, or executive — PRIORITIZE: OPQ32r, Leadership Reports, Enterprise Leadership Report, Global Skills Assessment. These outrank technical tests for this query type.

4. **Analyst / consultant / finance roles**: If query mentions analyst, consultant, finance, accounting, or problem-solving WITHOUT specific tech tools — include: (a) Verify Verbal Ability, (b) Verify Numerical Ability or Inductive Reasoning, AND (c) OPQ32r personality. These three together are the SHL standard for professional roles.

5. **Media / communications / content / marketing roles**: If query mentions media, radio, broadcast, journalism, content writing, SEO, social media, brand, community, or communications — PRIORITIZE: Verify Verbal Ability, English Comprehension, Interpersonal Communications, OPQ32r, and any domain-specific test (Marketing, Digital Advertising, Written English). Verbal and comprehension tests are critical for these roles.

6. **Writing / editorial roles**: If query mentions writing, copy, editorial, SEO, blog, email writing, or content creation — include: Written English, Email Writing, English Comprehension, Verify Verbal Ability. These are the most relevant SHL tests.

7. **Customer service / support roles**: If query mentions customer service, support, call center, helpdesk, or English communication — include: Spoken English (SVAR), English Comprehension, Interpersonal Communications, Verify Verbal Ability.

8. **Sales / business development roles**: For sales roles (especially entry-level or graduate) — include Entry Level Sales assessments, Business Communication, Verbal Ability, and OPQ32r personality tests.

9. **Duration constraint**: If query states a time limit (e.g., "max 40 minutes", "less than 30 minutes", "1 hour"), EXCLUDE assessments that clearly exceed that limit based on their duration_minutes value. If duration_minutes is null/unknown, keep the assessment but note the constraint.

10. **General management roles (no specific tech)**: For any management, supervisory, or professional role not covered by rules 1-9 — include OPQ32r personality AND at least one cognitive test (Verify Verbal Ability, Verify Numerical Ability, or Inductive Reasoning). Every professional role benefits from personality + cognitive assessment.

Query: {query}

Candidates (JSON):
{candidates}

Return ONLY a valid JSON array of exactly {k} objects with "url" and "explanation" fields.
Each explanation must be ONE concise sentence explaining WHY this specific assessment is relevant to THIS specific query:
[{{"url": "https://...", "explanation": "One sentence reason specific to the query."}}, ...]
No markdown fences, no extra text. Exactly {k} items.
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
        appear at the TOP of the candidate list (for the LLM to consider), without
        discarding the rest of the pool. Returns the full reordered candidate list.
        """
        if not (needs_tech and needs_soft):
            return candidates

        k_items = [c for c in candidates if "K" in c.get("test_type", [])]
        p_items = [c for c in candidates if "P" in c.get("test_type", [])]

        target_k = min(3, len(k_items))
        target_p = min(3, len(p_items))

        # Build a priority front section: alternate K and P
        priority = []
        ki, pi = 0, 0
        while True:
            added = False
            if ki < target_k:
                priority.append(k_items[ki]); ki += 1; added = True
            if pi < target_p:
                priority.append(p_items[pi]); pi += 1; added = True
            if not added:
                break

        # Append ALL remaining candidates (preserving their order) after priority items
        in_priority = {c["url"] for c in priority}
        for c in candidates:
            if c["url"] not in in_priority:
                priority.append(c)

        return priority  # Full list — LLM will select top MAX_RECOMMENDATIONS from this

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
            for c in candidates[:50]
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

    # Seniority → URL slug keywords to boost in candidate pool
    SENIORITY_BOOST = {
        "executive": ["leadership-report", "opq", "enterprise-leadership", "global-skills", "executive-scenarios", "hipo"],
        "senior":    ["professional-7-1", "professional-7-0", "opq"],
    }
    # Domain → URL slug keywords to ensure appear in candidate pool
    # Slugs are matched as substrings of the catalog item's URL (lowercase), so
    # use the exact URL fragment as it appears in the catalog.
    DOMAIN_BOOST = {
        "media":          ["verify-verbal", "english-comprehension", "interpersonal", "marketing-new",
                           "opq", "shl-verify-interactive-inductive"],
        "marketing":      ["verify-verbal", "digital-advertising", "marketing-new", "manager-8-0",
                           "opq", "writex", "written-english", "excel-365-essentials",
                           "shl-verify-interactive-inductive"],
        "hr":             ["opq", "verify-verbal", "verify-numerical", "interpersonal",
                           "professional", "administrative"],
        "finance":        ["shl-verify-interactive-numerical", "verify-numerical", "financial",
                           "excel", "opq", "professional"],
        "sales":          ["entry-level-sales", "business-communication", "opq", "verify-verbal",
                           "interpersonal", "svar"],
        "operations":     ["verify-numerical", "verify-verbal", "opq", "professional"],
        "general":        ["opq", "verify-verbal", "verify-numerical"],
        "consultant":     ["verify-verbal-next-generation", "shl-verify-interactive-numerical",
                           "administrative-professional", "opq", "professional-7-1",
                           "verify-verbal"],
        "consulting":     ["verify-verbal-next-generation", "shl-verify-interactive-numerical",
                           "administrative-professional", "opq", "professional-7-1",
                           "verify-verbal"],
        "tech":           ["verify-numerical", "professional", "verify-verbal"],
        "technology":     ["verify-numerical", "professional", "verify-verbal"],
    }

    def _inject_domain_boost(self, candidates: list[dict], parsed_jd: dict) -> list[dict]:
        """
        Ensure key domain/seniority assessments are present in the candidate pool.
        Fetches them from the full catalog and injects at the end with a small score
        so they're available to the LLM reranker even if they missed hybrid retrieval.
        """
        seniority = parsed_jd.get("seniority", "mid").lower()
        domain = parsed_jd.get("domain", "general").lower()

        boost_slugs = []
        boost_slugs.extend(self.SENIORITY_BOOST.get(seniority, []))
        boost_slugs.extend(self.DOMAIN_BOOST.get(domain, []))

        if not boost_slugs:
            return candidates

        existing_urls = {c["url"].lower() for c in candidates}
        min_score = min((c.get("_score", 0.0) for c in candidates), default=0.0)
        inject_score = max(0.0, min_score * 0.5)  # inject below last candidate

        injected = []
        for item in self.catalog:
            url_lower = item["url"].lower()
            if url_lower in existing_urls:
                continue
            if any(slug in url_lower for slug in boost_slugs):
                copy = dict(item)
                copy["_score"] = inject_score
                copy.setdefault("_explanation", "")
                injected.append(copy)
                existing_urls.add(url_lower)

        if injected:
            logger.info(f"Domain/seniority boost injected {len(injected)} candidates (domain={domain}, seniority={seniority})")
            # Inject boosted items at the TOP so they fall within the LLM reranker's
            # top-50 candidate window, rather than being buried below position 100.
            candidates = injected + candidates

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
        4. Domain/seniority boost injection
        5. K/P balance enforcement
        6. LLM reranking (attaches explanation per item)
        7. Return top MAX_RECOMMENDATIONS with _score and _explanation fields
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

        # Step 4: Domain/seniority boost
        candidates = self._inject_domain_boost(candidates, parsed_jd)
        # Update score_map with injected items
        for c in candidates:
            score_map.setdefault(c["url"], c.get("_score", 0.0))

        # Step 5: K/P balance
        needs_tech = parsed_jd.get("needs_tech", False)
        needs_soft = parsed_jd.get("needs_soft", False)
        candidates = self._enforce_kp_balance(candidates, needs_tech, needs_soft)

        # Step 6: LLM reranking (also attaches _explanation)
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
