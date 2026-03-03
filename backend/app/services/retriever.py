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
        self.catalog = catalog
        self.faiss_index = faiss_index
        self.url_to_item = {item["url"]: item for item in catalog}

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
        scores, indices = self.faiss_index.search(query_vec, k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _bm25_search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        max_score = scores[top_k_indices[0]] if scores[top_k_indices[0]] > 0 else 1.0
        return [(int(idx), float(scores[idx]) / max_score) for idx in top_k_indices]

    def _duration_penalty(self, item: dict, constraint_minutes: Optional[int]) -> float:
        if constraint_minutes is None:
            return 0.0
        item_dur = item.get("duration_minutes")
        if item_dur is None:
            return 0.0
        diff = abs(item_dur - constraint_minutes)
        if diff <= 10:
            return 0.0
        elif diff <= 25:
            return 0.05
        elif diff <= 60:
            return 0.15
        else:
            return 0.30

    def _enforce_kp_balance(self, candidates: list[dict], needs_tech: bool, needs_soft: bool) -> list[dict]:
        if not (needs_tech and needs_soft):
            return candidates

        k_items = [c for c in candidates if "K" in c.get("test_type", [])]
        p_items = [c for c in candidates if "P" in c.get("test_type", [])]

        target_k = min(3, len(k_items))
        target_p = min(3, len(p_items))

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

        in_priority = {c["url"] for c in priority}
        for c in candidates:
            if c["url"] not in in_priority:
                priority.append(c)

        return priority

    def _llm_rerank(self, query_text: str, candidates: list[dict], k: int) -> list[dict]:
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
        max_retries = 5
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
                        item = dict(cand)
                        item["_explanation"] = explanation
                        result.append(item)

                seen = {c["url"].strip().rstrip("/").lower() for c in result}
                for c in candidates:
                    if c["url"].strip().rstrip("/").lower() not in seen and len(result) < MAX_RECOMMENDATIONS:
                        item = dict(c)
                        item.setdefault("_explanation", "")
                        result.append(item)
                return result

            except Exception as e:
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

    SENIORITY_BOOST = {
        "executive": ["leadership-report", "opq", "enterprise-leadership", "global-skills", "executive-scenarios", "hipo"],
        "senior":    ["professional-7-1", "professional-7-0", "opq"],
    }
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
        seniority = parsed_jd.get("seniority", "mid").lower()
        domain = parsed_jd.get("domain", "general").lower()

        boost_slugs = []
        boost_slugs.extend(self.SENIORITY_BOOST.get(seniority, []))
        boost_slugs.extend(self.DOMAIN_BOOST.get(domain, []))

        if not boost_slugs:
            return candidates

        existing_urls = {c["url"].lower() for c in candidates}
        min_score = min((c.get("_score", 0.0) for c in candidates), default=0.0)
        inject_score = max(0.0, min_score * 0.5)

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
            candidates = injected + candidates

        return candidates

    def retrieve(
        self,
        query_vec: np.ndarray,
        query_text: str,
        parsed_jd: dict,
        use_llm_rerank: bool = True,
    ) -> list[dict]:
        query_tokens = self._tokenize(query_text)

        dense_results = self._dense_search(query_vec, CANDIDATE_SIZE)
        dense_map = {idx: score for idx, score in dense_results}

        bm25_results = self._bm25_search(query_tokens, CANDIDATE_SIZE)
        bm25_map = {idx: score for idx, score in bm25_results}

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

        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = []
        score_map = {}
        for final_score, item in scored[:CANDIDATE_SIZE]:
            item_copy = dict(item)
            item_copy["_score"] = round(float(final_score), 4)
            item_copy.setdefault("_explanation", "")
            candidates.append(item_copy)
            score_map[item["url"]] = round(float(final_score), 4)

        candidates = self._inject_domain_boost(candidates, parsed_jd)
        for c in candidates:
            score_map.setdefault(c["url"], c.get("_score", 0.0))

        needs_tech = parsed_jd.get("needs_tech", False)
        needs_soft = parsed_jd.get("needs_soft", False)
        candidates = self._enforce_kp_balance(candidates, needs_tech, needs_soft)

        if use_llm_rerank and len(candidates) > MIN_RECOMMENDATIONS:
            candidates = self._llm_rerank(
                parsed_jd.get("canonical_query", query_text),
                candidates,
                MAX_RECOMMENDATIONS,
            )

        for c in candidates:
            if "_score" not in c or c["_score"] == 0.0:
                c["_score"] = score_map.get(c["url"], 0.0)

        seen_urls = set()
        final = []
        for c in candidates:
            if c["url"] not in seen_urls:
                seen_urls.add(c["url"])
                final.append(c)
            if len(final) >= MAX_RECOMMENDATIONS:
                break

        return final[:MAX_RECOMMENDATIONS]
