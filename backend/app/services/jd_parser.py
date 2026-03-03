"""
JD Parser Service: handles plain query, raw JD text, and URL inputs.
Uses trafilatura for URL extraction + Gemini 1.5 Flash for structured summarization.
"""
import logging
import json
import re
from typing import Optional

import requests
import trafilatura
from google import genai as google_genai
from app.config import GEMINI_API_KEY, TECH_KEYWORDS, SOFT_KEYWORDS

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
JD_PARSE_PROMPT = """You are an expert HR analyst. Analyze this job description/query and extract structured info.
Return ONLY a valid JSON object (no markdown) with this structure:
{{
  "role": "<job title>",
  "seniority": "<entry|mid|senior|executive>",
  "hard_skills": ["<skill1>", "<skill2>"],
  "soft_skills": ["<skill1>", "<skill2>"],
  "domain": "<tech|sales|hr|finance|marketing|media|operations|consultant|general>",
  "duration_constraint_minutes": <number or null>,
  "canonical_query": "<concise 2-3 sentence query capturing role, skills, and constraints>"
}}

Domain selection guide:
- tech: software engineering, IT, data science, developer roles
- sales: sales, business development, account management
- hr: human resources, recruiting, L&D, I/O psychology, talent management, consulting firms
- finance: accounting, finance, banking, investment roles
- marketing: marketing, brand, advertising, content, SEO roles
- media: radio, broadcast, journalism, TV, film, communications
- operations: operations, supply chain, logistics, admin support
- consultant: management consulting, strategy, advisory, professional services
- general: anything else

Job description/query:
{text}
"""


def get_gemini_client():
    return google_genai.Client(api_key=GEMINI_API_KEY)


def fetch_url_text(url: str) -> str:
    """Fetch and extract main text from a URL using trafilatura."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text:
            return text[:5000]  # cap length
        # fallback: raw stripped HTML text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:5000]
    except Exception as e:
        logger.warning(f"URL fetch failed for {url}: {e}")
        return ""


def parse_with_llm(text: str) -> dict:
    """Use Gemini Flash to extract structured JD fields."""
    prompt = JD_PARSE_PROMPT.format(text=text[:3000])
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            content = response.text.strip()
            # Strip markdown code fences if present
            content = re.sub(r"```(?:json)?\n?", "", content).strip().rstrip("```").strip()
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = 15 * (attempt + 1)
                    logger.warning(f"429 Rate limit hit in JD parser (attempt {attempt+1}), sleeping for {sleep_time}s...")
                    import time
                    time.sleep(sleep_time)
                    continue
            logger.warning(f"LLM JD parsing failed: {e}")
            return {}
    return {}


def has_tech_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in TECH_KEYWORDS)


def has_soft_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in SOFT_KEYWORDS)


def parse_request(
    query: Optional[str] = None,
    jd_text: Optional[str] = None,
    jd_url: Optional[str] = None,
) -> dict:
    """
    Process input and return a unified dict:
    {
      raw_text: str,        # original input text
      canonical_query: str, # enriched query for embedding/BM25
      hard_skills: list,
      soft_skills: list,
      duration_constraint: int | None,
      needs_tech: bool,
      needs_soft: bool,
      seniority: str,
    }
    """
    # Step 1: get raw text
    raw_text = ""
    if jd_url:
        logger.info(f"Fetching JD from URL: {jd_url}")
        raw_text = fetch_url_text(jd_url)
        if not raw_text:
            raw_text = jd_url  # fallback to URL string

    elif jd_text:
        raw_text = jd_text

    elif query:
        raw_text = query

    if not raw_text:
        raise ValueError("No input provided. Supply query, jd_text, or jd_url.")

    # Step 2: LLM structural parse
    parsed = {}
    if len(raw_text) > 50:  # only for substantial text
        parsed = parse_with_llm(raw_text)

    # Step 3: build canonical query
    canonical = parsed.get("canonical_query", "")
    if not canonical:
        canonical = raw_text[:500]

    hard_skills = parsed.get("hard_skills", [])
    soft_skills = parsed.get("soft_skills", [])
    duration = parsed.get("duration_constraint_minutes")
    if isinstance(duration, str):
        try:
            duration = int(re.search(r"\d+", duration).group())
        except Exception:
            duration = None

    return {
        "raw_text": raw_text,
        "canonical_query": canonical,
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "duration_constraint": duration,
        "needs_tech": has_tech_signal(raw_text) or bool(hard_skills),
        "needs_soft": has_soft_signal(raw_text) or bool(soft_skills),
        "seniority": parsed.get("seniority", "mid"),
        "domain": parsed.get("domain", "general"),
    }
