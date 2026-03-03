"""
Enrich catalog_clean.csv with real descriptions and duration data
scraped from individual SHL product pages.

Handles:
- Removing navigation/header garbage from descriptions
- Multiple fallback selectors for description text
- Duration extraction via regex on page text
- Resumable: prints progress, can be re-run safely
"""
import csv
import re
import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CATALOG_CSV = BASE_DIR / "data" / "catalog_clean.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Nav/menu text markers that indicate scraped garbage
NAV_MARKERS = [
    "careers", "our culture", "join shl", "practice tests",
    "contact us", "sign in", "log in", "our teams", "our people",
    "latest jobs", "privacy policy", "cookie policy",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def looks_like_nav(text: str) -> bool:
    """Return True if text looks like scraped navigation/footer, not a description."""
    t = text.lower()
    hits = sum(1 for m in NAV_MARKERS if m in t)
    return hits >= 2


def extract_detail(url: str) -> tuple[str, int | None]:
    """
    Fetch a product page and return (description, duration_minutes).
    Returns ("", None) on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.debug(f"Fetch failed {url}: {e}")
        return "", None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove nav/header/footer noise before any text extraction
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    # ── Description extraction ──────────────────────────────────────────────
    description = ""

    # Priority selectors (SHL uses different class names across product pages)
    desc_selectors = [
        ".product-hero__description",
        ".product-hero__content > p",
        ".product-catalogue__description",
        ".product-overview__description",
        ".product-overview p",
        "[class*='hero'] p",
        "[class*='description'] p",
        "article p",
    ]

    for sel in desc_selectors:
        el = soup.select_one(sel)
        if el:
            text = clean_text(el.get_text())
            if 40 < len(text) < 2000 and not looks_like_nav(text):
                description = text[:700]
                break

    # Fallback: first substantive paragraph in main content
    if not description:
        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(id=re.compile(r"main|content", re.I))
            or soup
        )
        for p in main.find_all("p"):
            text = clean_text(p.get_text())
            if 60 < len(text) < 1500 and not looks_like_nav(text):
                description = text[:700]
                break

    # ── Duration extraction ─────────────────────────────────────────────────
    duration = None
    full_text = clean_text(soup.get_text())

    duration_patterns = [
        # "Approximate Completion Time: 20 minutes" or "Duration: 25 mins"
        r"(?:approximate(?:ly)?[:\s]+)?(?:completion\s+time|duration|time(?:\s+to\s+complete)?)[:\s]+(\d+)(?:\s*[-–]\s*\d+)?\s*(?:minutes?|mins?)",
        # "20-25 minutes to complete"
        r"(\d+)(?:\s*[-–]\s*\d+)?\s*(?:minutes?|mins?)\s*(?:to\s+complete|completion|untimed)?",
        # "approximately 30 minutes"
        r"approximately\s+(\d+)\s*(?:minutes?|mins?)",
    ]

    for pattern in duration_patterns:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 5 <= val <= 180:
                duration = val
                break

    return description, duration


def enrich():
    # Load catalog
    with open(CATALOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    logger.info(f"Loaded {len(rows)} assessments from catalog")

    enriched = 0
    failed = 0

    for i, row in enumerate(rows):
        url = row["url"]
        logger.info(f"[{i+1}/{len(rows)}] {row['name'][:50]}")

        desc, dur = extract_detail(url)

        if desc:
            row["description"] = desc
            enriched += 1
        elif not row.get("description") or looks_like_nav(row.get("description", "")):
            row["description"] = ""  # clear garbage
            failed += 1

        if dur is not None:
            row["duration_minutes"] = str(dur)
        # Keep existing duration if we couldn't extract a new one
        # (don't overwrite known values)

        time.sleep(0.4)  # polite crawl delay

        # Save checkpoint every 50 rows
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"  Checkpoint saved. enriched={enriched}, failed={failed}")

    # Final save
    with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"\nDone. {enriched}/{len(rows)} got descriptions, {failed} empty/failed")

    # Stats
    with_desc = sum(1 for r in rows if r.get("description", "").strip())
    with_dur = sum(1 for r in rows if r.get("duration_minutes", "").strip().isdigit())
    logger.info(f"Final: {with_desc} descriptions, {with_dur} durations")


if __name__ == "__main__":
    enrich()
