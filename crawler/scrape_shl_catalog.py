import requests
from bs4 import BeautifulSoup
import json
import csv
import sqlite3
import time
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PAGE_SIZE = 12
TOTAL_PAGES = 32


def fetch_page(start: int, type_filter: int = 1, retries: int = 3) -> str | None:
    params = {"start": start, "type": type_filter}
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed (start={start}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_catalog_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    assessments = []

    rows = soup.select("tr[data-course-id]")
    if not rows:
        rows = soup.select(".custom__table-responsive tr")

    for row in rows:
        try:
            name_el = row.select_one("td.custom__table-heading__title a")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            href = name_el.get("href", "")
            if not href:
                continue
            url = href if href.startswith("http") else f"https://www.shl.com{href}"

            general_cells = row.select("td.custom__table-heading__general")
            remote_testing = False
            adaptive_irt = False

            if len(general_cells) >= 1:
                circle = general_cells[0].select_one("span.catalogue__circle")
                remote_testing = circle is not None and "-yes" in circle.get("class", [])

            if len(general_cells) >= 2:
                circle = general_cells[1].select_one("span.catalogue__circle")
                adaptive_irt = circle is not None and "-yes" in circle.get("class", [])

            type_spans = row.select("span.product-catalogue__key")
            test_types = [s.get_text(strip=True) for s in type_spans if s.get_text(strip=True)]

            duration = None
            for cell in row.select("td"):
                text = cell.get_text(strip=True)
                m = re.match(r"^(\d+)\s*$", text)
                if m:
                    val = int(m.group(1))
                    if 5 <= val <= 180:
                        duration = val
                        break

            if name:
                assessments.append({
                    "name": name,
                    "url": url,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "test_type": test_types,
                    "duration_minutes": duration,
                    "description": "",
                })

        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            continue

    return assessments


def scrape_detail_page(url: str) -> dict:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        desc = ""
        for selector in [
            ".product-hero__description",
            ".product-catalogue__description",
            "[class*='description'] p",
            "main .rich-text p",
            "main p",
        ]:
            el = soup.select_one(selector)
            if el:
                desc = el.get_text(strip=True)
                if len(desc) > 30:
                    break

        return {"description": desc[:600]}
    except Exception as e:
        logger.debug(f"Detail fetch failed for {url}: {e}")
        return {"description": ""}


def scrape_all() -> list[dict]:
    logger.info("Starting SHL catalog scrape (Individual Test Solutions, type=1)...")
    all_assessments = []
    seen_urls = set()

    for page_num in range(TOTAL_PAGES + 1):  # +1 safety buffer
        start = page_num * PAGE_SIZE
        logger.info(f"Page {page_num + 1}/{TOTAL_PAGES} — start={start}")

        html = fetch_page(start, type_filter=1)
        if not html:
            logger.warning(f"Failed to fetch page start={start}, skipping")
            continue

        items = parse_catalog_page(html)
        if not items:
            logger.info(f"No items on start={start} — stopping")
            break

        new_count = 0
        for item in items:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_assessments.append(item)
                new_count += 1

        logger.info(f"  → {new_count} new items (total so far: {len(all_assessments)})")

        if new_count == 0:
            logger.info("No new items — catalog exhausted")
            break

        time.sleep(0.5)

    logger.info(f"Scraped {len(all_assessments)} unique assessments total")

    logger.info("Enriching first 60 items with detail page descriptions...")
    for i, a in enumerate(all_assessments[:60]):
        detail = scrape_detail_page(a["url"])
        a["description"] = detail["description"]
        if i % 10 == 0:
            logger.info(f"  Enriched {i}/{min(60, len(all_assessments))}")
        time.sleep(0.3)

    return all_assessments


def save_data(assessments: list[dict]):
    json_path = DATA_DIR / "catalog_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    csv_path = DATA_DIR / "catalog_clean.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "name", "url", "description", "test_type",
            "remote_testing", "adaptive_irt", "duration_minutes",
        ])
        writer.writeheader()
        for i, a in enumerate(assessments, 1):
            writer.writerow({
                "id": i,
                "name": a["name"],
                "url": a["url"],
                "description": a.get("description", ""),
                "test_type": ",".join(a.get("test_type", [])),
                "remote_testing": a.get("remote_testing", False),
                "adaptive_irt": a.get("adaptive_irt", False),
                "duration_minutes": a.get("duration_minutes", ""),
            })
    logger.info(f"Saved {len(assessments)} rows to {csv_path}")

    db_path = DATA_DIR / "shl_catalog.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            description TEXT,
            test_type TEXT,
            remote_testing INTEGER,
            adaptive_irt INTEGER,
            duration_minutes INTEGER,
            embedding_text TEXT
        )
    """)
    conn.execute("DELETE FROM assessments")
    for i, a in enumerate(assessments, 1):
        embedding_text = (
            f"{a['name']}. "
            f"Test type: {','.join(a.get('test_type', []))}. "
            f"Duration: {a.get('duration_minutes', 'unknown')} minutes. "
            f"{a.get('description', '')}"
        ).strip()
        conn.execute(
            "INSERT OR REPLACE INTO assessments VALUES (?,?,?,?,?,?,?,?,?)",
            (
                i, a["name"], a["url"], a.get("description", ""),
                ",".join(a.get("test_type", [])),
                int(a.get("remote_testing", False)),
                int(a.get("adaptive_irt", False)),
                a.get("duration_minutes"),
                embedding_text,
            )
        )
    conn.commit()
    conn.close()
    logger.info(f"SQLite DB saved to {db_path}")


def validate(assessments: list[dict]):
    assert len(assessments) >= 377, (
        f"Only {len(assessments)} assessments scraped — expected ≥ 377. "
        "SHL catalog may have changed. Check HTML selectors."
    )
    logger.info(f"✅ Validation passed: {len(assessments)} assessments (≥ 377)")


if __name__ == "__main__":
    assessments = scrape_all()
    validate(assessments)
    save_data(assessments)
    print(f"\n✅ Done! {len(assessments)} Individual Test Solutions saved to {DATA_DIR}")
    print(f"   Sample: {assessments[0]['name']} — {assessments[0]['url']}")
