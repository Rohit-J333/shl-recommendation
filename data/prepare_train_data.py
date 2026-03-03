"""
Prepare train_set.csv from the Excel dataset.
Run once: python data/prepare_train_data.py
"""
import csv
from pathlib import Path
from collections import defaultdict
import openpyxl

EXCEL_PATH = Path(__file__).parent.parent / "Gen_AI Dataset.xlsx"
OUT_CSV = Path(__file__).parent / "train_set.csv"


def main():
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb["Train-Set"]

    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:  # header
            continue
        q, url = row[0], row[1]
        if q and url:
            rows.append({"Query": str(q).strip(), "Assessment_url": str(url).strip()})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} train rows to {OUT_CSV}")
    # Print summary
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["Query"][:60]].append(r["Assessment_url"])
    print(f"   Unique queries: {len(by_query)}")


if __name__ == "__main__":
    main()
