# scripts/01_prepare_data.py
import json
import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm
import kagglehub

INPUT_FILE  = "data/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000

os.makedirs("data", exist_ok=True)

# Check if input file exists; if not, download it from kaggle and move it to the data folder
if not os.path.exists(INPUT_FILE):
    print(f"File {INPUT_FILE} not found. Downloading via kagglehub...")
    download_path = kagglehub.dataset_download("Cornell-University/arxiv")
    downloaded_file = os.path.join(download_path, "arxiv-metadata-oai-snapshot.json")
    
    if os.path.exists(downloaded_file):
        print(f"Moving downloaded file to {INPUT_FILE}...")
        shutil.copy2(downloaded_file, INPUT_FILE)
    else:
        print("Error: Downloaded dataset does not contain the expected JSON file.")
        sys.exit(1)

def extract_year(paper: dict) -> int:
    """
    Extract the year from the first version of the paper — this is the arXiv publication date.
    update_date is the date of the last update, which could be years later.
    created format: "Mon, 2 Apr 2007 19:18:42 GMT"
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]  # "Mon, 2 Apr 2007 19:18:42 GMT"
            # The year is at the 4th position after splitting by space
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    # Fallback: update_date in "YYYY-MM-DD" format
    return int(paper.get("update_date", "2000-01-01")[:4])

def format_authors(paper: dict) -> str:
    """
    authors_parsed is a structured list [["Last Name", "Initials", ""]].
    We assemble it into a readable string "Last Name I., Last Name I."
    If authors_parsed is missing, we take the raw authors string.
    """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # no more than 10 authors
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last}{first}".strip())
        return ", ".join(parts)
    # Fallback: raw authors string
    return paper.get("authors", "").replace("\\n", " ")

records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading dataset"):
        if len(records) >= MAX_RECORDS:
            break
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()

        # Skip records without abstract or title
        if not abstract or not title:
            continue

        # categories can contain multiple categories separated by space: "cs.LG cs.AI"
        # Take the first one as primary
        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]

        records.append({
            "id":       paper["id"],
            "title":    title.replace("\\n", " ").strip(),
            "abstract": abstract.replace("\\n", " ").strip(),
            "authors":  format_authors(paper),
            "year":     extract_year(paper),
            "category": primary_category,
        })

df = pd.DataFrame(records)
print(f"\nLoaded papers: {len(df)}")
print(f"\nCategory distribution (top 10):")
print(df["category"].value_counts().head(10))
print(f"\nYear distribution:")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nSample record:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")

# Clean up temporary zip file if exists
if os.path.exists("arxiv.zip"):
    print("\nCleaning up: removing arxiv.zip...")
    os.remove("arxiv.zip")

print("\nData preparation complete.")