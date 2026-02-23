"""
Luxembourg Website Topic Analysis - Step 3: Blog Statistics
=============================================================
Generates the blog's stats.json from pipeline output and manual
sector assignments.

Reads:
  - output/topics_over_time.csv (per-topic per-year frequencies)
  - output/metadata.json (yearly totals)
  - sector_mapping.json (manual topic → sector assignments)

Writes:
  - output/blog_stats.json (ready to copy to blog directory)

Author: Julio Garbers with contributions from Claude
Date: February 2026
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

# ===================================================================
# Paths
# ===================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MAPPING_FILE = BASE_DIR / "sector_mapping.json"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
TOPICS_OVER_TIME_FILE = OUTPUT_DIR / "topics_over_time.csv"
BLOG_STATS_FILE = OUTPUT_DIR / "blog_stats.json"

# ===================================================================
# Config
# ===================================================================

START_YEAR = 2016
END_YEAR = 2024
LATEST_YEAR = 2024


# ===================================================================
# Loading
# ===================================================================


def load_sector_mapping() -> dict[int, dict]:
    """Load manual topic → sector assignments."""
    with open(MAPPING_FILE) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_metadata() -> dict:
    """Load pipeline metadata (yearly totals)."""
    with open(METADATA_FILE) as f:
        return json.load(f)


def load_topics_over_time() -> dict[int, dict[int, int]]:
    """Load topics_over_time.csv → {topic_id: {year: frequency}}."""
    result: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    with open(TOPICS_OVER_TIME_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_id = int(row["Topic"])
            if topic_id == -1:
                continue
            year = int(row["Timestamp"][:4])
            freq = int(row["Frequency"])
            result[topic_id][year] = freq
    return dict(result)


# ===================================================================
# Building blog data
# ===================================================================


def build_topics_latest(
    mapping: dict[int, dict],
    topics_over_time: dict[int, dict[int, int]],
) -> list[dict]:
    """Build topics list for treemap (latest year only)."""
    topics = []
    for topic_id, info in sorted(mapping.items()):
        count = topics_over_time.get(topic_id, {}).get(LATEST_YEAR, 0)
        if count > 0:
            topics.append(
                {
                    "id": topic_id,
                    "count": count,
                    "sector": info["sector"],
                    "label": info["label"],
                    "words": info["words"],
                }
            )
    return topics


def build_sector_evolution(
    mapping: dict[int, dict],
    topics_over_time: dict[int, dict[int, int]],
    metadata: dict,
) -> dict:
    """Build sector-level evolution with shares per year."""
    years = list(range(START_YEAR, END_YEAR + 1))

    # Aggregate topic frequencies by sector per year
    sector_counts: dict[str, dict[int, int]] = defaultdict(
        lambda: {y: 0 for y in years}
    )
    for topic_id, year_counts in topics_over_time.items():
        sector = mapping.get(topic_id, {}).get("sector", "Other")
        for year in years:
            sector_counts[sector][year] += year_counts.get(year, 0)

    # Get n_classified per year from metadata
    n_classified = {}
    for year_str, y_data in metadata["yearly"].items():
        year = int(year_str)
        if START_YEAR <= year <= END_YEAR:
            n_classified[year] = y_data["n_classified"]

    # Sort sectors by total count (descending), keep "Other" last
    sector_totals = {s: sum(counts.values()) for s, counts in sector_counts.items()}
    named_sectors = sorted(
        [s for s in sector_totals if s != "Other"],
        key=lambda s: -sector_totals[s],
    )
    all_sectors = named_sectors + (["Other"] if "Other" in sector_totals else [])

    sectors = []
    for sector in all_sectors:
        counts = sector_counts[sector]
        values = [counts[y] for y in years]
        shares = [
            round(counts[y] / n_classified[y] * 100, 1) if n_classified.get(y) else 0
            for y in years
        ]
        sectors.append({"name": sector, "values": values, "shares": shares})

    return {"years": years, "sectors": sectors}


def build_yearly(metadata: dict) -> list[dict]:
    """Build yearly summary stats."""
    yearly = []
    for year_str in sorted(metadata["yearly"]):
        year = int(year_str)
        if year < START_YEAR:
            continue
        y_data = metadata["yearly"][year_str]
        yearly.append(
            {
                "year": year,
                "n_websites": y_data["n_websites"],
                "n_classified": y_data["n_classified"],
                "outlier_pct": y_data["outlier_pct"],
            }
        )
    return yearly


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    print("Loading data...", flush=True)
    mapping = load_sector_mapping()
    metadata = load_metadata()
    topics_over_time = load_topics_over_time()

    print(f"Sector mapping: {len(mapping)} topics across "
          f"{len(set(v['sector'] for v in mapping.values()))} sectors", flush=True)
    print(f"Topics over time: {len(topics_over_time)} topics", flush=True)

    # Build blog data
    topics_latest = build_topics_latest(mapping, topics_over_time)
    sector_evolution = build_sector_evolution(mapping, topics_over_time, metadata)
    yearly = build_yearly(metadata)

    # Summary stats
    n_mapped = sum(t["count"] for t in topics_latest)
    n_classified = metadata["yearly"][str(LATEST_YEAR)]["n_classified"]

    blog_stats = {
        "summary": {
            "first_year": START_YEAR,
            "last_year": END_YEAR,
            "total_website_years": metadata["total_website_years"],
            "n_topics": metadata["n_topics"],
            "outlier_pct": metadata["outlier_pct"],
            "n_classified_latest": n_classified,
            "n_mapped_latest": n_mapped,
        },
        "yearly": yearly,
        "topics_2024": topics_latest,
        "sector_evolution": sector_evolution,
    }

    with open(BLOG_STATS_FILE, "w") as f:
        json.dump(blog_stats, f, indent=2, ensure_ascii=False)

    print(f"\nBlog stats written to {BLOG_STATS_FILE}", flush=True)
    print(f"  Years: {START_YEAR}–{END_YEAR}", flush=True)
    print(f"  Topics mapped: {len(topics_latest)} ({n_mapped} websites in {LATEST_YEAR})", flush=True)
    print(f"  Sectors in evolution: {len(sector_evolution['sectors'])}", flush=True)

    # Print sector evolution summary for latest year
    print(f"\nSector shares in {LATEST_YEAR}:", flush=True)
    for s in sector_evolution["sectors"]:
        latest_val = s["values"][-1]
        latest_share = s["shares"][-1]
        print(f"  {s['name']:20s}  {latest_val:5d}  ({latest_share:5.1f}%)", flush=True)


if __name__ == "__main__":
    main()
