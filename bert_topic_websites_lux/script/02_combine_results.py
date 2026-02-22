"""
Luxembourg Website Topic Analysis - Step 2: Combine Results
============================================================
Combines BERTopic results from all years and generates statistics
for the visualization.

This script:
1. Loads topic summaries from each year
2. Tracks topic evolution over time
3. Identifies potential "government" topics
4. Generates stats.json for the blog post visualization

Author: Julio Garbers with contributions from Claude
Date: January 2025
"""

import json
from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/output")
FINAL_OUTPUT = OUTPUT_DIR / "stats.json"

# Keywords that might indicate government websites
GOVERNMENT_KEYWORDS = [
    "government",
    "gouvernement",
    "regierung",
    "minister",
    "ministry",
    "public",
    "official",
    "state",
    "national",
    "luxembourg",
    "administration",
    "commune",
    "municipality",
    "city",
    "ville",
    "stadt",
    "service",
    "citizen",
    "law",
    "regulation",
]


# =============================================================================
# Data Loading
# =============================================================================


def load_yearly_results() -> dict[int, dict]:
    """Load topic summaries from all years."""
    print("[LOAD] Loading yearly results...")

    yearly_data = {}
    years = sorted([int(d.name) for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()])

    for year in years:
        year_dir = OUTPUT_DIR / str(year)

        # Load topic summary
        summary_file = year_dir / "topic_summary.csv"
        if not summary_file.exists():
            print(f"   [WARN] Missing: {summary_file}")
            continue

        topic_summary = pl.read_csv(summary_file)

        # Load metadata
        metadata_file = year_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load website topics for classification
        website_topics_file = year_dir / "website_topics.parquet"
        if website_topics_file.exists():
            website_topics = pl.read_parquet(website_topics_file)
        else:
            website_topics = None

        yearly_data[year] = {
            "topic_summary": topic_summary,
            "metadata": metadata,
            "website_topics": website_topics,
        }

        n_topics = len(topic_summary.filter(pl.col("topic_id") != -1))
        print(f"   {year}: {n_topics} topics, {metadata.get('n_websites', '?')} websites")

    return yearly_data


# =============================================================================
# Analysis Functions
# =============================================================================


def identify_government_topics(topic_summary: pl.DataFrame) -> list[int]:
    """
    Identify topics that likely represent government websites.
    Returns list of topic IDs.
    """
    government_topics = []

    for row in topic_summary.iter_rows(named=True):
        if row["topic_id"] == -1:
            continue

        # Check if any government keywords appear in top words
        top_words = row["top_words"].lower() if row["top_words"] else ""

        for keyword in GOVERNMENT_KEYWORDS:
            if keyword in top_words:
                government_topics.append(row["topic_id"])
                break

    return government_topics


def compute_topic_distribution(yearly_data: dict[int, dict]) -> list[dict]:
    """Compute topic distribution over years."""
    yearly_stats = []

    for year, data in sorted(yearly_data.items()):
        topic_summary = data["topic_summary"]
        metadata = data["metadata"]

        # Filter out outliers
        valid_topics = topic_summary.filter(pl.col("topic_id") != -1)

        # Get top 5 topics by count
        top_topics = valid_topics.sort("count", descending=True).head(5)

        # Identify government topics
        gov_topics = identify_government_topics(topic_summary)

        # Calculate government website percentage
        if data["website_topics"] is not None:
            website_topics = data["website_topics"]
            n_gov = website_topics.filter(pl.col("topic").is_in(gov_topics)).height
            n_total = website_topics.height
            gov_pct = round(n_gov / n_total * 100, 1) if n_total > 0 else 0
        else:
            n_gov = 0
            gov_pct = 0

        yearly_stats.append({
            "year": year,
            "n_websites": metadata.get("n_websites", 0),
            "n_topics": metadata.get("n_topics", 0),
            "n_outliers": metadata.get("n_outliers", 0),
            "outlier_pct": round(
                metadata.get("n_outliers", 0) / metadata.get("n_websites", 1) * 100, 1
            ),
            "top_topics": [
                {
                    "topic_id": row["topic_id"],
                    "name": row["name"],
                    "count": row["count"],
                    "top_words": row["top_words"],
                }
                for row in top_topics.iter_rows(named=True)
            ],
            "government_topics": gov_topics,
            "government_count": n_gov,
            "government_pct": gov_pct,
        })

    return yearly_stats


def find_common_topics(yearly_data: dict[int, dict]) -> list[dict]:
    """
    Find topics that appear across multiple years (based on keyword similarity).
    This is a simple approach - could be enhanced with embedding similarity.
    """
    # Collect all topic keywords by year
    all_topics = []

    for year, data in yearly_data.items():
        for row in data["topic_summary"].iter_rows(named=True):
            if row["topic_id"] == -1:
                continue

            all_topics.append({
                "year": year,
                "topic_id": row["topic_id"],
                "name": row["name"],
                "top_words": set(row["top_words"].lower().split(", ")) if row["top_words"] else set(),
                "count": row["count"],
            })

    # Group topics by keyword overlap
    # This is a simplified approach - topics with >50% keyword overlap are considered similar
    topic_clusters = []

    for topic in all_topics:
        matched = False
        for cluster in topic_clusters:
            # Check overlap with cluster representative
            overlap = len(topic["top_words"] & cluster["keywords"]) / max(
                len(topic["top_words"]), 1
            )
            if overlap > 0.3:  # 30% overlap threshold
                cluster["occurrences"].append({
                    "year": topic["year"],
                    "topic_id": topic["topic_id"],
                    "count": topic["count"],
                })
                cluster["keywords"] = cluster["keywords"] | topic["top_words"]
                matched = True
                break

        if not matched:
            topic_clusters.append({
                "name": topic["name"],
                "keywords": topic["top_words"],
                "occurrences": [{
                    "year": topic["year"],
                    "topic_id": topic["topic_id"],
                    "count": topic["count"],
                }],
            })

    # Filter to topics appearing in multiple years
    recurring_topics = [
        {
            "name": cluster["name"],
            "keywords": ", ".join(sorted(cluster["keywords"])[:10]),
            "years_present": len(set(o["year"] for o in cluster["occurrences"])),
            "total_count": sum(o["count"] for o in cluster["occurrences"]),
            "occurrences": sorted(cluster["occurrences"], key=lambda x: x["year"]),
        }
        for cluster in topic_clusters
        if len(set(o["year"] for o in cluster["occurrences"])) >= 3  # Present in 3+ years
    ]

    # Sort by number of years present
    recurring_topics.sort(key=lambda x: (-x["years_present"], -x["total_count"]))

    return recurring_topics[:20]  # Top 20 recurring topics


def extract_government_websites(yearly_data: dict[int, dict]) -> pl.DataFrame:
    """
    Extract all websites classified under government topics.
    Useful for follow-up analysis.
    """
    gov_websites = []

    for year, data in yearly_data.items():
        if data["website_topics"] is None:
            continue

        gov_topics = identify_government_topics(data["topic_summary"])

        if gov_topics:
            year_gov = data["website_topics"].filter(
                pl.col("topic").is_in(gov_topics)
            ).with_columns(pl.lit(year).alias("year"))

            gov_websites.append(year_gov)

    if gov_websites:
        return pl.concat(gov_websites)
    else:
        return pl.DataFrame()


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Luxembourg Website Topic Analysis - Combine Results")
    print("=" * 70)

    # Load all yearly results
    yearly_data = load_yearly_results()

    if not yearly_data:
        print("\n[ERROR] No yearly results found!")
        return

    # Compute statistics
    print("\n[ANALYZE] Computing statistics...")
    yearly_stats = compute_topic_distribution(yearly_data)

    print("\n[ANALYZE] Finding recurring topics...")
    recurring_topics = find_common_topics(yearly_data)

    # Summary statistics
    years = sorted(yearly_data.keys())
    total_websites = sum(data["metadata"].get("n_websites", 0) for data in yearly_data.values())

    summary = {
        "first_year": min(years),
        "last_year": max(years),
        "years_covered": len(years),
        "total_website_years": total_websites,
        "avg_topics_per_year": round(
            sum(s["n_topics"] for s in yearly_stats) / len(yearly_stats), 1
        ),
    }

    # Build final stats
    stats = {
        "summary": summary,
        "yearly": yearly_stats,
        "recurring_topics": recurring_topics,
    }

    # Save stats.json
    print(f"\n[SAVE] Saving stats to {FINAL_OUTPUT}...")
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved: {FINAL_OUTPUT}")

    # Extract and save government websites for follow-up analysis
    print("\n[EXTRACT] Extracting government websites...")
    gov_websites = extract_government_websites(yearly_data)

    if len(gov_websites) > 0:
        gov_file = OUTPUT_DIR / "government_websites.parquet"
        gov_websites.write_parquet(gov_file, compression="zstd", compression_level=10)
        print(f"   [OK] Found {len(gov_websites):,} government website-years")
        print(f"   [OK] Saved: {gov_file}")
    else:
        print("   [INFO] No government websites identified")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nYears: {summary['first_year']} - {summary['last_year']}")
    print(f"Total website-years: {summary['total_website_years']:,}")
    print(f"Average topics per year: {summary['avg_topics_per_year']}")

    print("\n[TOP RECURRING TOPICS]")
    for i, topic in enumerate(recurring_topics[:10], 1):
        print(f"\n   {i}. {topic['name']}")
        print(f"      Years present: {topic['years_present']}")
        print(f"      Keywords: {topic['keywords'][:80]}...")

    print("\n[GOVERNMENT WEBSITES BY YEAR]")
    for stat in yearly_stats:
        if stat["government_pct"] > 0:
            print(f"   {stat['year']}: {stat['government_count']:,} ({stat['government_pct']}%)")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"   1. Review topics in output/*/topic_summary.csv")
    print(f"   2. Use government_websites.parquet for language analysis")
    print(f"   3. Build visualization with stats.json")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
