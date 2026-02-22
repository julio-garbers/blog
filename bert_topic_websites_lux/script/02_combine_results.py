"""
Luxembourg Website Topic Analysis - Step 2: Combine Results
============================================================
Combines BERTopic results from all years and generates statistics
for the visualization.

Updated for page-level analysis: topics are discovered at the page level,
then aggregated to website-year level. A website can appear in multiple
topics through its different pages.

This script:
1. Loads topic summaries from each year
2. Tracks topic evolution over time (using website counts)
3. Identifies potential "government" topics
4. Generates stats.json for the blog post visualization

Author: Julio Garbers with contributions from Claude
Date: February 2026
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
    print("[LOAD] Loading yearly results...", flush=True)

    yearly_data = {}
    years = sorted(
        [int(d.name) for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    for year in years:
        year_dir = OUTPUT_DIR / str(year)

        # Load topic summary
        summary_file = year_dir / "topic_summary.csv"
        if not summary_file.exists():
            print(f"   [WARN] Missing: {summary_file}", flush=True)
            continue

        topic_summary = pl.read_csv(summary_file)

        # Load metadata
        metadata_file = year_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load page-level topics (for richer website-topic analysis)
        page_topics_file = year_dir / "page_topics.parquet"
        page_topics = None
        if page_topics_file.exists():
            page_topics = pl.read_parquet(page_topics_file)

        # Load website-level summary
        website_topics_file = year_dir / "website_topics.parquet"
        website_topics = None
        if website_topics_file.exists():
            website_topics = pl.read_parquet(website_topics_file)

        yearly_data[year] = {
            "topic_summary": topic_summary,
            "metadata": metadata,
            "page_topics": page_topics,
            "website_topics": website_topics,
        }

        n_topics = len(topic_summary.filter(pl.col("topic_id") != -1))
        n_pages = metadata.get("n_pages", "?")
        n_websites = metadata.get("n_websites", "?")
        print(
            f"   {year}: {n_topics} topics, {n_pages} pages, {n_websites} websites",
            flush=True,
        )

    return yearly_data


# =============================================================================
# Analysis Functions
# =============================================================================


def _get_count_col(topic_summary: pl.DataFrame) -> str:
    """Determine the count column name (backward compatibility)."""
    if "website_count" in topic_summary.columns:
        return "website_count"
    return "count"


def identify_government_topics(topic_summary: pl.DataFrame) -> list[int]:
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
    yearly_stats = []

    for year, data in sorted(yearly_data.items()):
        topic_summary = data["topic_summary"]
        metadata = data["metadata"]

        # Filter out outliers
        valid_topics = topic_summary.filter(pl.col("topic_id") != -1)

        # Get count column
        count_col = _get_count_col(topic_summary)

        # Get top 5 topics by count
        top_topics = valid_topics.sort(count_col, descending=True).head(5)

        # Identify government topics
        gov_topics = identify_government_topics(topic_summary)

        # Government website count
        # With page-level data: a website is "government" if ANY page matches
        n_gov = 0
        gov_pct = 0

        if data["page_topics"] is not None and gov_topics:
            n_gov = (
                data["page_topics"]
                .filter(pl.col("topic").is_in(gov_topics))["website_url"]
                .n_unique()
            )
            n_total = metadata.get(
                "n_websites", data["page_topics"]["website_url"].n_unique()
            )
            gov_pct = round(n_gov / n_total * 100, 1) if n_total > 0 else 0
        elif data["website_topics"] is not None and gov_topics:
            # Fallback: use primary_topic or topic column
            wt = data["website_topics"]
            topic_col = "primary_topic" if "primary_topic" in wt.columns else "topic"
            n_gov = wt.filter(pl.col(topic_col).is_in(gov_topics)).height
            n_total = wt.height
            gov_pct = round(n_gov / n_total * 100, 1) if n_total > 0 else 0

        # Build top topics list
        top_topics_list = []
        for row in top_topics.iter_rows(named=True):
            entry = {
                "topic_id": row["topic_id"],
                "name": row["name"],
                "count": row[count_col],
                "top_words": row["top_words"],
            }
            if "page_count" in row:
                entry["page_count"] = row["page_count"]
            top_topics_list.append(entry)

        stat = {
            "year": year,
            "n_websites": metadata.get("n_websites", 0),
            "n_pages": metadata.get("n_pages", 0),
            "n_topics": metadata.get("n_topics", 0),
            "n_outlier_pages": metadata.get(
                "n_outlier_pages", metadata.get("n_outliers", 0)
            ),
            "n_outlier_websites": metadata.get("n_outlier_websites", 0),
            "page_outlier_pct": metadata.get(
                "page_outlier_pct", metadata.get("outlier_pct", 0)
            ),
            "website_outlier_pct": metadata.get("website_outlier_pct", 0),
            "n_classified_websites": metadata.get("n_classified_websites", 0),
            "avg_topics_per_website": metadata.get("avg_topics_per_website", 0),
            "top_topics": top_topics_list,
            "government_topics": gov_topics,
            "government_count": n_gov,
            "government_pct": gov_pct,
        }

        yearly_stats.append(stat)

    return yearly_stats


def find_common_topics(yearly_data: dict[int, dict]) -> list[dict]:
    # Collect all topic keywords by year
    all_topics = []

    for year, data in yearly_data.items():
        count_col = _get_count_col(data["topic_summary"])

        for row in data["topic_summary"].iter_rows(named=True):
            if row["topic_id"] == -1:
                continue

            all_topics.append(
                {
                    "year": year,
                    "topic_id": row["topic_id"],
                    "name": row["name"],
                    "top_words": set(row["top_words"].lower().split(", "))
                    if row["top_words"]
                    else set(),
                    "count": row[count_col],
                }
            )

    # Group topics by keyword overlap
    topic_clusters = []

    for topic in all_topics:
        matched = False
        for cluster in topic_clusters:
            # Check overlap with cluster representative
            overlap = len(topic["top_words"] & cluster["keywords"]) / max(
                len(topic["top_words"]), 1
            )
            if overlap > 0.3:  # 30% overlap threshold
                cluster["occurrences"].append(
                    {
                        "year": topic["year"],
                        "topic_id": topic["topic_id"],
                        "count": topic["count"],
                    }
                )
                cluster["keywords"] = cluster["keywords"] | topic["top_words"]
                matched = True
                break

        if not matched:
            topic_clusters.append(
                {
                    "name": topic["name"],
                    "keywords": topic["top_words"],
                    "occurrences": [
                        {
                            "year": topic["year"],
                            "topic_id": topic["topic_id"],
                            "count": topic["count"],
                        }
                    ],
                }
            )

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
    gov_websites = []

    for year, data in yearly_data.items():
        gov_topics = identify_government_topics(data["topic_summary"])

        if not gov_topics:
            continue

        # Use page_topics to find websites with ANY government page
        if data["page_topics"] is not None:
            year_gov = (
                data["page_topics"]
                .filter(pl.col("topic").is_in(gov_topics))
                .select(["website_url", "year"])
                .unique()
            )
            gov_websites.append(year_gov)
        elif data["website_topics"] is not None:
            wt = data["website_topics"]
            topic_col = "primary_topic" if "primary_topic" in wt.columns else "topic"
            year_gov = (
                wt.filter(pl.col(topic_col).is_in(gov_topics))
                .select(["website_url", "year"])
                .unique()
            )
            gov_websites.append(year_gov)

    if gov_websites:
        return pl.concat(gov_websites)
    else:
        return pl.DataFrame({"website_url": [], "year": []})


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print("Luxembourg Website Topic Analysis - Combine Results", flush=True)
    print("=" * 70, flush=True)

    # Load all yearly results
    yearly_data = load_yearly_results()

    if not yearly_data:
        print("\n[ERROR] No yearly results found!", flush=True)
        return

    # Compute statistics
    print("\n[ANALYZE] Computing statistics...", flush=True)
    yearly_stats = compute_topic_distribution(yearly_data)

    print("\n[ANALYZE] Finding recurring topics...", flush=True)
    recurring_topics = find_common_topics(yearly_data)

    # Summary statistics
    years = sorted(yearly_data.keys())
    total_pages = sum(
        data["metadata"].get("n_pages", 0) for data in yearly_data.values()
    )
    total_websites = sum(
        data["metadata"].get("n_websites", 0) for data in yearly_data.values()
    )

    summary = {
        "first_year": min(years),
        "last_year": max(years),
        "years_covered": len(years),
        "total_pages": total_pages,
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
    print(f"\n[SAVE] Saving stats to {FINAL_OUTPUT}...", flush=True)
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved: {FINAL_OUTPUT}", flush=True)

    # Extract and save government websites for follow-up analysis
    print("\n[EXTRACT] Extracting government websites...", flush=True)
    gov_websites = extract_government_websites(yearly_data)

    if len(gov_websites) > 0:
        gov_file = OUTPUT_DIR / "government_websites.parquet"
        gov_websites.write_parquet(gov_file, compression="zstd", compression_level=10)
        print(f"   [OK] Found {len(gov_websites):,} government website-years", flush=True)
        print(f"   [OK] Saved: {gov_file}", flush=True)
    else:
        print("   [INFO] No government websites identified", flush=True)

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\nYears: {summary['first_year']} - {summary['last_year']}", flush=True)
    print(f"Total pages: {summary['total_pages']:,}", flush=True)
    print(f"Total website-years: {summary['total_website_years']:,}", flush=True)
    print(f"Average topics per year: {summary['avg_topics_per_year']}", flush=True)

    print("\n[TOP RECURRING TOPICS]", flush=True)
    for i, topic in enumerate(recurring_topics[:10], 1):
        print(f"\n   {i}. {topic['name']}", flush=True)
        print(f"      Years present: {topic['years_present']}", flush=True)
        print(f"      Total website count: {topic['total_count']:,}", flush=True)
        print(f"      Keywords: {topic['keywords'][:80]}...", flush=True)

    print("\n[GOVERNMENT WEBSITES BY YEAR]", flush=True)
    for stat in yearly_stats:
        if stat["government_pct"] > 0:
            print(
                f"   {stat['year']}: {stat['government_count']:,} ({stat['government_pct']}%)",
                flush=True,
            )

    # Page-level stats
    has_page_data = any(
        data.get("metadata", {}).get("avg_topics_per_website", 0) > 0
        for data in yearly_data.values()
    )
    if has_page_data:
        print("\n[MULTI-TOPIC STATS BY YEAR]", flush=True)
        for stat in yearly_stats:
            if stat["avg_topics_per_website"] > 0:
                print(
                    f"   {stat['year']}: avg {stat['avg_topics_per_website']} topics/website, "
                    f"{stat['n_classified_websites']:,} classified websites",
                    flush=True,
                )

    print("\n" + "=" * 70, flush=True)
    print("DONE!", flush=True)
    print("=" * 70, flush=True)
    print("\nNext steps:", flush=True)
    print("   1. Review topics in output/*/topic_summary.csv", flush=True)
    print("   2. Use government_websites.parquet for language analysis", flush=True)
    print("   3. Build visualization with stats.json", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
