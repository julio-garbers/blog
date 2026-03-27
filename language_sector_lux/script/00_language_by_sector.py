"""
Luxembourg Language x Sector Analysis
======================================
Joins language availability data with BERTopic sector classifications
to analyze which languages essential services are available in.

Reads:
  - website_languages_lux/data/lux_sample_with_languages.parquet
  - bert_topic_websites_lux/output/website_topics.parquet
  - bert_topic_websites_lux/sector_mapping.json
  - Raw CommonCrawl parquets (FastText fallback for undetected languages)

Writes:
  - output/stats.json (all statistics for blog visualization)

Author: Julio Garbers with contributions from Claude
Date: March 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# ===================================================================
# Paths
# ===================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOG_DIR = Path(__file__).resolve().parent.parent.parent

# Language data (with regex + LLM flags)
LANGUAGE_FILE = Path(
    "/project/home/p200812/blog/website_languages_lux/data/"
    "lux_sample_with_languages.parquet"
)

# Topic assignments (from global BERTopic)
TOPICS_FILE = Path(
    "/project/home/p200812/blog/bert_topic_websites_lux/output/"
    "website_topics.parquet"
)

# Manual topic -> sector mapping
SECTOR_MAPPING_FILE = BLOG_DIR / "bert_topic_websites_lux" / "sector_mapping.json"

# FastText fallback data (raw CommonCrawl parquets)
FASTTEXT_DATA_DIR = Path(
    "/project/home/p201125/firm_websites/data/clean/luxembourg"
)

# Output
STATS_FILE = OUTPUT_DIR / "stats.json"

# ===================================================================
# Config
# ===================================================================

LANGUAGES = ["fr", "de", "en", "lb", "pt", "nl"]
LANGUAGE_LABELS = {
    "fr": "French",
    "de": "German",
    "en": "English",
    "lb": "Luxembourgish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "other": "Other",
}

START_YEAR = 2016
END_YEAR = 2024

# Sector groupings for policy comparison
ESSENTIAL_SECTORS = ["Public Services", "Healthcare", "Childcare"]
MARKET_SECTORS = ["Real Estate", "Restaurants", "Retail", "Finance & Law"]


# ===================================================================
# Loading
# ===================================================================


def load_language_data() -> pl.DataFrame:
    """Load language data with FastText fallback.

    Replicates the fallback logic from website_languages_lux/02_final_data.py
    so that the ~6% of sites without regex/LLM detection get a language
    assignment from FastText.
    """
    print("[LOAD] Loading language extraction results...", flush=True)
    df = pl.read_parquet(LANGUAGE_FILE)
    print(f"  Language results: {len(df):,} website-years", flush=True)

    # Check if any language was detected via regex or LLM
    df = df.with_columns(
        (
            pl.col("fr").fill_null(False)
            | pl.col("de").fill_null(False)
            | pl.col("en").fill_null(False)
            | pl.col("lb").fill_null(False)
            | pl.col("pt").fill_null(False)
            | pl.col("nl").fill_null(False)
            | pl.col("other").fill_null(False)
        ).alias("has_language_info")
    )

    needs_fallback = df.filter(~pl.col("has_language_info")).height
    print(f"  Needs FastText fallback: {needs_fallback:,}", flush=True)

    # Load FastText data for fallback
    print("[LOAD] Loading FastText data for fallback...", flush=True)
    parquet_files = list(FASTTEXT_DATA_DIR.glob("*.parquet"))
    df_fasttext = (
        pl.scan_parquet(parquet_files)
        .filter(pl.col("website_url").str.ends_with(".lu"))
        .filter(pl.col("confidence_fasttext") >= 0.5)
        .group_by(["website_url", "year"])
        .agg(pl.col("language_fasttext").mode().first().alias("fasttext_lang"))
        .collect()
    )
    print(f"  FastText records: {len(df_fasttext):,}", flush=True)

    # Join and apply fallback
    df = df.join(df_fasttext, on=["website_url", "year"], how="left")

    for lang in LANGUAGES + ["other"]:
        df = df.with_columns(
            pl.when(
                ~pl.col("has_language_info")
                & (pl.col("fasttext_lang") == lang)
            )
            .then(True)
            .otherwise(pl.col(lang))
            .alias(lang)
        )

    # Handle FastText languages not in main list -> "other"
    df = df.with_columns(
        pl.when(
            ~pl.col("has_language_info")
            & pl.col("fasttext_lang").is_not_null()
            & ~pl.col("fasttext_lang").is_in(LANGUAGES)
        )
        .then(True)
        .otherwise(pl.col("other"))
        .alias("other")
    )

    # Fill remaining nulls
    for lang in LANGUAGES + ["other"]:
        df = df.with_columns(pl.col(lang).fill_null(False))

    # Count of detected languages per website
    df = df.with_columns(
        (
            pl.col("fr").cast(pl.Int8)
            + pl.col("de").cast(pl.Int8)
            + pl.col("en").cast(pl.Int8)
            + pl.col("lb").cast(pl.Int8)
            + pl.col("pt").cast(pl.Int8)
            + pl.col("nl").cast(pl.Int8)
        ).alias("n_languages")
    )

    fallback_applied = df.filter(
        ~pl.col("has_language_info") & pl.col("fasttext_lang").is_not_null()
    ).height
    print(f"  FastText fallback applied: {fallback_applied:,}", flush=True)

    return df.select(
        ["website_url", "year"] + LANGUAGES + ["other", "n_languages"]
    )


def load_topic_data() -> pl.DataFrame:
    """Load topic assignments per website-year, excluding outliers."""
    print("[LOAD] Loading topic assignments...", flush=True)
    df = pl.read_parquet(TOPICS_FILE)
    print(f"  Topic assignments: {len(df):,} website-years", flush=True)

    # Remove outliers (topic -1 = websites too unique to cluster)
    df = df.filter(pl.col("topic") != -1)
    print(f"  After removing outliers: {len(df):,}", flush=True)
    return df


def load_sector_mapping() -> dict[int, str]:
    """Load topic -> sector name mapping."""
    with open(SECTOR_MAPPING_FILE) as f:
        raw = json.load(f)
    return {int(k): v["sector"] for k, v in raw.items()}


# ===================================================================
# Analysis
# ===================================================================


def compute_sector_language_stats(
    df: pl.DataFrame, year: int
) -> list[dict]:
    """Compute language availability by sector for a given year."""
    year_df = df.filter(pl.col("year") == year)

    results = []
    for sector in sorted(year_df["sector"].unique().to_list()):
        sector_df = year_df.filter(pl.col("sector") == sector)
        n = len(sector_df)
        if n < 5:
            continue

        row = {"sector": sector, "n_websites": n}
        for lang in LANGUAGES:
            count = sector_df.filter(pl.col(lang)).height
            row[f"{lang}_pct"] = round(count / n * 100, 1)

        multi = sector_df.filter(pl.col("n_languages") >= 2).height
        row["multilingual_pct"] = round(multi / n * 100, 1)

        results.append(row)

    return sorted(results, key=lambda x: -x["n_websites"])


def compute_evolution(df: pl.DataFrame) -> dict:
    """Compute language availability evolution per sector over time."""
    years = list(range(START_YEAR, END_YEAR + 1))
    all_sectors = sorted(df["sector"].unique().to_list())

    sectors_data = []
    for sector in all_sectors:
        sector_df = df.filter(pl.col("sector") == sector)

        lang_series = {}
        for lang in LANGUAGES:
            pcts = []
            for year in years:
                year_df = sector_df.filter(pl.col("year") == year)
                n = len(year_df)
                if n < 10:
                    pcts.append(None)
                else:
                    count = year_df.filter(pl.col(lang)).height
                    pcts.append(round(count / n * 100, 1))
            lang_series[f"{lang}_pct"] = pcts

        # Multilingual share
        multi_pcts = []
        for year in years:
            year_df = sector_df.filter(pl.col("year") == year)
            n = len(year_df)
            if n < 10:
                multi_pcts.append(None)
            else:
                multi = year_df.filter(pl.col("n_languages") >= 2).height
                multi_pcts.append(round(multi / n * 100, 1))
        lang_series["multilingual_pct"] = multi_pcts

        # Website counts per year
        counts = []
        for year in years:
            counts.append(
                sector_df.filter(pl.col("year") == year).height
            )

        sectors_data.append(
            {"sector": sector, "n_websites": counts, **lang_series}
        )

    return {"years": years, "sectors": sectors_data}


def compute_essential_vs_market(df: pl.DataFrame, year: int) -> dict:
    """Compare language availability in essential vs market-driven sectors."""
    year_df = df.filter(pl.col("year") == year)

    def group_stats(sectors: list[str], label: str) -> dict:
        group_df = year_df.filter(pl.col("sector").is_in(sectors))
        n = len(group_df)
        if n == 0:
            return {"label": label, "n_websites": 0}
        result = {"label": label, "n_websites": n}
        for lang in LANGUAGES:
            count = group_df.filter(pl.col(lang)).height
            result[f"{lang}_pct"] = round(count / n * 100, 1)
        multi = group_df.filter(pl.col("n_languages") >= 2).height
        result["multilingual_pct"] = round(multi / n * 100, 1)
        return result

    return {
        "essential": group_stats(ESSENTIAL_SECTORS, "Essential Services"),
        "market": group_stats(MARKET_SECTORS, "Market-Driven Sectors"),
        "essential_sectors": ESSENTIAL_SECTORS,
        "market_sectors": MARKET_SECTORS,
    }


def compute_portuguese_gap(df: pl.DataFrame, year: int) -> list[dict]:
    """Compute Portuguese availability vs population share by sector.

    Portuguese speakers are 14.5% of Luxembourg's population but
    appear on very few websites. This computes the gap per sector.
    """
    year_df = df.filter(pl.col("year") == year)
    population_share = 14.5

    results = []
    for sector in sorted(year_df["sector"].unique().to_list()):
        sector_df = year_df.filter(pl.col("sector") == sector)
        n = len(sector_df)
        if n < 10:
            continue
        pt_count = sector_df.filter(pl.col("pt")).height
        pt_pct = round(pt_count / n * 100, 1)
        results.append({
            "sector": sector,
            "n_websites": n,
            "pt_pct": pt_pct,
            "population_pct": population_share,
            "gap": round(population_share - pt_pct, 1),
        })

    return sorted(results, key=lambda x: -x["gap"])


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    print("=" * 70, flush=True)
    print("Language x Sector Analysis - Luxembourg Websites", flush=True)
    print("=" * 70, flush=True)

    # Load all data sources
    df_lang = load_language_data()
    df_topics = load_topic_data()
    mapping = load_sector_mapping()

    # Map topic IDs to sector names
    df_topics = df_topics.with_columns(
        pl.col("topic")
        .replace_strict(mapping, default="Other")
        .alias("sector")
    )

    # Join language + topic data on (website_url, year)
    print("\n[JOIN] Merging language and topic data...", flush=True)
    df = df_topics.join(df_lang, on=["website_url", "year"], how="inner")
    print(f"  Matched: {len(df):,} website-years", flush=True)
    print(
        f"  Year range: {df['year'].min()} - {df['year'].max()}", flush=True
    )

    # Filter to analysis window
    df = df.filter(
        (pl.col("year") >= START_YEAR) & (pl.col("year") <= END_YEAR)
    )
    print(
        f"  After year filter ({START_YEAR}-{END_YEAR}): {len(df):,}",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Compute all statistics
    # ---------------------------------------------------------------

    print("\n[STATS] Computing sector x language statistics...", flush=True)

    sector_language = compute_sector_language_stats(df, END_YEAR)
    print(f"  Sectors in {END_YEAR}: {len(sector_language)}", flush=True)

    evolution = compute_evolution(df)
    print(
        f"  Evolution: {len(evolution['sectors'])} sectors x "
        f"{len(evolution['years'])} years",
        flush=True,
    )

    essential_vs_market = compute_essential_vs_market(df, END_YEAR)
    print(
        f"  Essential: {essential_vs_market['essential']['n_websites']} "
        f"websites",
        flush=True,
    )
    print(
        f"  Market: {essential_vs_market['market']['n_websites']} websites",
        flush=True,
    )

    portuguese_gap = compute_portuguese_gap(df, END_YEAR)
    print(
        f"  Portuguese gap computed for {len(portuguese_gap)} sectors",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    n_sectors = len(set(r["sector"] for r in sector_language))
    latest_df = df.filter(pl.col("year") == END_YEAR)

    summary = {
        "first_year": START_YEAR,
        "last_year": END_YEAR,
        "n_matched_total": len(df),
        "n_websites_latest": len(latest_df),
        "n_sectors": n_sectors,
    }

    # Overall language stats for the matched dataset (latest year)
    for lang in LANGUAGES:
        count = latest_df.filter(pl.col(lang)).height
        summary[f"{lang}_pct"] = round(count / len(latest_df) * 100, 1)
    multi = latest_df.filter(pl.col("n_languages") >= 2).height
    summary["multilingual_pct"] = round(multi / len(latest_df) * 100, 1)

    # ---------------------------------------------------------------
    # Build output
    # ---------------------------------------------------------------

    stats = {
        "summary": summary,
        "sector_language": sector_language,
        "essential_vs_market": essential_vs_market,
        "portuguese_gap": portuguese_gap,
        "evolution": evolution,
    }

    print(f"\n[SAVE] Writing {STATS_FILE}...", flush=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("  Done!", flush=True)

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------

    print(f"\n{'=' * 70}", flush=True)
    print(f"Language availability by sector ({END_YEAR})", flush=True)
    print(f"{'=' * 70}", flush=True)
    header = (
        f"  {'Sector':25s} {'N':>5s} {'FR':>5s} {'DE':>5s} {'EN':>5s} "
        f"{'LB':>5s} {'PT':>5s} {'Multi':>6s}"
    )
    print(header, flush=True)
    print(f"  {'-' * 72}", flush=True)
    for row in sector_language:
        print(
            f"  {row['sector']:25s} {row['n_websites']:5d} "
            f"{row['fr_pct']:5.1f} {row['de_pct']:5.1f} "
            f"{row['en_pct']:5.1f} {row['lb_pct']:5.1f} "
            f"{row['pt_pct']:5.1f} {row['multilingual_pct']:6.1f}",
            flush=True,
        )

    print(f"\n{'=' * 70}", flush=True)
    print("Essential vs Market comparison", flush=True)
    print(f"{'=' * 70}", flush=True)
    for group in ["essential", "market"]:
        g = essential_vs_market[group]
        print(
            f"  {g['label']:25s} N={g['n_websites']:4d}  "
            f"FR={g.get('fr_pct', 0):5.1f}  EN={g.get('en_pct', 0):5.1f}  "
            f"PT={g.get('pt_pct', 0):5.1f}  Multi={g.get('multilingual_pct', 0):5.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
