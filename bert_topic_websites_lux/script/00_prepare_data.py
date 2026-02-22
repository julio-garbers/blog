"""
Luxembourg Website Topic Analysis - Step 0: Prepare Data
=========================================================
Saves individual web pages per year for page-level BERTopic processing.

This script:
1. Loads the sample of websites from the language analysis
2. Joins with raw data to get text content
3. Keeps individual pages (one row per page, NOT aggregated per website)
4. Saves yearly parquet files for array job processing

Uses the same sample as the language analysis for consistency.

Author: Julio Garbers with contributions from Claude
Date: February 2026
"""

import json
from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

# Input: Sample from language analysis (defines which website-years to include)
LANGUAGE_SAMPLE_FILE = Path(
    "/project/home/p200812/blog/website_languages_lux/data/lux_sample_with_languages.parquet"
)

# Input: Raw website data with text content
RAW_DATA_DIR = Path("/project/home/p201125/firm_websites/data/clean/luxembourg")

# Output: Individual pages per year
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum text length per page (characters)
MIN_TEXT_LENGTH = 100


# =============================================================================
# Data Loading
# =============================================================================


def load_language_sample() -> pl.DataFrame:
    print("\n[LOAD] Loading language analysis sample...", flush=True)
    print(f"   File: {LANGUAGE_SAMPLE_FILE}", flush=True)

    df = pl.read_parquet(LANGUAGE_SAMPLE_FILE)
    print(f"   Total website-years in sample: {len(df):,}", flush=True)

    # Get unique website-year combinations
    sample = df.select(["website_url", "year"]).unique()
    print(f"   Unique website-years: {len(sample):,}", flush=True)

    return sample


def load_raw_data() -> pl.LazyFrame:
    parquet_files = list(RAW_DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DATA_DIR}")

    print("\n[LOAD] Loading raw data...", flush=True)
    print(f"   Found {len(parquet_files)} parquet files", flush=True)

    lf = pl.concat(
        [
            pl.scan_parquet(f).select(["website_url", "year", "md_text"])
            for f in parquet_files
        ]
    )

    return lf


# =============================================================================
# Data Processing
# =============================================================================


def extract_pages(
    sample: pl.DataFrame,
    raw_lf: pl.LazyFrame,
) -> pl.DataFrame:
    print("\n[PROCESS] Extracting individual pages...", flush=True)
    print(f"   Filtering to {len(sample):,} website-years from language sample", flush=True)

    # Keep individual pages (one row per page) instead of aggregating
    df = (
        raw_lf.filter(
            # Filter to .lu domains
            pl.col("website_url").str.ends_with(".lu")
            # Filter out null/empty text
            & pl.col("md_text").is_not_null()
            & (pl.col("md_text").str.len_chars() >= MIN_TEXT_LENGTH)
        )
        # Join with sample to keep only matching website-years
        .join(sample.lazy(), on=["website_url", "year"], how="inner")
        .select(["website_url", "year", "md_text"])
        .rename({"md_text": "page_text"})
        .collect(engine="streaming")
    )

    print(f"   Total pages: {len(df):,}", flush=True)
    print(f"   Unique websites: {df['website_url'].n_unique():,}", flush=True)

    # Report coverage
    websites_with_pages = df.select(["website_url", "year"]).unique()
    coverage = len(websites_with_pages) / len(sample) * 100
    print(f"   Coverage of language sample: {coverage:.1f}%", flush=True)

    return df


def save_yearly_files(df: pl.DataFrame, output_dir: Path) -> dict[int, dict]:
    print(f"\n[SAVE] Saving yearly files to {output_dir}", flush=True)

    years = sorted(df["year"].unique().to_list())
    year_stats = {}

    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n_pages = len(year_df)
        n_websites = year_df["website_url"].n_unique()
        year_stats[year] = {"n_pages": n_pages, "n_websites": n_websites}

        output_file = output_dir / f"pages_{year}.parquet"
        year_df.write_parquet(output_file, compression="zstd", compression_level=10)

        avg_pages = n_pages / n_websites if n_websites > 0 else 0
        print(
            f"   {year}: {n_pages:,} pages from {n_websites:,} websites "
            f"(avg {avg_pages:.1f} pages/website) -> {output_file.name}",
            flush=True,
        )

    return year_stats


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary(df: pl.DataFrame, year_stats: dict[int, dict]) -> None:
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 70, flush=True)

    print(f"\nTotal pages: {len(df):,}", flush=True)
    print(f"Unique websites: {df['website_url'].n_unique():,}", flush=True)
    print(f"Years covered: {min(year_stats.keys())} - {max(year_stats.keys())}", flush=True)

    total_pages = sum(v["n_pages"] for v in year_stats.values())
    total_websites = sum(v["n_websites"] for v in year_stats.values())
    print(f"Avg pages per website-year: {total_pages / total_websites:.1f}", flush=True)

    print("\nPages and websites per year:", flush=True)
    for year, stats in sorted(year_stats.items()):
        print(f"   {year}: {stats['n_pages']:,} pages, {stats['n_websites']:,} websites", flush=True)

    print("\nPage text statistics:", flush=True)
    text_lengths = df["page_text"].str.len_chars()
    print(f"   Mean text length: {text_lengths.mean():,.0f} chars", flush=True)
    print(f"   Median text length: {text_lengths.median():,.0f} chars", flush=True)
    print(f"   Max text length: {text_lengths.max():,.0f} chars", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print("Luxembourg Website Topic Analysis - Data Preparation (Page-Level)", flush=True)
    print("=" * 70, flush=True)

    # Load sample from language analysis
    print("\n[STEP 1] Loading language analysis sample...", flush=True)
    sample = load_language_sample()

    # Load raw data
    print("\n[STEP 2] Loading raw data with text content...", flush=True)
    raw_lf = load_raw_data()

    # Extract individual pages
    print("\n[STEP 3] Extracting individual pages...", flush=True)
    df = extract_pages(sample, raw_lf)

    # Save yearly files
    print("\n[STEP 4] Saving yearly files...", flush=True)
    year_stats = save_yearly_files(df, OUTPUT_DIR)

    # Print summary
    print_summary(df, year_stats)

    # Save metadata for the array job
    metadata = {
        "years": sorted(year_stats.keys()),
        "total_pages": len(df),
        "total_website_years": df.select(["website_url", "year"]).unique().height,
        "year_stats": {str(k): v for k, v in year_stats.items()},
    }

    metadata_file = OUTPUT_DIR.parent / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Metadata saved: {metadata_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("DONE! Ready for BERTopic analysis.", flush=True)
    print("Run: sbatch script/01_bert_topic.sh", flush=True)
    print("=" * 70, flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
