"""
Luxembourg Website Topic Analysis - Step 0: Prepare Data
=========================================================
Aggregates website text content by website-year and saves one parquet file
per year for parallel BERTopic processing.

This script:
1. Loads the sample of websites from the language analysis
2. Joins with raw data to get text content
3. Aggregates all page text for each website within each year
4. Saves yearly parquet files for array job processing

Uses the same sample as the language analysis for consistency.

Author: Julio Garbers with contributions from Claude
Date: January 2026
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

# Output: Aggregated data per year
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum text length to include (characters)
MIN_TEXT_LENGTH = 100

# Minimum pages per website to include
MIN_PAGES_PER_WEBSITE = 1


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

    # Read each file selecting only needed columns, then concatenate
    # This handles schema differences across files (some have extra columns)
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


def aggregate_by_website_year(
    sample: pl.DataFrame,
    raw_lf: pl.LazyFrame,
) -> pl.DataFrame:
    print("\n[PROCESS] Aggregating text by website-year...", flush=True)
    print(f"   Filtering to {len(sample):,} website-years from language sample", flush=True)

    # Filter raw data to only include websites in our sample
    # and aggregate text by website-year
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
        # Aggregate by website-year
        .group_by(["website_url", "year"])
        .agg(
            [
                # Concatenate all page texts
                pl.col("md_text")
                .str.join(delimiter="\n\n---PAGE BREAK---\n\n")
                .alias("aggregated_text"),
                # Count pages
                pl.len().alias("n_pages"),
                # Sum text lengths
                pl.col("md_text").str.len_chars().sum().alias("total_text_length"),
            ]
        )
        .filter(pl.col("n_pages") >= MIN_PAGES_PER_WEBSITE)
        .collect(engine="streaming")
    )

    print(f"   Total website-years with text: {len(df):,}", flush=True)
    print(f"   Unique websites: {df['website_url'].n_unique():,}", flush=True)

    # Report coverage
    coverage = len(df) / len(sample) * 100
    print(f"   Coverage of language sample: {coverage:.1f}%", flush=True)

    return df


def save_yearly_files(df: pl.DataFrame, output_dir: Path) -> dict[int, int]:
    print(f"\n[SAVE] Saving yearly files to {output_dir}", flush=True)

    years = sorted(df["year"].unique().to_list())
    year_counts = {}

    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n_websites = len(year_df)
        year_counts[year] = n_websites

        output_file = output_dir / f"websites_{year}.parquet"
        year_df.write_parquet(output_file, compression="zstd", compression_level=10)

        print(f"   {year}: {n_websites:,} websites -> {output_file.name}", flush=True)

    return year_counts


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary(df: pl.DataFrame, year_counts: dict[int, int]) -> None:
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 70, flush=True)

    print(f"\nTotal website-years: {len(df):,}", flush=True)
    print(f"Unique websites: {df['website_url'].n_unique():,}", flush=True)
    print(f"Years covered: {min(year_counts.keys())} - {max(year_counts.keys())}", flush=True)

    print("\nWebsites per year:", flush=True)
    for year, count in sorted(year_counts.items()):
        print(f"   {year}: {count:,}", flush=True)

    print("\nText statistics:", flush=True)
    print(f"   Mean pages per website: {df['n_pages'].mean():.1f}", flush=True)
    print(f"   Median pages per website: {df['n_pages'].median():.0f}", flush=True)
    print(f"   Mean text length: {df['total_text_length'].mean():,.0f} chars", flush=True)
    print(f"   Median text length: {df['total_text_length'].median():,.0f} chars", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print("Luxembourg Website Topic Analysis - Data Preparation", flush=True)
    print("=" * 70, flush=True)

    # Load sample from language analysis
    print("\n[STEP 1] Loading language analysis sample...", flush=True)
    sample = load_language_sample()

    # Load raw data
    print("\n[STEP 2] Loading raw data with text content...", flush=True)
    raw_lf = load_raw_data()

    # Aggregate by website-year
    print("\n[STEP 3] Aggregating text by website-year...", flush=True)
    df = aggregate_by_website_year(sample, raw_lf)

    # Save yearly files
    print("\n[STEP 4] Saving yearly files...", flush=True)
    year_counts = save_yearly_files(df, OUTPUT_DIR)

    # Print summary
    print_summary(df, year_counts)

    # Save metadata for the array job
    metadata = {
        "years": sorted(year_counts.keys()),
        "total_website_years": len(df),
        "year_counts": year_counts,
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
