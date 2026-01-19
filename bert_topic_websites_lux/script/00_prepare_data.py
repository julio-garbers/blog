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
    """Load the website-year sample from language analysis."""
    print("\n[LOAD] Loading language analysis sample...")
    print(f"   File: {LANGUAGE_SAMPLE_FILE}")

    df = pl.read_parquet(LANGUAGE_SAMPLE_FILE)
    print(f"   Total website-years in sample: {len(df):,}")

    # Get unique website-year combinations
    sample = df.select(["website_url", "year"]).unique()
    print(f"   Unique website-years: {len(sample):,}")

    return sample


def load_raw_data() -> pl.LazyFrame:
    """Load raw website data with text content."""
    parquet_files = list(RAW_DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DATA_DIR}")

    print("\n[LOAD] Loading raw data...")
    print(f"   Found {len(parquet_files)} parquet files")

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
    """
    Aggregate text content by website and year.

    For each (website_url, year) combination in the sample:
    - Join with raw data to get text content
    - Concatenate all page texts with newlines
    - Count number of pages
    - Calculate total text length
    """
    print("\n[PROCESS] Aggregating text by website-year...")
    print(f"   Filtering to {len(sample):,} website-years from language sample")

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

    print(f"   Total website-years with text: {len(df):,}")
    print(f"   Unique websites: {df['website_url'].n_unique():,}")

    # Report coverage
    coverage = len(df) / len(sample) * 100
    print(f"   Coverage of language sample: {coverage:.1f}%")

    return df


def save_yearly_files(df: pl.DataFrame, output_dir: Path) -> dict[int, int]:
    """Save one parquet file per year and return counts."""
    print(f"\n[SAVE] Saving yearly files to {output_dir}")

    years = sorted(df["year"].unique().to_list())
    year_counts = {}

    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n_websites = len(year_df)
        year_counts[year] = n_websites

        output_file = output_dir / f"websites_{year}.parquet"
        year_df.write_parquet(output_file, compression="zstd", compression_level=10)

        print(f"   {year}: {n_websites:,} websites -> {output_file.name}")

    return year_counts


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary(df: pl.DataFrame, year_counts: dict[int, int]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal website-years: {len(df):,}")
    print(f"Unique websites: {df['website_url'].n_unique():,}")
    print(f"Years covered: {min(year_counts.keys())} - {max(year_counts.keys())}")

    print("\nWebsites per year:")
    for year, count in sorted(year_counts.items()):
        print(f"   {year}: {count:,}")

    print("\nText statistics:")
    print(f"   Mean pages per website: {df['n_pages'].mean():.1f}")
    print(f"   Median pages per website: {df['n_pages'].median():.0f}")
    print(f"   Mean text length: {df['total_text_length'].mean():,.0f} chars")
    print(f"   Median text length: {df['total_text_length'].median():,.0f} chars")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Luxembourg Website Topic Analysis - Data Preparation")
    print("=" * 70)

    # Load sample from language analysis
    print("\n[STEP 1] Loading language analysis sample...")
    sample = load_language_sample()

    # Load raw data
    print("\n[STEP 2] Loading raw data with text content...")
    raw_lf = load_raw_data()

    # Aggregate by website-year
    print("\n[STEP 3] Aggregating text by website-year...")
    df = aggregate_by_website_year(sample, raw_lf)

    # Save yearly files
    print("\n[STEP 4] Saving yearly files...")
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
    print(f"\n[OK] Metadata saved: {metadata_file}")

    print("\n" + "=" * 70)
    print("DONE! Ready for BERTopic analysis.")
    print("Run: sbatch script/01_bert_topic.sh")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()