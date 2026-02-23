"""
Luxembourg Website Topic Analysis - Step 0: Prepare Data
=========================================================
Aggregates web pages per website-year with paragraph-level deduplication.

This script:
1. Loads the sample of websites from the language analysis
2. Joins with raw data to get text content
3. Deduplicates repeated paragraphs within each website-year (nav, footer, etc.)
4. Saves one parquet file per year with clean aggregated text

The deduplication removes boilerplate that repeats across pages of the same
website (navigation bars, footers, cookie banners) while preserving unique
content from each page.

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

# Output: Aggregated website text per year
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum text length per page (characters) to include
MIN_PAGE_LENGTH = 100

# Minimum paragraph length (characters) to keep during dedup
MIN_PARAGRAPH_LENGTH = 20


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
# Paragraph Deduplication
# =============================================================================


def deduplicate_pages(pages: list[str]) -> str:
    """Merge multiple pages into one text, removing duplicate paragraphs.

    Pages are processed longest-first so that the most content-rich page
    contributes its paragraphs first. Subsequent pages only add paragraphs
    not already seen (based on normalized whitespace comparison).

    This removes repeated navigation, footers, cookie banners, etc.
    """
    pages_sorted = sorted(pages, key=len, reverse=True)
    seen: set[str] = set()
    unique_parts: list[str] = []

    for page in pages_sorted:
        for para in page.split("\n\n"):
            para = para.strip()
            if len(para) < MIN_PARAGRAPH_LENGTH:
                continue
            normalized = " ".join(para.split())
            if normalized not in seen:
                seen.add(normalized)
                unique_parts.append(para)

    return "\n\n".join(unique_parts) if unique_parts else pages_sorted[0]


# =============================================================================
# Data Processing
# =============================================================================


def process_websites(
    sample: pl.DataFrame,
    raw_lf: pl.LazyFrame,
) -> pl.DataFrame:
    print("\n[PROCESS] Extracting and deduplicating pages...", flush=True)
    print(f"   Filtering to {len(sample):,} website-years from language sample", flush=True)

    # Collect all pages matching the sample
    df = (
        raw_lf.filter(
            pl.col("website_url").str.ends_with(".lu")
            & pl.col("md_text").is_not_null()
            & (pl.col("md_text").str.len_chars() >= MIN_PAGE_LENGTH)
        )
        .join(sample.lazy(), on=["website_url", "year"], how="inner")
        .select(["website_url", "year", "md_text"])
        .collect(engine="streaming")
    )

    print(f"   Total pages before dedup: {len(df):,}", flush=True)
    print(f"   Unique website-years: {df.select(['website_url', 'year']).unique().height:,}", flush=True)

    # Group pages by website-year, deduplicate, and aggregate
    print("   Deduplicating paragraphs within each website-year...", flush=True)
    grouped = df.group_by(["website_url", "year"]).agg(
        pl.col("md_text").alias("pages"),
        pl.len().alias("n_pages"),
    )

    # Apply deduplication
    clean_texts = []
    for row in grouped.iter_rows(named=True):
        clean_texts.append(deduplicate_pages(row["pages"]))

    result = grouped.select(["website_url", "year", "n_pages"]).with_columns(
        pl.Series("clean_text", clean_texts)
    )

    # Text length stats
    original_total = df["md_text"].str.len_chars().sum()
    deduped_total = result["clean_text"].str.len_chars().sum()
    reduction = (1 - deduped_total / original_total) * 100 if original_total > 0 else 0

    print(f"   Text reduction from dedup: {reduction:.1f}%", flush=True)
    print(f"   Website-years with text: {len(result):,}", flush=True)

    return result


def save_yearly_files(df: pl.DataFrame, output_dir: Path) -> dict[int, dict]:
    print(f"\n[SAVE] Saving yearly files to {output_dir}", flush=True)

    years = sorted(df["year"].unique().to_list())
    year_stats = {}

    for year in years:
        year_df = df.filter(pl.col("year") == year)
        n_websites = len(year_df)
        n_pages = year_df["n_pages"].sum()
        avg_text_len = year_df["clean_text"].str.len_chars().mean()

        year_stats[year] = {
            "n_websites": n_websites,
            "n_pages": int(n_pages),
            "avg_text_length": int(avg_text_len),
        }

        output_file = output_dir / f"websites_{year}.parquet"
        year_df.write_parquet(output_file, compression="zstd", compression_level=10)

        print(
            f"   {year}: {n_websites:,} websites ({n_pages:,} pages, "
            f"avg {avg_text_len:,.0f} chars) -> {output_file.name}",
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

    print(f"\nTotal website-years: {len(df):,}", flush=True)
    print(f"Unique websites: {df['website_url'].n_unique():,}", flush=True)
    print(f"Years covered: {min(year_stats.keys())} - {max(year_stats.keys())}", flush=True)

    total_pages = sum(v["n_pages"] for v in year_stats.values())
    print(f"Total pages (before dedup): {total_pages:,}", flush=True)

    text_lengths = df["clean_text"].str.len_chars()
    print(f"\nClean text statistics (after dedup):", flush=True)
    print(f"   Mean text length: {text_lengths.mean():,.0f} chars", flush=True)
    print(f"   Median text length: {text_lengths.median():,.0f} chars", flush=True)
    print(f"   P25: {text_lengths.quantile(0.25):,.0f} chars", flush=True)
    print(f"   P75: {text_lengths.quantile(0.75):,.0f} chars", flush=True)
    print(f"   P95: {text_lengths.quantile(0.95):,.0f} chars", flush=True)
    print(f"   Max: {text_lengths.max():,.0f} chars", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print("Luxembourg Website Topic Analysis - Data Preparation", flush=True)
    print("(Paragraph-Level Deduplication)", flush=True)
    print("=" * 70, flush=True)

    # Load sample from language analysis
    print("\n[STEP 1] Loading language analysis sample...", flush=True)
    sample = load_language_sample()

    # Load raw data
    print("\n[STEP 2] Loading raw data with text content...", flush=True)
    raw_lf = load_raw_data()

    # Process: extract, deduplicate, aggregate
    print("\n[STEP 3] Processing websites with paragraph deduplication...", flush=True)
    df = process_websites(sample, raw_lf)

    # Save yearly files
    print("\n[STEP 4] Saving yearly files...", flush=True)
    year_stats = save_yearly_files(df, OUTPUT_DIR)

    # Print summary
    print_summary(df, year_stats)

    # Save metadata
    metadata = {
        "years": sorted(year_stats.keys()),
        "total_website_years": len(df),
        "total_pages_before_dedup": sum(v["n_pages"] for v in year_stats.values()),
        "year_stats": {str(k): v for k, v in year_stats.items()},
    }

    metadata_file = OUTPUT_DIR.parent / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Metadata saved: {metadata_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("DONE! Ready for embedding.", flush=True)
    print("Run: sbatch script/01_embed.sh", flush=True)
    print("=" * 70, flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
