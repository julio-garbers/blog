"""
Luxembourg Web Trackers Analysis - Step 1: Classify Third Parties
==================================================================
Takes the per-website-year third-party domain sets from step 0, looks each
domain up in the curated entity map (owner, country, category, tracker flag),
and produces a long table of (website_url, year, third_party) classifications
plus a per-website-year summary with tracker counts and owner-country mix.

Classification rules:
  - A third party whose registrable domain ends in ".lu" is local/sovereign
    (country = "LU", entity = "Luxembourg (local)").
  - A domain present in reference/entity_map.json uses its mapped values.
  - Anything else is country = "Other", category = "other", tracker = False
    (we never over-count tracking we cannot positively identify).

Input:  data/yearly/thirdparty_{YEAR}.parquet (from step 0)
        reference/entity_map.json
Output: data/classified_requests.parquet   (long: one row per site x third party)
        data/site_summary.parquet           (one row per website-year)

Author: Julio Garbers with contributions from Claude
Date: June 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path("/project/home/p200812/blog/web_trackers_lux")
YEARLY_DIR = BASE_DIR / "data" / "yearly"
OUTPUT_DIR = BASE_DIR / "data"

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
ENTITY_MAP_FILE = REFERENCE_DIR / "entity_map.json"

CLASSIFIED_FILE = OUTPUT_DIR / "classified_requests.parquet"
SUMMARY_FILE = OUTPUT_DIR / "site_summary.parquet"

TRACKER_CATEGORIES = {"advertising", "analytics", "social", "tag-manager", "marketing"}


# =============================================================================
# Entity map
# =============================================================================


def load_entity_map() -> pl.DataFrame:
    """Load the curated domain -> entity/country/category map as a frame."""
    with open(ENTITY_MAP_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for domain, info in raw.items():
        if domain.startswith("_"):  # skip _meta
            continue
        rows.append(
            {
                "third_party": domain.lower(),
                "entity": info["entity"],
                "country": info["country"],
                "category": info["category"],
                "tracker": bool(info["tracker"]),
            }
        )

    df = pl.DataFrame(rows)
    print(f"[LOAD] Entity map: {len(df):,} domains", flush=True)
    return df


# =============================================================================
# Load + explode third-party sets
# =============================================================================


def load_third_parties() -> pl.DataFrame:
    files = sorted(YEARLY_DIR.glob("thirdparty_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No yearly files in {YEARLY_DIR}")
    print(f"[LOAD] {len(files)} yearly third-party files", flush=True)

    df = pl.concat([pl.read_parquet(f) for f in files])
    print(f"[LOAD] Total website-years: {len(df):,}", flush=True)
    return df


def classify(df_sites: pl.DataFrame, entity_map: pl.DataFrame) -> pl.DataFrame:
    """Explode to (website_url, year, third_party) and attach entity metadata."""
    long = (
        df_sites.select(["website_url", "year", "third_parties"])
        .explode("third_parties")
        .rename({"third_parties": "third_party"})
        .filter(pl.col("third_party").is_not_null())
        .with_columns(pl.col("third_party").str.to_lowercase())
    )
    print(f"[CLASSIFY] Third-party requests (long): {len(long):,}", flush=True)

    long = long.join(entity_map, on="third_party", how="left")

    # Local .lu third parties: sovereign, even if not in the curated map.
    is_local = pl.col("third_party").str.ends_with(".lu")

    long = long.with_columns(
        pl.when(is_local & pl.col("entity").is_null())
        .then(pl.lit("Luxembourg (local)"))
        .otherwise(pl.col("entity"))
        .alias("entity"),
        pl.when(is_local & pl.col("country").is_null())
        .then(pl.lit("LU"))
        .otherwise(pl.col("country"))
        .alias("country"),
        pl.col("category").fill_null("other"),
        pl.col("tracker").fill_null(False),
    ).with_columns(
        # Anything still unknown -> Other country.
        pl.col("country").fill_null("Other"),
        pl.col("entity").fill_null("Unknown"),
    )

    # Coverage diagnostic: share of requests we could positively name.
    known = long.filter(pl.col("entity") != "Unknown").height
    print(
        f"[CLASSIFY] Identified entity for {known:,}/{len(long):,} requests "
        f"({known / len(long) * 100:.1f}%)",
        flush=True,
    )
    return long


# =============================================================================
# Per-site summary
# =============================================================================


def build_summary(
    df_sites: pl.DataFrame, classified: pl.DataFrame
) -> pl.DataFrame:
    """One row per website-year: tracker counts, owner mix, big-tech flags."""
    per_site = classified.group_by(["website_url", "year"]).agg(
        pl.col("tracker").sum().alias("n_trackers"),
        pl.col("third_party").n_unique().alias("n_third_parties_classified"),
        (pl.col("country") == "US").sum().alias("n_us"),
        (pl.col("country") == "EU").sum().alias("n_eu"),
        (pl.col("country") == "LU").sum().alias("n_lu"),
        (pl.col("country") == "Other").sum().alias("n_other"),
        (pl.col("entity") == "Google").any().alias("has_google"),
        (pl.col("entity") == "Meta").any().alias("has_meta"),
        ((pl.col("country") == "US") & pl.col("tracker")).any().alias("has_us_tracker"),
        (pl.col("tracker").sum() > 0).alias("has_any_tracker"),
    )

    summary = df_sites.join(per_site, on=["website_url", "year"], how="left")

    # Sites with zero third parties get null aggregates -> fill with 0/False.
    summary = summary.with_columns(
        pl.col("n_trackers").fill_null(0),
        pl.col("n_third_parties_classified").fill_null(0),
        pl.col("n_us").fill_null(0),
        pl.col("n_eu").fill_null(0),
        pl.col("n_lu").fill_null(0),
        pl.col("n_other").fill_null(0),
        pl.col("has_google").fill_null(False),
        pl.col("has_meta").fill_null(False),
        pl.col("has_us_tracker").fill_null(False),
        pl.col("has_any_tracker").fill_null(False),
    )
    return summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 70, flush=True)
    print("Web Trackers - Classify Third Parties", flush=True)
    print("=" * 70, flush=True)

    entity_map = load_entity_map()
    df_sites = load_third_parties()

    classified = classify(df_sites, entity_map)
    summary = build_summary(df_sites, classified)

    print(f"\n[SAVE] {CLASSIFIED_FILE}", flush=True)
    classified.write_parquet(
        CLASSIFIED_FILE, compression="zstd", compression_level=10
    )
    print(f"[SAVE] {SUMMARY_FILE}", flush=True)
    summary.write_parquet(SUMMARY_FILE, compression="zstd", compression_level=10)

    # Sanity readout for the latest year.
    latest = summary["year"].max()
    last = summary.filter(pl.col("year") == latest)
    print(f"\n[SUMMARY] {latest}: {len(last):,} sites", flush=True)
    print(f"  Median trackers/site: {last['n_trackers'].median():.1f}", flush=True)
    print(
        f"  Sites embedding Google: {last['has_google'].mean() * 100:.1f}%",
        flush=True,
    )
    print(
        f"  Sites embedding Meta:   {last['has_meta'].mean() * 100:.1f}%",
        flush=True,
    )
    print(
        f"  Sites with a US tracker: {last['has_us_tracker'].mean() * 100:.1f}%",
        flush=True,
    )
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
