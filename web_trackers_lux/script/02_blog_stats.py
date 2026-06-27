"""
Luxembourg Web Trackers Analysis - Step 2: Blog Statistics
===========================================================
Aggregates the classified third-party data into the stats.json consumed by
the blog post. Produces four story blocks:

  1. over_time        - the "consent illusion": cookie-banner prevalence vs the
                        actual number of trackers per site, 2013-2024.
  2. sovereignty      - share of embedded third parties by owner country (US /
                        EU / LU / Other) over time, plus Google/Meta reach.
  3. top_entities     - which companies are embedded in the most .lu sites.
  4. by_sector        - tracker intensity and big-tech reach per sector, reusing
                        the BERTopic sector mapping from the topic-modelling post,
                        with the essential-vs-market cut.

Input:  data/site_summary.parquet, data/classified_requests.parquet (step 1)
        ../bert_topic_websites_lux/output/website_topics.parquet
        ../bert_topic_websites_lux/sector_mapping.json
Output: output/stats.json   (copy to the blog post directory)

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
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = DATA_DIR / "site_summary.parquet"
CLASSIFIED_FILE = DATA_DIR / "classified_requests.parquet"

TOPICS_FILE = Path(
    "/project/home/p200812/blog/bert_topic_websites_lux/output/website_topics.parquet"
)
SECTOR_MAPPING_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "bert_topic_websites_lux"
    / "sector_mapping.json"
)

STATS_FILE = OUTPUT_DIR / "stats.json"

START_YEAR = 2013
END_YEAR = 2024
COUNTRIES = ["US", "EU", "LU", "Other"]

# Same policy groupings as the language x sector post, for continuity.
ESSENTIAL_SECTORS = ["Public Services", "Healthcare", "Childcare"]
MARKET_SECTORS = ["Real Estate", "Restaurants", "Retail", "Finance & Law"]


# =============================================================================
# Loading
# =============================================================================


def load_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    summary = pl.read_parquet(SUMMARY_FILE)
    classified = pl.read_parquet(CLASSIFIED_FILE)
    print(
        f"[LOAD] site_summary: {len(summary):,}  "
        f"classified_requests: {len(classified):,}",
        flush=True,
    )
    return summary, classified


def load_sector_mapping() -> dict[int, str]:
    with open(SECTOR_MAPPING_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v["sector"] for k, v in raw.items()}


def attach_sectors(summary: pl.DataFrame) -> pl.DataFrame:
    """Join the BERTopic sector label onto each website-year (inner join)."""
    topics = pl.read_parquet(TOPICS_FILE).filter(pl.col("topic") != -1)
    mapping = load_sector_mapping()
    topics = topics.with_columns(
        pl.col("topic").replace_strict(mapping, default="Other").alias("sector")
    ).select(["website_url", "year", "sector"])

    joined = summary.join(topics, on=["website_url", "year"], how="inner")
    print(
        f"[JOIN] Sites with a sector label: {len(joined):,} "
        f"(of {len(summary):,})",
        flush=True,
    )
    return joined


# =============================================================================
# Block 1: over time (the consent illusion)
# =============================================================================


def compute_over_time(summary: pl.DataFrame) -> dict:
    years = list(range(START_YEAR, END_YEAR + 1))
    agg = (
        summary.group_by("year")
        .agg(
            pl.len().alias("n_sites"),
            pl.col("n_third_parties").median().alias("median_third_parties"),
            pl.col("n_trackers").median().alias("median_trackers"),
            pl.col("n_trackers").mean().alias("mean_trackers"),
            (pl.col("has_cmp").mean() * 100).alias("cmp_pct"),
            (pl.col("has_cookie_text").mean() * 100).alias("cookie_text_pct"),
            (pl.col("has_any_tracker").mean() * 100).alias("any_tracker_pct"),
            (pl.col("any_https").mean() * 100).alias("https_pct"),
        )
        .sort("year")
    )

    def col(name: str, ndigits: int = 1) -> list:
        lut = dict(zip(agg["year"].to_list(), agg[name].to_list()))
        return [
            round(lut[y], ndigits) if lut.get(y) is not None else None
            for y in years
        ]

    return {
        "years": years,
        "n_sites": [int(v) if v is not None else 0 for v in col("n_sites", 0)],
        "median_third_parties": col("median_third_parties"),
        "median_trackers": col("median_trackers"),
        "mean_trackers": col("mean_trackers", 2),
        "cmp_pct": col("cmp_pct"),
        "cookie_text_pct": col("cookie_text_pct"),
        "any_tracker_pct": col("any_tracker_pct"),
        "https_pct": col("https_pct"),
    }


# =============================================================================
# Block 1b: CMP blind spot (consent we cannot see)
# =============================================================================


def compute_cmp_blindspot(summary: pl.DataFrame, classified: pl.DataFrame) -> dict:
    """Quantify how much consent activity our HTML fingerprinting may miss.

    `has_cmp` only fires when a known consent-platform string is present in the
    *server-rendered* HTML. A consent banner injected at runtime through Google
    Tag Manager leaves no such fingerprint - the raw HTML shows only
    googletagmanager.com. So GTM-present-but-no-CMP sites are an upper bound on
    the consent we could be silently missing. We also surface the weak
    cookie-text signal as the noisy upper bound on "some cookie language".
    """
    years = list(range(START_YEAR, END_YEAR + 1))

    gtm = (
        classified.filter(pl.col("third_party") == "googletagmanager.com")
        .select(["website_url", "year"])
        .unique()
        .with_columns(pl.lit(True).alias("has_gtm"))
    )
    s = summary.join(gtm, on=["website_url", "year"], how="left").with_columns(
        pl.col("has_gtm").fill_null(False)
    )

    agg = (
        s.group_by("year")
        .agg(
            pl.len().alias("n_sites"),
            (pl.col("has_gtm").mean() * 100).alias("gtm_pct"),
            # GTM present but no detected CMP: consent we cannot see in the HTML.
            ((pl.col("has_gtm") & pl.col("has_cmp").not_()).mean() * 100).alias(
                "gtm_no_cmp_pct"
            ),
            # Among CMP-negative sites, the share that run GTM.
            (
                (pl.col("has_gtm") & pl.col("has_cmp").not_()).sum()
                / pl.col("has_cmp").not_().sum()
                * 100
            ).alias("gtm_share_of_no_cmp_pct"),
        )
        .sort("year")
    )

    def col(name: str) -> list:
        lut = dict(zip(agg["year"].to_list(), agg[name].to_list()))
        return [round(lut[y], 1) if lut.get(y) is not None else None for y in years]

    return {
        "years": years,
        "gtm_pct": col("gtm_pct"),
        "gtm_no_cmp_pct": col("gtm_no_cmp_pct"),
        "gtm_share_of_no_cmp_pct": col("gtm_share_of_no_cmp_pct"),
    }


# =============================================================================
# Block 2: sovereignty
# =============================================================================


def compute_sovereignty(summary: pl.DataFrame, classified: pl.DataFrame) -> dict:
    years = list(range(START_YEAR, END_YEAR + 1))

    # Domains we could not attribute to a known entity are reported as their own
    # "Unidentified" band, so the owner-country mix is not silently inflated by
    # the unknown long tail. Identified-only shares are computed separately for
    # the figure (the cleaner "of the third parties we could name, X% are US").
    cl = classified.with_columns(
        pl.when(pl.col("entity") == "Unknown")
        .then(pl.lit("Unidentified"))
        .otherwise(pl.col("country"))
        .alias("bucket")
    )
    by_bucket = cl.group_by(["year", "bucket"]).agg(pl.len().alias("n"))
    lut = {(r["year"], r["bucket"]): r["n"] for r in by_bucket.iter_rows(named=True)}
    totals = dict(cl.group_by("year").agg(pl.len().alias("t")).iter_rows())

    raw_buckets = COUNTRIES + ["Unidentified"]
    country_share = {b: [] for b in raw_buckets}
    country_share_identified = {c: [] for c in COUNTRIES}
    for y in years:
        total = totals.get(y, 0)
        for b in raw_buckets:
            n = lut.get((y, b), 0)
            country_share[b].append(round(n / total * 100, 1) if total else None)
        id_total = sum(lut.get((y, c), 0) for c in COUNTRIES)
        for c in COUNTRIES:
            n = lut.get((y, c), 0)
            country_share_identified[c].append(
                round(n / id_total * 100, 1) if id_total else None
            )

    # Site-level reach (share of sites embedding each), per year. This avoids the
    # coverage artifact in request shares: a site either loads Google or it does
    # not. "embeds Google" spans analytics, fonts, maps, reCAPTCHA, tag manager.
    reach = (
        summary.group_by("year")
        .agg(
            pl.len().alias("n_sites"),
            (pl.col("has_google").mean() * 100).alias("google_pct"),
            (pl.col("has_meta").mean() * 100).alias("meta_pct"),
            (pl.col("has_us_tracker").mean() * 100).alias("us_tracker_pct"),
        )
        .sort("year")
    )
    reach_lut = {r["year"]: r for r in reach.iter_rows(named=True)}

    def reach_series(key: str) -> list:
        return [
            round(reach_lut[y][key], 1) if y in reach_lut else None for y in years
        ]

    # Local reach: share of sites embedding at least one .lu third party.
    lu_sites = (
        cl.filter(pl.col("bucket") == "LU")
        .group_by("year")
        .agg(pl.col("website_url").n_unique().alias("n_lu_sites"))
    )
    lu_lut = {r["year"]: r["n_lu_sites"] for r in lu_sites.iter_rows(named=True)}
    lu_pct = [
        round(lu_lut.get(y, 0) / reach_lut[y]["n_sites"] * 100, 1)
        if y in reach_lut and reach_lut[y]["n_sites"]
        else None
        for y in years
    ]

    return {
        "years": years,
        "country_share": country_share,
        "country_share_identified": country_share_identified,
        "google_pct": reach_series("google_pct"),
        "meta_pct": reach_series("meta_pct"),
        "us_tracker_pct": reach_series("us_tracker_pct"),
        "lu_pct": lu_pct,
    }


# =============================================================================
# Block 3: top entities (latest year)
# =============================================================================


def compute_top_entities(
    summary: pl.DataFrame, classified: pl.DataFrame, year: int, top_n: int = 15
) -> list[dict]:
    n_sites = summary.filter(pl.col("year") == year).height
    last = classified.filter(pl.col("year") == year)

    # Aggregate to the entity level: a site counts once for a company no matter
    # how many of its domains (analytics + fonts + maps ...) it loads. tracker is
    # true if any of the entity's embedded domains is a tracker.
    per_entity = (
        last.filter(pl.col("entity") != "Unknown")
        .group_by(["website_url", "entity"])
        .agg(
            pl.col("country").first().alias("country"),
            pl.col("tracker").max().alias("tracker"),
        )
        .group_by("entity")
        .agg(
            pl.col("website_url").n_unique().alias("n_sites"),
            pl.col("country").first().alias("country"),
            pl.col("tracker").max().alias("tracker"),
        )
        .sort("n_sites", descending=True)
        .head(top_n)
    )

    return [
        {
            "entity": r["entity"],
            "country": r["country"],
            "tracker": bool(r["tracker"]),
            "pct_sites": round(r["n_sites"] / n_sites * 100, 1) if n_sites else 0,
        }
        for r in per_entity.iter_rows(named=True)
    ]


# =============================================================================
# Block 4: by sector (latest year) + essential vs market
# =============================================================================


def _group_stats(df: pl.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n_sites": 0}
    return {
        "n_sites": n,
        "median_trackers": round(df["n_trackers"].median(), 1),
        "mean_trackers": round(df["n_trackers"].mean(), 2),
        "cmp_pct": round(df["has_cmp"].mean() * 100, 1),
        "google_pct": round(df["has_google"].mean() * 100, 1),
        "meta_pct": round(df["has_meta"].mean() * 100, 1),
        "us_tracker_pct": round(df["has_us_tracker"].mean() * 100, 1),
        "any_tracker_pct": round(df["has_any_tracker"].mean() * 100, 1),
    }


def compute_by_sector(summary_sec: pl.DataFrame, year: int) -> list[dict]:
    year_df = summary_sec.filter(pl.col("year") == year)
    results = []
    for sector in sorted(year_df["sector"].unique().to_list()):
        if sector == "Other":
            continue
        sec_df = year_df.filter(pl.col("sector") == sector)
        if len(sec_df) < 5:
            continue
        results.append({"sector": sector, **_group_stats(sec_df)})
    return sorted(results, key=lambda r: -r["n_sites"])


def compute_essential_vs_market(summary_sec: pl.DataFrame, year: int) -> dict:
    year_df = summary_sec.filter(pl.col("year") == year)
    ess = year_df.filter(pl.col("sector").is_in(ESSENTIAL_SECTORS))
    mkt = year_df.filter(pl.col("sector").is_in(MARKET_SECTORS))
    return {
        "essential": {"label": "Essential Services", **_group_stats(ess)},
        "market": {"label": "Market-Driven Sectors", **_group_stats(mkt)},
        "essential_sectors": ESSENTIAL_SECTORS,
        "market_sectors": MARKET_SECTORS,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 70, flush=True)
    print("Web Trackers - Blog Statistics", flush=True)
    print("=" * 70, flush=True)

    summary, classified = load_inputs()
    summary_sec = attach_sectors(summary)

    over_time = compute_over_time(summary)
    cmp_blindspot = compute_cmp_blindspot(summary, classified)
    sovereignty = compute_sovereignty(summary, classified)
    top_entities = compute_top_entities(summary, classified, END_YEAR)
    by_sector = compute_by_sector(summary_sec, END_YEAR)
    essential_vs_market = compute_essential_vs_market(summary_sec, END_YEAR)

    last = summary.filter(pl.col("year") == END_YEAR)
    summary_block = {
        "first_year": START_YEAR,
        "last_year": END_YEAR,
        "n_sites_latest": len(last),
        "median_third_parties_latest": round(last["n_third_parties"].median(), 1),
        "median_trackers_latest": round(last["n_trackers"].median(), 1),
        "cmp_pct_latest": round(last["has_cmp"].mean() * 100, 1),
        "cookie_text_pct_latest": round(last["has_cookie_text"].mean() * 100, 1),
        "gtm_pct_latest": cmp_blindspot["gtm_pct"][-1],
        "gtm_no_cmp_pct_latest": cmp_blindspot["gtm_no_cmp_pct"][-1],
        "any_tracker_pct_latest": round(last["has_any_tracker"].mean() * 100, 1),
        "google_pct_latest": round(last["has_google"].mean() * 100, 1),
        "meta_pct_latest": round(last["has_meta"].mean() * 100, 1),
        "us_tracker_pct_latest": round(last["has_us_tracker"].mean() * 100, 1),
        "lu_pct_latest": sovereignty["lu_pct"][-1],
        "us_share_identified_latest": sovereignty["country_share_identified"]["US"][-1],
        "lu_share_identified_latest": sovereignty["country_share_identified"]["LU"][-1],
        "unidentified_share_latest": sovereignty["country_share"]["Unidentified"][-1],
        "n_sectors": len(by_sector),
    }

    stats = {
        "summary": summary_block,
        "over_time": over_time,
        "cmp_blindspot": cmp_blindspot,
        "sovereignty": sovereignty,
        "top_entities": top_entities,
        "by_sector": by_sector,
        "essential_vs_market": essential_vs_market,
    }

    print(f"\n[SAVE] {STATS_FILE}", flush=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Console readout of the headline story.
    print("\n" + "=" * 70, flush=True)
    print("THE CONSENT ILLUSION (per year)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Year':>4}  {'Sites':>6}  {'MedTrk':>6}  {'CMP%':>6}  {'US%':>6}", flush=True)
    for i, y in enumerate(over_time["years"]):
        print(
            f"  {y:>4}  {over_time['n_sites'][i]:>6}  "
            f"{str(over_time['median_trackers'][i]):>6}  "
            f"{str(over_time['cmp_pct'][i]):>6}  "
            f"{str(sovereignty['country_share']['US'][i]):>6}",
            flush=True,
        )

    print("\n" + "=" * 70, flush=True)
    print("CMP BLIND SPOT (consent possibly delivered via GTM)", flush=True)
    print("=" * 70, flush=True)
    print(
        f"  {END_YEAR}: detected CMP {summary_block['cmp_pct_latest']}% | "
        f"any cookie text {summary_block['cookie_text_pct_latest']}% (upper bound)",
        flush=True,
    )
    print(
        f"  {END_YEAR}: load GTM {summary_block['gtm_pct_latest']}% | "
        f"GTM but no detected CMP {summary_block['gtm_no_cmp_pct_latest']}% of all sites "
        f"({cmp_blindspot['gtm_share_of_no_cmp_pct'][-1]}% of CMP-negative sites)",
        flush=True,
    )

    print("\n[NEXT] Copy output/stats.json to the blog post directory:", flush=True)
    print(
        "  cp output/stats.json "
        "<website>/posts/web_trackers_lux/stats.json",
        flush=True,
    )
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
