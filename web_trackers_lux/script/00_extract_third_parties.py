"""
Luxembourg Web Trackers Analysis - Step 0: Extract Third-Party Requests
========================================================================
Parses the raw HTML of Luxembourg websites and, for each website-year,
records the set of *third-party* domains the site embeds, whether it loads
a consent-management platform (cookie banner), and whether it is served
over HTTPS.

This is the raw-infrastructure counterpart to the language and topic posts:
instead of asking what a website *says*, we ask how it is *built* and who it
silently connects you to.

Run as a SLURM array job, one task per year (2013-2024 -> array IDs 0-11),
mirroring the embedding step of the topic-modelling pipeline.

For one website-year, a "third party" is any resource host whose registrable
domain (eTLD+1) differs from the site's own registrable domain. We scan the
src/href attributes of <script>, <img>, <iframe>, <link>, <source> and any
absolute/protocol-relative URLs in inline scripts.

Input:  Raw HTML gz parquets (url, year, html)
        Same website-year universe as the language analysis (inner join).
Output: One parquet per year with per-website-year third-party sets + flags.

Author: Julio Garbers with contributions from Claude
Date: June 2026
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import tldextract

# =============================================================================
# Configuration
# =============================================================================

# Raw HTML data (one subdir per year, gz parquet files with url/year/html)
RAW_DATA_DIR = Path("/project/home/p201125/firm_websites/data/raw/luxembourg")

# Universe of website-years to keep (same as language + topic posts)
LANGUAGE_SAMPLE_FILE = Path(
    "/project/home/p200812/blog/website_languages_lux/data/"
    "lux_sample_with_languages.parquet"
)

# Consent-management-platform fingerprints (shipped in the repo)
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
CMP_SIGNATURES_FILE = REFERENCE_DIR / "cmp_signatures.json"

# Output: per-year third-party extraction
OUTPUT_DIR = Path("/project/home/p200812/blog/web_trackers_lux/data/yearly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2013, 2025))  # 2013-2024

# Cap third-party domains kept per website-year (defensive against runaway pages)
MAX_THIRD_PARTIES = 200

# tldextract without network access on compute nodes: rely on the bundled
# public-suffix snapshot rather than fetching a fresh list at runtime.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


# =============================================================================
# URL / domain helpers
# =============================================================================

# Capture src="..." / href="..." attribute values, plus absolute and
# protocol-relative URLs that appear inside inline scripts.
_ATTR_URL_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"""(?:https?:)?//[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}""")

# Hosts/schemes that never represent a real third party
_SKIP_PREFIXES = ("data:", "javascript:", "mailto:", "tel:", "#", "blob:", "about:")


def registrable_domain(host: str) -> str | None:
    """Return the eTLD+1 (e.g. 'google.com') for a host, or None if not a domain."""
    if not host:
        return None
    ext = _EXTRACT(host)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}".lower()


def host_from_url(url: str, page_scheme: str) -> str | None:
    """Extract the host from a (possibly protocol-relative or relative) URL."""
    url = url.strip()
    if not url:
        return None
    low = url.lower()
    if low.startswith(_SKIP_PREFIXES):
        return None
    if url.startswith("//"):
        url = f"{page_scheme}:{url}"
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        return None
    if not netloc:
        return None  # relative URL -> first party, ignore
    return netloc.split("@")[-1].split(":")[0].lower()


def extract_third_parties(html: str, page_url: str) -> tuple[list[str], bool]:
    """Extract third-party registrable domains and an HTTPS flag from one page.

    Returns (list_of_third_party_domains, is_https). Third party means a
    registrable domain different from the page's own registrable domain.
    """
    if not html:
        return [], page_url.lower().startswith("https")

    parsed = urlparse(page_url)
    page_scheme = parsed.scheme or "http"
    first_party = registrable_domain(parsed.netloc.split(":")[0])

    candidates = _ATTR_URL_RE.findall(html)
    candidates.extend(_BARE_URL_RE.findall(html))

    found: set[str] = set()
    for raw in candidates:
        host = host_from_url(raw, page_scheme)
        if host is None:
            continue
        reg = registrable_domain(host)
        if reg is None or reg == first_party:
            continue
        found.add(reg)
        if len(found) >= MAX_THIRD_PARTIES:
            break

    return sorted(found), page_scheme == "https"


# =============================================================================
# CMP (cookie-banner) detection
# =============================================================================


def load_cmp_signatures() -> tuple[list[str], list[str]]:
    with open(CMP_SIGNATURES_FILE, encoding="utf-8") as f:
        sigs = json.load(f)
    return (
        [s.lower() for s in sigs["strong"]],
        [s.lower() for s in sigs["weak"]],
    )


def detect_cmp(html_lower: str, strong: list[str], weak: list[str]) -> tuple[bool, bool]:
    """Return (has_cmp_strong, has_cookie_text)."""
    has_strong = any(sig in html_lower for sig in strong)
    has_weak = any(sig in html_lower for sig in weak)
    return has_strong, has_weak


# =============================================================================
# Per-year processing
# =============================================================================


def resolve_year() -> int:
    """Pick the year from the SLURM array task id (falls back to env/arg)."""
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        return YEARS[int(task_id)]
    # Fallback for manual runs: WEB_TRACKERS_YEAR env var
    return int(os.getenv("WEB_TRACKERS_YEAR", str(YEARS[0])))


def load_sample_universe(year: int) -> pl.DataFrame:
    """Website-years to keep for this year (same universe as the other posts)."""
    df = (
        pl.scan_parquet(LANGUAGE_SAMPLE_FILE)
        .select(["website_url", "year"])
        .filter(pl.col("year") == year)
        .unique()
        .collect()
    )
    return df


def process_year(year: int, strong: list[str], weak: list[str]) -> pl.DataFrame:
    raw_dir = RAW_DATA_DIR / str(year) / "gz"
    gz_files = sorted(raw_dir.glob("*.gz"))
    print(f"[{year}] Found {len(gz_files):,} raw gz files", flush=True)
    if not gz_files:
        raise FileNotFoundError(f"No raw files for {year} in {raw_dir}")

    # Per-website-year accumulator: domain set + flags + page count.
    acc: dict[str, dict] = {}
    total_pages = 0

    for i, gz_file in enumerate(gz_files, start=1):
        page_df = (
            pl.scan_parquet(gz_file)
            .select(["url", "html"])
            .filter(pl.col("html").is_not_null())
            .collect()
        )

        for row in page_df.iter_rows(named=True):
            page_url = row["url"]
            html = row["html"]
            total_pages += 1

            # First-party domain = the .lu site this page belongs to.
            host = urlparse(page_url).netloc.split(":")[0]
            site = registrable_domain(host)
            if site is None or not site.endswith(".lu"):
                continue

            third, is_https = extract_third_parties(html, page_url)
            html_lower = html.lower()
            has_strong, has_weak = detect_cmp(html_lower, strong, weak)

            rec = acc.get(site)
            if rec is None:
                rec = {
                    "domains": set(),
                    "has_cmp": False,
                    "has_cookie_text": False,
                    "any_https": False,
                    "n_pages": 0,
                }
                acc[site] = rec
            rec["domains"].update(third)
            rec["has_cmp"] = rec["has_cmp"] or has_strong
            rec["has_cookie_text"] = rec["has_cookie_text"] or has_weak
            rec["any_https"] = rec["any_https"] or is_https
            rec["n_pages"] += 1

        if i % 25 == 0 or i == len(gz_files):
            print(
                f"[{year}]   processed {i}/{len(gz_files)} files, "
                f"{len(acc):,} sites so far",
                flush=True,
            )

    print(f"[{year}] Total pages scanned: {total_pages:,}", flush=True)
    print(f"[{year}] Unique .lu sites: {len(acc):,}", flush=True)

    # Build the result frame.
    rows = []
    for site, rec in acc.items():
        domains = sorted(rec["domains"])[:MAX_THIRD_PARTIES]
        rows.append(
            {
                "website_url": site,
                "year": year,
                "third_parties": domains,
                "n_third_parties": len(domains),
                "has_cmp": rec["has_cmp"],
                "has_cookie_text": rec["has_cookie_text"],
                "any_https": rec["any_https"],
                "n_pages": rec["n_pages"],
            }
        )

    result = pl.DataFrame(
        rows,
        schema={
            "website_url": pl.String,
            "year": pl.Int64,
            "third_parties": pl.List(pl.String),
            "n_third_parties": pl.Int64,
            "has_cmp": pl.Boolean,
            "has_cookie_text": pl.Boolean,
            "any_https": pl.Boolean,
            "n_pages": pl.Int64,
        },
    )

    # Restrict to the shared website-year universe for cross-post comparability.
    universe = load_sample_universe(year)
    before = len(result)
    result = result.join(universe, on=["website_url", "year"], how="inner")
    print(
        f"[{year}] Sites after universe join: {len(result):,} "
        f"(dropped {before - len(result):,} not in language sample)",
        flush=True,
    )

    return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    year = resolve_year()
    print("=" * 70, flush=True)
    print(f"Web Trackers - Third-Party Extraction for {year}", flush=True)
    print("=" * 70, flush=True)

    strong, weak = load_cmp_signatures()
    print(
        f"[CONFIG] {len(strong)} strong + {len(weak)} weak CMP signatures loaded",
        flush=True,
    )

    result = process_year(year, strong, weak)

    out_file = OUTPUT_DIR / f"thirdparty_{year}.parquet"
    result.write_parquet(out_file, compression="zstd", compression_level=10)
    print(f"\n[SAVE] Wrote {len(result):,} website-years -> {out_file}", flush=True)

    # Quick sanity summary.
    if len(result) > 0:
        print("\n[SUMMARY]", flush=True)
        print(
            f"  Median third parties / site: "
            f"{result['n_third_parties'].median():.1f}",
            flush=True,
        )
        print(
            f"  Mean third parties / site:   "
            f"{result['n_third_parties'].mean():.2f}",
            flush=True,
        )
        print(
            f"  Sites with a CMP banner:     "
            f"{result['has_cmp'].sum():,} "
            f"({result['has_cmp'].mean() * 100:.1f}%)",
            flush=True,
        )
        print(
            f"  Sites served over HTTPS:     "
            f"{result['any_https'].sum():,} "
            f"({result['any_https'].mean() * 100:.1f}%)",
            flush=True,
        )

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
