"""
Luxembourg Language Analysis - Step 2: Extract Available Languages via LLM
===========================================================================
Joins the sample with raw HTML data, extracts <head> and navigation,
then uses Magistral to detect which languages each website offers.

Input:  Sample parquet + raw HTML gz files
Output: Parquet with detected languages per page

Author: Julio Garbers with contributions from Claude
Date: January 2026
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# Configuration
# =============================================================================

# Silence noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="aiohttp")

# Parse command-line arguments
cli = argparse.ArgumentParser(
    description="Extract available languages from Luxembourg websites"
)
cli.add_argument("--host")
cli.add_argument("--model")
cli.add_argument("--tensor-parallel-size")
cli.add_argument("--pipeline-parallel-size")
args, _ = cli.parse_known_args()

# Directories
SAMPLE_FILE = Path("/project/home/p200812/blog/data/lux_sample_for_llm.parquet")
RAW_DATA_DIR = Path("/project/home/p201125/firm_websites/data/raw/luxembourg")
OUTPUT_DIR = Path("/project/home/p200812/blog/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lux_sample_with_languages.parquet"

# Years to process
YEARS = list(range(2013, 2025))

# Model and API configuration
API_URL = (
    os.getenv("VLLM_SERVER_URL", args.host or "http://localhost:8000").rstrip("/")
    + "/v1/chat/completions"
)
HF_MODEL = os.getenv("HF_MODEL", args.model or "mistralai/Magistral-Small-2506")

PIPELINE_STAGES = int(
    os.getenv("PIPELINE_PARALLEL_SIZE", args.pipeline_parallel_size or "1")
)
CONCURRENCY = PIPELINE_STAGES * 32
TIMEOUT_S = int(os.getenv("TIMEOUT_S", "300"))
MAX_RETRIES = 3


# =============================================================================
# Prompt Templates
# =============================================================================

SYS_PROMPT = """\
You are a web analyst. Given the HTML <head> section and navigation area of a Luxembourg website, \
identify which languages the website is available in.

Look for these signals (this is not an exhaustive list, there may be other ways languages are indicated):
1. <link rel="alternate" hreflang="..."> tags (most reliable)
2. Language switcher links: text like "FR", "DE", "EN", "LU", "Français", "Deutsch", "English", "Lëtzebuergesch"
3. URL patterns in links: /fr/, /de/, /en/, /lu/, ?lang=fr, etc.
4. <meta> language tags
5. Navigation menus with language options
6. Dropdown selectors or flags indicating language choices

Output a JSON object with boolean values for each language. Only mark true if you have clear evidence.

Languages to detect:
- fr: French
- de: German  
- en: English
- lb: Luxembourgish (also indicated by "lu", "LU", "Lëtzebuergesch", "Luxembourgish")
- pt: Portuguese
- nl: Dutch
- other: Any other language not listed above (e.g., Spanish, Italian, Chinese, etc.)

Output format (*exactly like this*):
{
  "fr": true/false,
  "de": true/false,
  "en": true/false,
  "lb": true/false,
  "pt": true/false,
  "nl": true/false,
  "other": true/false
}

*Do not provide any additional explanation.*\
"""

USER_PROMPT = "HTML content:\n{html}\n\nIdentify the available languages."


# =============================================================================
# HTML Extraction
# =============================================================================


def extract_head_and_nav(
    html: str, head_limit: int = 15000, body_limit: int = 8000
) -> str:
    """
    Extract the <head> section and first part of <body> from HTML.
    These contain hreflang tags and navigation with language switchers.
    """
    if not html or not isinstance(html, str):
        return ""

    result_parts = []

    # Extract <head>...</head>
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.IGNORECASE | re.DOTALL)
    if head_match:
        head_content = head_match.group(0)
        # Limit head size
        if len(head_content) > head_limit:
            head_content = head_content[:head_limit] + "...</head>"
        result_parts.append(head_content)

    # Extract first part of <body> (contains navigation)
    body_match = re.search(r"<body[^>]*>(.*)", html, re.IGNORECASE | re.DOTALL)
    if body_match:
        body_content = body_match.group(0)
        # Take first N characters of body
        if len(body_content) > body_limit:
            body_content = body_content[:body_limit] + "..."
        result_parts.append(body_content)

    # If no head/body found, just take the beginning of HTML
    if not result_parts:
        result_parts.append(html[: head_limit + body_limit])

    return "\n".join(result_parts)


# =============================================================================
# Type Definitions
# =============================================================================


class Parsed(dict):
    """Typed dictionary for parsed extraction results."""

    fr: bool | None
    de: bool | None
    en: bool | None
    lb: bool | None
    pt: bool | None
    nl: bool | None
    other: bool | None


@dataclass
class ValidationStats:
    """Track statistics for validation and cleaning operations."""

    json_parse_success: int = 0
    json_parse_failed: int = 0

    # Language detection counts
    detected_fr: int = 0
    detected_de: int = 0
    detected_en: int = 0
    detected_lb: int = 0
    detected_pt: int = 0
    detected_nl: int = 0
    detected_other: int = 0

    def print_summary(self, total: int) -> None:
        """Print validation statistics."""
        print(
            "===============================================================================",
            flush=True,
        )
        print("[VALIDATION STATS] Summary:", flush=True)
        print("  JSON parsing:", flush=True)
        print(f"    Successful: {self.json_parse_success:,}", flush=True)
        print(f"    Failed:     {self.json_parse_failed:,}", flush=True)

        if total > 0:
            print("\n  Languages detected (pages offering each language):", flush=True)
            print(
                f"    French (fr):        {self.detected_fr:,} ({self.detected_fr / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    German (de):        {self.detected_de:,} ({self.detected_de / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    English (en):       {self.detected_en:,} ({self.detected_en / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Luxembourgish (lb): {self.detected_lb:,} ({self.detected_lb / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Portuguese (pt):    {self.detected_pt:,} ({self.detected_pt / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Dutch (nl):         {self.detected_nl:,} ({self.detected_nl / total * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Other:              {self.detected_other:,} ({self.detected_other / total * 100:.1f}%)",
                flush=True,
            )
        print(
            "===============================================================================",
            flush=True,
        )


# Global stats tracker
validation_stats = ValidationStats()


# =============================================================================
# API Request Functions
# =============================================================================


def build_payload(prompt: str) -> dict[str, Any]:
    """Build JSON payload for vLLM server."""
    return {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
        "seed": 666,
    }


async def post_request(sess: aiohttp.ClientSession, prompt: str) -> str:
    """Send POST request to vLLM server and return response content."""
    async with sess.post(API_URL, json=build_payload(prompt)) as response:
        response.raise_for_status()
        data = await response.json()
        return data["choices"][0]["message"]["content"].strip()


async def retry_post(
    sess: aiohttp.ClientSession, prompt: str, sem: asyncio.Semaphore
) -> str | None:
    """Retry POST request with exponential backoff on failure."""
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            async with sem:
                return await post_request(sess, prompt)
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            if attempt == MAX_RETRIES:
                print(
                    f"[ERROR] Request failed after {MAX_RETRIES} retries: {err}",
                    flush=True,
                )
                return None
            await asyncio.sleep((2**attempt) + random.random())
            attempt += 1
    return None


async def run_inference(prompts: list[str]) -> list[str | None]:
    """Run batch inference on all prompts with concurrency control."""
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [retry_post(session, prompt, semaphore) for prompt in prompts]
        return await tqdm_asyncio.gather(*tasks, desc="inference")


# =============================================================================
# Response Parsing
# =============================================================================


def safe_parse_json(text: str | None) -> Parsed:
    """Parse JSON response from LLM."""
    global validation_stats

    empty: Parsed = {
        "fr": None,
        "de": None,
        "en": None,
        "lb": None,
        "pt": None,
        "nl": None,
        "other": None,
    }

    if text is None:
        validation_stats.json_parse_failed += 1
        return empty

    text = text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        data: Parsed = json.loads(text)
        validation_stats.json_parse_success += 1
    except json.JSONDecodeError:
        validation_stats.json_parse_failed += 1
        return empty

    # Ensure all expected keys exist and count detections
    for key in ["fr", "de", "en", "lb", "pt", "nl", "other"]:
        if key not in data:
            data[key] = None
        elif data[key] is True:
            # Count detections
            if key == "fr":
                validation_stats.detected_fr += 1
            elif key == "de":
                validation_stats.detected_de += 1
            elif key == "en":
                validation_stats.detected_en += 1
            elif key == "lb":
                validation_stats.detected_lb += 1
            elif key == "pt":
                validation_stats.detected_pt += 1
            elif key == "nl":
                validation_stats.detected_nl += 1
            elif key == "other":
                validation_stats.detected_other += 1

    return data


# =============================================================================
# Main Pipeline
# =============================================================================


async def main_async() -> None:
    """Main extraction pipeline."""
    print(
        "===============================================================================",
        flush=True,
    )
    print("[LOAD] Loading sample and raw data for all years...", flush=True)

    # Load full sample
    df_sample = pl.scan_parquet(SAMPLE_FILE).collect()
    print(f"  Total sample pages: {len(df_sample):,}", flush=True)

    if len(df_sample) == 0:
        print("  No pages in sample, exiting.", flush=True)
        return

    # Load raw HTML data for all years
    all_gz_files = []
    for year in YEARS:
        raw_dir = RAW_DATA_DIR / str(year) / "gz"
        gz_files = sorted(raw_dir.glob("*.gz"))
        all_gz_files.extend(gz_files)
        print(f"  Found {len(gz_files)} raw gz files for {year}", flush=True)

    print(f"  Total raw gz files: {len(all_gz_files):,}", flush=True)

    if not all_gz_files:
        print("  No raw files found, exiting.", flush=True)
        return

    # Load raw data and join with sample
    print("[JOIN] Joining sample with raw HTML...", flush=True)

    df_raw = (
        pl.scan_parquet(all_gz_files)
        .select(["url", "year", "html"])
        .rename({"url": "page_url"})
    )

    # Join: keep only pages that are in our sample
    df = (
        df_sample.lazy()
        .join(df_raw, on=["page_url", "year"], how="inner")
        .collect(engine="streaming")
    )

    print(f"  Matched pages with HTML: {len(df):,}", flush=True)

    if len(df) == 0:
        print("  No matches found, exiting.", flush=True)
        return

    # Extract head + navigation from HTML
    print("[EXTRACT] Extracting <head> and navigation from HTML...", flush=True)

    df = df.with_columns(
        pl.col("html")
        .map_elements(extract_head_and_nav, return_dtype=pl.String)
        .alias("html_extract")
    )

    # Drop full HTML to save memory
    df = df.drop("html")

    # Filter out pages with empty extracts
    before_filter = len(df)
    df = df.filter(pl.col("html_extract").str.len_chars() > 100)
    print(
        f"  Pages with valid HTML extract: {len(df):,} (dropped {before_filter - len(df):,})",
        flush=True,
    )

    if len(df) == 0:
        print("  No valid HTML extracts, exiting.", flush=True)
        return

    # Build prompts
    print("[PROMPT] Building prompts...", flush=True)
    prompts = [
        USER_PROMPT.format(html=row["html_extract"]) for row in df.iter_rows(named=True)
    ]

    # Run inference
    print("[INFERENCE] Starting batch inference...", flush=True)
    raw_results = await run_inference(prompts)

    # Track failed requests
    failed_count = sum(1 for r in raw_results if r is None)
    if failed_count > 0:
        print(f"  [WARNING] {failed_count} requests failed", flush=True)

    # Parse responses
    print("[PARSE] Parsing JSON responses...", flush=True)
    parsed_rows = [safe_parse_json(r) for r in raw_results]

    parsed_df = pl.DataFrame(
        parsed_rows,
        schema={
            "fr": pl.Boolean,
            "de": pl.Boolean,
            "en": pl.Boolean,
            "lb": pl.Boolean,
            "pt": pl.Boolean,
            "nl": pl.Boolean,
            "other": pl.Boolean,
        },
    )

    # Print validation stats
    validation_stats.print_summary(len(df))

    # Merge results
    print("[MERGE] Merging results...", flush=True)
    df = df.drop("html_extract")  # Don't need this in output
    df = df.with_columns(parsed_df)
    df = df.with_columns(pl.Series("raw_response", raw_results))

    # Select final columns
    result_df = df.select(
        [
            "website_url",
            "page_url",
            "year",
            "fr",
            "de",
            "en",
            "lb",
            "pt",
            "nl",
            "other",
            "raw_response",
        ]
    )

    # Save results
    print("[SAVE] Writing results...", flush=True)
    result_df.write_parquet(OUTPUT_FILE, compression="zstd", compression_level=10)
    print(f"  Saved to: {OUTPUT_FILE}", flush=True)

    print(
        "===============================================================================",
        flush=True,
    )
    print("Processing complete for all years.", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    asyncio.run(main_async())
