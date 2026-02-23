"""
Luxembourg Website Topic Analysis - Step 1: Generate Embeddings
================================================================
Encodes website texts using BAAI/bge-m3 (8192-token context, multilingual).

This script:
1. Loads aggregated website text for a single year
2. Truncates to EMBED_MAX_LENGTH characters (~7,000 tokens, within 8,192 limit)
3. Encodes with BAAI/bge-m3 on GPU
4. Saves embeddings as .npy file

Designed to be called via SLURM array job, with YEAR passed as environment variable.

Input:  data/yearly/websites_{YEAR}.parquet (from 00_prepare_data.py)
Output: output/embeddings/embeddings_{YEAR}.npy

Author: Julio Garbers with contributions from Claude
Date: February 2026
"""

import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

# =============================================================================
# Configuration
# =============================================================================

# Year to process (from environment variable)
YEAR = int(os.environ.get("YEAR"))

# Input/Output Paths
DATA_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/output/embeddings")
MODEL_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model: BAAI/bge-m3 (1024-dim, 8192-token context, multilingual)
EMBEDDING_MODEL = "BAAI/bge-m3"

# Max characters to embed (~4 chars per token -> 30,000 chars ~ 7,500 tokens)
EMBED_MAX_LENGTH = 30000

# Batch size for encoding (tune based on GPU memory)
BATCH_SIZE = 32


# =============================================================================
# Data Loading
# =============================================================================


def load_year_data(year: int) -> pl.DataFrame:
    input_file = DATA_DIR / f"websites_{year}.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    print(f"\n[LOAD] Loading websites for {year}...", flush=True)
    df = pl.read_parquet(input_file)
    print(f"   Loaded: {len(df):,} websites", flush=True)

    # Text length stats
    text_lengths = df["clean_text"].str.len_chars()
    print(f"   Text length - mean: {text_lengths.mean():,.0f}, median: {text_lengths.median():,.0f}", flush=True)
    print(f"   Text length - max: {text_lengths.max():,.0f}", flush=True)

    n_truncated = (text_lengths > EMBED_MAX_LENGTH).sum()
    print(f"   Will truncate {n_truncated:,} texts to {EMBED_MAX_LENGTH:,} chars", flush=True)

    return df


# =============================================================================
# Embedding
# =============================================================================


def generate_embeddings(texts: list[str]) -> np.ndarray:
    print(f"\n[EMBED] Loading model: {EMBEDDING_MODEL}", flush=True)

    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODEL_DIR))
    model.max_seq_length = 8192

    print(f"   Model loaded. Max sequence length: {model.max_seq_length}", flush=True)
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}", flush=True)
    print(f"   Encoding {len(texts):,} texts (batch_size={BATCH_SIZE})...", flush=True)

    start = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start

    print(f"   [OK] Embeddings shape: {embeddings.shape}", flush=True)
    print(f"   [OK] Time: {elapsed:.1f}s ({len(texts) / elapsed:.1f} texts/sec)", flush=True)
    print(f"   [OK] Size: {embeddings.nbytes / 1e6:.1f} MB", flush=True)

    return embeddings


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print(f"Luxembourg Website Topic Analysis - Embeddings (Year {YEAR})", flush=True)
    print("=" * 70, flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"   Year: {YEAR}", flush=True)
    print(f"   Model: {EMBEDDING_MODEL}", flush=True)
    print(f"   Max text length: {EMBED_MAX_LENGTH:,} chars", flush=True)
    print(f"   Batch size: {BATCH_SIZE}", flush=True)

    # Load data
    df = load_year_data(YEAR)

    # Truncate texts for embedding
    texts = df["clean_text"].str.slice(0, EMBED_MAX_LENGTH).to_list()

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    # Save embeddings
    output_file = OUTPUT_DIR / f"embeddings_{YEAR}.npy"
    np.save(output_file, embeddings)
    print(f"\n[SAVE] Saved: {output_file}", flush=True)

    # Also save the website order (for alignment with embeddings)
    order_file = OUTPUT_DIR / f"order_{YEAR}.parquet"
    df.select(["website_url", "year"]).write_parquet(
        order_file, compression="zstd", compression_level=10
    )
    print(f"[SAVE] Saved: {order_file}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"DONE! Embeddings for {YEAR}: {embeddings.shape}", flush=True)
    print("=" * 70, flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
