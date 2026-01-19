"""
Luxembourg Website Topic Analysis - Step 1: BERTopic Analysis
=============================================================
Runs BERTopic on aggregated website text to discover topics for a single year.
Designed to be called via SLURM array job, with YEAR passed as environment variable.

Pipeline:
1. Load aggregated website text for the specified year
2. Generate sentence embeddings using SentenceTransformer
3. Run BERTopic (UMAP + HDBSCAN + c-TF-IDF) to discover topics
4. Save topic assignments, summaries, and visualizations

Input:  data/yearly/websites_{YEAR}.parquet (from 00_prepare_data.py)
Output: output/{YEAR}/website_topics.parquet, topic_summary.csv, visualizations

Author: Julio Garbers with contributions from Claude
Date: January 2026
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from umap import UMAP

warnings.filterwarnings("ignore")


# =============================================================================
# Stopwords
# =============================================================================

# Luxembourgish stopwords (not in stopwordsiso)
LUXEMBOURGISH_STOPS = {
    "an",
    "ass",
    "bei",
    "dat",
    "de",
    "den",
    "déi",
    "dir",
    "d'",
    "e",
    "een",
    "en",
    "eng",
    "et",
    "fir",
    "hien",
    "hun",
    "ech",
    "mat",
    "mir",
    "no",
    "ob",
    "op",
    "seng",
    "si",
    "sinn",
    "vun",
    "wann",
    "wat",
    "wéi",
    "ze",
}

# Combined stopwords for all Luxembourg languages
ALL_STOPWORDS = list(
    stopwords("en")
    | stopwords("fr")
    | stopwords("de")
    | stopwords("pt")
    | stopwords("nl")
    | LUXEMBOURGISH_STOPS
)


# =============================================================================
# Configuration
# =============================================================================

# Year to process (from environment variable)
YEAR = int(os.environ.get("YEAR"))

# Input/Output Paths
DATA_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/output")
MODEL_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/models")

# BERTopic Parameters
MIN_TOPIC_SIZE = 10
TOP_N_WORDS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Text processing
MAX_TEXT_LENGTH = 15000


# =============================================================================
# Data Loading
# =============================================================================


def load_year_data(year: int) -> pl.DataFrame:
    """Load aggregated website data for a specific year."""
    input_file = DATA_DIR / f"websites_{year}.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    print(f"\n[LOAD] Loading data for {year}...")
    df = pl.read_parquet(input_file)
    print(f"   Loaded: {len(df):,} websites")

    # Truncate very long texts
    df = df.with_columns(
        pl.col("aggregated_text").str.slice(0, MAX_TEXT_LENGTH).alias("text")
    )

    # Filter out very short texts
    df = df.filter(pl.col("text").str.len_chars() >= 50)
    print(f"   After filtering: {len(df):,} websites")

    return df


# =============================================================================
# BERTopic Analysis
# =============================================================================


def run_bertopic(
    texts: list[str],
) -> tuple[BERTopic, list[int], pl.DataFrame, np.ndarray]:
    """
    Run BERTopic on text data.

    Returns:
        topic_model: Trained BERTopic model
        topics: Topic assignments for each document
        topic_info: DataFrame with topic information
        embeddings: Document embeddings
    """
    print("\n" + "=" * 70)
    print("[BERTOPIC] Running BERTopic Analysis")
    print("=" * 70)

    # Generate embeddings
    print("\n   Generating embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    print(f"   [OK] Embeddings shape: {embeddings.shape}")

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Configure vectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words=ALL_STOPWORDS,
        min_df=2,
        ngram_range=(1, 2),
        max_features=10000,
    )

    # Create and train BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=TOP_N_WORDS,
        verbose=True,
        calculate_probabilities=False,
    )

    print("\n   Training BERTopic model...")
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info_pd = topic_model.get_topic_info()
    topic_info = pl.from_pandas(topic_info_pd)

    # Summary statistics
    n_topics = len(topic_info) - 1
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = n_outliers / len(topics) * 100

    print(f"\n   [OK] Discovered {n_topics} topics")
    print(f"   [OK] Outliers: {n_outliers:,} ({outlier_pct:.1f}%)")

    return topic_model, topics, topic_info, embeddings


# =============================================================================
# Save Results
# =============================================================================


def save_model(topic_model: BERTopic, year: int) -> Path:
    """Save BERTopic model using pickle (safetensors has numpy int64 bug)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"bertopic_{year}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(topic_model, f)

    print(f"\n   [OK] Model saved: {model_path}")
    return model_path


def save_results(
    df: pl.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    topic_info: pl.DataFrame,
    embeddings: np.ndarray,
    year: int,
) -> None:
    """Save all results for a year."""
    year_output_dir = OUTPUT_DIR / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] Saving results to {year_output_dir}")

    # 1. Website-topic assignments
    df_results = df.with_columns(pl.Series("topic", topics))
    df_results.write_parquet(
        year_output_dir / "website_topics.parquet",
        compression="zstd",
        compression_level=10,
    )
    print("   [OK] Website topics: website_topics.parquet")

    # 2. Topic summary
    topic_summary = []
    for row in topic_info.iter_rows(named=True):
        topic_id = row["Topic"]

        # Get representative docs
        if topic_id != -1 and row["Count"] >= 3:
            try:
                rep_docs = topic_model.get_representative_docs(topic_id)[:3]
                rep_docs_str = "\n---\n".join(
                    [doc[:500] + "..." if len(doc) > 500 else doc for doc in rep_docs]
                )
            except Exception:
                rep_docs_str = ""
        else:
            rep_docs_str = ""

        # Get top words
        if topic_id != -1:
            try:
                top_words = ", ".join(
                    [word for word, _ in topic_model.get_topic(topic_id)[:TOP_N_WORDS]]
                )
            except Exception:
                top_words = ""
        else:
            top_words = "outliers"

        topic_summary.append(
            {
                "topic_id": topic_id,
                "count": row["Count"],
                "name": row["Name"],
                "top_words": top_words,
                "representative_docs": rep_docs_str,
            }
        )

    topic_summary_df = pl.DataFrame(topic_summary)
    topic_summary_df.write_csv(year_output_dir / "topic_summary.csv")
    print("   [OK] Topic summary: topic_summary.csv")

    # 3. Embeddings (for later analysis)
    np.save(year_output_dir / "embeddings.npy", embeddings)
    print("   [OK] Embeddings: embeddings.npy")

    # 4. Metadata
    metadata = {
        "year": year,
        "n_websites": len(df),
        "n_topics": len(topic_info) - 1,
        "n_outliers": sum(1 for t in topics if t == -1),
        "embedding_model": EMBEDDING_MODEL,
        "min_topic_size": MIN_TOPIC_SIZE,
    }
    with open(year_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   [OK] Metadata: metadata.json")


def create_visualizations(
    topic_model: BERTopic,
    topics: list[int],
    embeddings: np.ndarray,
    year: int,
) -> None:
    """Create and save visualizations."""
    year_output_dir = OUTPUT_DIR / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[VIZ] Creating visualizations...")

    # 1. Topic barchart
    try:
        fig = topic_model.visualize_barchart(top_n_topics=15, n_words=8)
        fig.write_html(str(year_output_dir / "topic_barchart.html"))
        print("   [OK] topic_barchart.html")
    except Exception as e:
        print(f"   [WARN] Skipped barchart: {e}")

    # 2. Topic map (intertopic distance)
    try:
        fig = topic_model.visualize_topics()
        fig.write_html(str(year_output_dir / "topic_map.html"))
        print("   [OK] topic_map.html")
    except Exception as e:
        print(f"   [WARN] Skipped topic map: {e}")

    # 3. Topic hierarchy
    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(str(year_output_dir / "topic_hierarchy.html"))
        print("   [OK] topic_hierarchy.html")
    except Exception as e:
        print(f"   [WARN] Skipped hierarchy: {e}")

    # 4. 2D scatter plot using PCA
    try:
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=topics,
            cmap="tab20",
            alpha=0.6,
            s=50,
        )
        plt.colorbar(scatter, label="Topic")
        plt.title(f"Luxembourg Website Topics - {year}", fontsize=14, fontweight="bold")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.tight_layout()
        plt.savefig(year_output_dir / "topic_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_scatter.png")
    except Exception as e:
        print(f"   [WARN] Skipped scatter: {e}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print(f"Luxembourg Website Topic Analysis - Year {YEAR}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"   Year: {YEAR}")
    print(f"   Min topic size: {MIN_TOPIC_SIZE}")
    print(f"   Max text length: {MAX_TEXT_LENGTH}")

    # Load data
    df = load_year_data(YEAR)
    texts = df["text"].to_list()

    if len(texts) < 50:
        print(f"\n[WARN] Only {len(texts)} websites. Results may not be meaningful.")

    # Run BERTopic
    topic_model, topics, topic_info, embeddings = run_bertopic(texts)

    # Save model
    save_model(topic_model, YEAR)

    # Save results
    save_results(df, topic_model, topics, topic_info, embeddings, YEAR)

    # Create visualizations
    create_visualizations(topic_model, topics, embeddings, YEAR)

    # Print top topics
    print("\n" + "=" * 70)
    print(f"TOP TOPICS FOR {YEAR}")
    print("=" * 70)

    topic_info_sorted = topic_info.filter(pl.col("Topic") != -1).sort(
        "Count", descending=True
    )

    for row in topic_info_sorted.head(10).iter_rows(named=True):
        print(f"\n   Topic {row['Topic']} (n={row['Count']})")
        print(f"   {row['Name']}")

    print("\n" + "=" * 70)
    print(f"DONE! Results saved to: {OUTPUT_DIR / str(YEAR)}")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
