"""
Luxembourg Website Topic Analysis - Step 1: BERTopic Analysis (Page-Level)
==========================================================================
Runs BERTopic on individual web pages to discover topics for a single year.
After clustering, aggregates results back to website-year level.

Each page gets a single topic assignment. A website can have multiple topics
through its different pages, providing a richer topic distribution per website.

Designed to be called via SLURM array job, with YEAR passed as environment variable.

Pipeline:
1. Load individual pages for the specified year
2. Generate sentence embeddings using SentenceTransformer
3. Run BERTopic (UMAP + HDBSCAN + c-TF-IDF) to discover topics
4. Aggregate page-level results to website-year level
5. Save page topics, website topics, summaries, and visualizations

Input:  data/yearly/pages_{YEAR}.parquet (from 00_prepare_data.py)
Output: output/{YEAR}/page_topics.parquet, website_topics.parquet, topic_summary.csv

Author: Julio Garbers with contributions from Claude
Date: February 2026
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
MIN_TOPIC_SIZE = 30
TOP_N_WORDS = 10
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Text processing (per page — individual pages are shorter than aggregated text)
MAX_TEXT_LENGTH = 10000

# Visualization subsampling (UMAP scatter is slow with 100k+ points)
MAX_VIZ_POINTS = 20000


# =============================================================================
# Data Loading
# =============================================================================


def load_year_data(year: int) -> pl.DataFrame:
    input_file = DATA_DIR / f"pages_{year}.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    print(f"\n[LOAD] Loading pages for {year}...", flush=True)
    df = pl.read_parquet(input_file)
    n_websites = df["website_url"].n_unique()
    print(f"   Loaded: {len(df):,} pages from {n_websites:,} websites", flush=True)

    # Truncate very long pages
    df = df.with_columns(
        pl.col("page_text").str.slice(0, MAX_TEXT_LENGTH).alias("text")
    )

    # Filter out very short pages
    df = df.filter(pl.col("text").str.len_chars() >= 50)
    print(f"   After filtering: {len(df):,} pages", flush=True)

    return df


# =============================================================================
# BERTopic Analysis
# =============================================================================


def run_bertopic(
    texts: list[str],
) -> tuple[BERTopic, list[int], pl.DataFrame, np.ndarray]:
    print("\n" + "=" * 70, flush=True)
    print("[BERTOPIC] Running BERTopic Analysis (Page-Level)", flush=True)
    print("=" * 70, flush=True)

    # Generate embeddings
    print(f"\n   Generating embeddings for {len(texts):,} pages...", flush=True)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODEL_DIR))
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=256)
    print(f"   [OK] Embeddings shape: {embeddings.shape}", flush=True)

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

    print("\n   Training BERTopic model...", flush=True)
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Get topic info
    topic_info_pd = topic_model.get_topic_info()
    topic_info = pl.from_pandas(topic_info_pd)

    # Summary statistics
    n_topics = len(topic_info) - 1
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = n_outliers / len(topics) * 100

    print(f"\n   [OK] Discovered {n_topics} topics", flush=True)
    print(f"   [OK] Page outliers: {n_outliers:,} ({outlier_pct:.1f}%)", flush=True)

    return topic_model, topics, topic_info, embeddings


# =============================================================================
# Website-Level Aggregation
# =============================================================================


def aggregate_to_websites(
    df: pl.DataFrame, topics: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Aggregate page-level topics to website-year level.

    Returns:
        page_df: Page-level DataFrame with topic assignments
        website_summary: Website-level summary with primary topic and stats
    """
    print("\n[AGGREGATE] Computing website-level topic distributions...", flush=True)

    page_df = df.select(["website_url", "year"]).with_columns(
        pl.Series("topic", topics)
    )

    n_websites = page_df["website_url"].n_unique()

    # Primary topic per website: most frequent non-outlier topic
    primary_topics = (
        page_df.filter(pl.col("topic") != -1)
        .group_by(["website_url", "year", "topic"])
        .agg(pl.len().alias("topic_pages"))
        .sort(["website_url", "year", "topic_pages"], descending=[False, False, True])
        .group_by(["website_url", "year"])
        .first()
        .select(["website_url", "year", "topic"])
        .rename({"topic": "primary_topic"})
    )

    # Website summary
    website_summary = (
        page_df.group_by(["website_url", "year"])
        .agg(
            [
                pl.len().alias("n_pages"),
                (pl.col("topic") != -1).sum().alias("n_classified"),
                pl.col("topic")
                .filter(pl.col("topic") != -1)
                .n_unique()
                .alias("n_topics"),
            ]
        )
        .join(primary_topics, on=["website_url", "year"], how="left")
        .with_columns(pl.col("primary_topic").fill_null(-1))
    )

    n_classified = website_summary.filter(pl.col("primary_topic") != -1).height
    n_outlier = website_summary.filter(pl.col("primary_topic") == -1).height

    # Avg topics per classified website
    classified = website_summary.filter(pl.col("n_topics") > 0)
    avg_topics = classified["n_topics"].mean() if len(classified) > 0 else 0

    print(f"   Total websites: {n_websites:,}", flush=True)
    print(
        f"   Classified websites: {n_classified:,} ({n_classified/n_websites*100:.1f}%)",
        flush=True,
    )
    print(f"   Outlier websites (all pages outlier): {n_outlier:,}", flush=True)
    print(f"   Avg topics per classified website: {avg_topics:.1f}", flush=True)

    return page_df, website_summary


# =============================================================================
# Save Results
# =============================================================================


def save_results(
    df: pl.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    topic_info: pl.DataFrame,
    page_df: pl.DataFrame,
    website_summary: pl.DataFrame,
    embeddings: np.ndarray,
    year: int,
) -> None:
    year_output_dir = OUTPUT_DIR / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] Saving results to {year_output_dir}", flush=True)

    # 1. Page-level topic assignments (without text to save space)
    page_df.write_parquet(
        year_output_dir / "page_topics.parquet",
        compression="zstd",
        compression_level=10,
    )
    print(
        f"   [OK] Page topics: page_topics.parquet ({len(page_df):,} pages)", flush=True
    )

    # 2. Website-level summary
    website_summary.write_parquet(
        year_output_dir / "website_topics.parquet",
        compression="zstd",
        compression_level=10,
    )
    print(
        f"   [OK] Website topics: website_topics.parquet ({len(website_summary):,} websites)",
        flush=True,
    )

    # 3. Topic summary with both page and website counts
    website_counts = (
        page_df.filter(pl.col("topic") != -1)
        .group_by("topic")
        .agg(pl.col("website_url").n_unique().alias("website_count"))
    )
    website_count_dict = dict(
        zip(
            website_counts["topic"].to_list(),
            website_counts["website_count"].to_list(),
        )
    )

    n_outlier_websites = website_summary.filter(pl.col("primary_topic") == -1).height

    topic_summary = []
    for row in topic_info.iter_rows(named=True):
        topic_id = row["Topic"]
        page_count = row["Count"]

        if topic_id == -1:
            website_count = n_outlier_websites
        else:
            website_count = website_count_dict.get(topic_id, 0)

        # Get representative docs
        if topic_id != -1 and page_count >= 3:
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
                "page_count": page_count,
                "website_count": website_count,
                "name": row["Name"],
                "top_words": top_words,
                "representative_docs": rep_docs_str,
            }
        )

    topic_summary_df = pl.DataFrame(topic_summary)
    topic_summary_df.write_csv(year_output_dir / "topic_summary.csv")
    print("   [OK] Topic summary: topic_summary.csv", flush=True)

    # 4. Embeddings
    np.save(year_output_dir / "embeddings.npy", embeddings)
    print(
        f"   [OK] Embeddings: embeddings.npy ({embeddings.nbytes / 1e6:.1f} MB)",
        flush=True,
    )

    # 5. Metadata
    n_pages = len(df)
    n_websites = df["website_url"].n_unique()
    n_outlier_pages = sum(1 for t in topics if t == -1)
    n_classified_websites = website_summary.filter(pl.col("primary_topic") != -1).height

    classified = website_summary.filter(pl.col("n_topics") > 0)
    avg_topics = (
        round(classified["n_topics"].mean(), 1) if len(classified) > 0 else 0
    )

    metadata = {
        "year": year,
        "n_pages": n_pages,
        "n_websites": n_websites,
        "n_topics": len(topic_info) - 1,
        "n_outlier_pages": n_outlier_pages,
        "n_outlier_websites": n_outlier_websites,
        "page_outlier_pct": round(n_outlier_pages / n_pages * 100, 1),
        "website_outlier_pct": round(n_outlier_websites / n_websites * 100, 1),
        "n_classified_websites": n_classified_websites,
        "avg_topics_per_website": avg_topics,
        "embedding_model": EMBEDDING_MODEL,
        "min_topic_size": MIN_TOPIC_SIZE,
    }
    with open(year_output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   [OK] Metadata: metadata.json", flush=True)


def create_visualizations(
    topic_model: BERTopic,
    topics: list[int],
    embeddings: np.ndarray,
    year: int,
) -> None:
    year_output_dir = OUTPUT_DIR / str(year)
    year_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[VIZ] Creating visualizations...", flush=True)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_info_filtered = topic_info[topic_info["Topic"] != -1].head(15)

    # 1. Topic distribution (horizontal bar chart)
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        topics_sorted = topic_info_filtered.sort_values("Count", ascending=True)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(topics_sorted)))
        ax.barh(
            range(len(topics_sorted)),
            topics_sorted["Count"],
            color=colors,
        )
        ax.set_yticks(range(len(topics_sorted)))
        ax.set_yticklabels(topics_sorted["Name"], fontsize=9)
        ax.set_xlabel("Number of Pages", fontsize=11)
        ax.set_title(
            f"Topic Distribution (Pages) - {year}", fontsize=14, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            year_output_dir / "topic_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("   [OK] topic_distribution.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_distribution: {e}", flush=True)

    # 2. Top words per topic (grid of horizontal bar charts)
    try:
        n_topics = min(12, len(topic_info_filtered))
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
        axes = axes.flatten() if n_topics > 1 else [axes]

        for idx, topic_id in enumerate(topic_info_filtered["Topic"].head(n_topics)):
            ax = axes[idx]
            words_scores = topic_model.get_topic(topic_id)[:8]
            words = [w for w, _ in words_scores]
            scores = [s for _, s in words_scores]

            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words)))
            ax.barh(range(len(words)), scores[::-1], color=colors[::-1])
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words[::-1], fontsize=8)
            ax.set_title(f"Topic {topic_id}", fontsize=10, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)

        # Hide empty subplots
        for idx in range(n_topics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"Top Words per Topic - {year}", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(year_output_dir / "topic_words.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_words.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_words: {e}", flush=True)

    # 3. UMAP 2D scatter (subsampled for large datasets)
    try:
        if len(embeddings) > MAX_VIZ_POINTS:
            print(
                f"   Subsampling {MAX_VIZ_POINTS:,} of {len(embeddings):,} pages for UMAP scatter...",
                flush=True,
            )
            rng = np.random.RandomState(42)
            idx = rng.choice(len(embeddings), MAX_VIZ_POINTS, replace=False)
            viz_embeddings = embeddings[idx]
            viz_topics = np.array(topics)[idx]
        else:
            viz_embeddings = embeddings
            viz_topics = np.array(topics)

        umap_2d = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        embeddings_2d = umap_2d.fit_transform(viz_embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))

        unique_topics = sorted(set(viz_topics))

        # Plot outliers first (in gray)
        outlier_mask = viz_topics == -1
        if outlier_mask.any():
            ax.scatter(
                embeddings_2d[outlier_mask, 0],
                embeddings_2d[outlier_mask, 1],
                c="lightgray",
                alpha=0.2,
                s=10,
                label="Outliers",
            )

        # Plot topics with colors
        non_outlier_topics = [t for t in unique_topics if t != -1]
        colors = plt.cm.tab20(np.linspace(0, 1, len(non_outlier_topics)))

        for idx, topic_id in enumerate(non_outlier_topics[:20]):
            mask = viz_topics == topic_id
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx % 20]],
                alpha=0.5,
                s=15,
                label=f"Topic {topic_id}",
            )

        ax.set_title(
            f"Page Topics (UMAP) - {year}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend outside plot
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

        plt.tight_layout()
        plt.savefig(year_output_dir / "topic_umap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_umap.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_umap: {e}", flush=True)

    # 4. Topic similarity heatmap
    try:
        topic_ids = [t for t in sorted(set(topics)) if t != -1][:15]
        topic_embeddings = []

        for topic_id in topic_ids:
            mask = np.array(topics) == topic_id
            if mask.any():
                topic_embeddings.append(embeddings[mask].mean(axis=0))

        if len(topic_embeddings) > 1:
            topic_embeddings = np.array(topic_embeddings)
            similarity_matrix = cosine_similarity(topic_embeddings)

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(similarity_matrix, cmap="YlOrRd", aspect="auto")

            ax.set_xticks(range(len(topic_ids)))
            ax.set_yticks(range(len(topic_ids)))
            ax.set_xticklabels([f"T{t}" for t in topic_ids], fontsize=9)
            ax.set_yticklabels([f"T{t}" for t in topic_ids], fontsize=9)

            plt.colorbar(im, ax=ax, label="Cosine Similarity")
            ax.set_title(
                f"Topic Similarity - {year}", fontsize=14, fontweight="bold"
            )

            plt.tight_layout()
            plt.savefig(
                year_output_dir / "topic_similarity.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            print("   [OK] topic_similarity.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_similarity: {e}", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print(f"Luxembourg Website Topic Analysis - Year {YEAR} (Page-Level)", flush=True)
    print("=" * 70, flush=True)
    print("\nConfiguration:", flush=True)
    print(f"   Year: {YEAR}", flush=True)
    print(f"   Min topic size: {MIN_TOPIC_SIZE}", flush=True)
    print(f"   Max text length: {MAX_TEXT_LENGTH}", flush=True)
    print(f"   Embedding model: {EMBEDDING_MODEL}", flush=True)

    # Load data
    df = load_year_data(YEAR)
    texts = df["text"].to_list()

    if len(texts) < 50:
        print(
            f"\n[WARN] Only {len(texts)} pages. Results may not be meaningful.",
            flush=True,
        )

    # Run BERTopic
    topic_model, topics, topic_info, embeddings = run_bertopic(texts)

    # Aggregate to website level
    page_df, website_summary = aggregate_to_websites(df, topics)

    # Save results
    save_results(
        df, topic_model, topics, topic_info, page_df, website_summary, embeddings, YEAR
    )

    # Create visualizations
    create_visualizations(topic_model, topics, embeddings, YEAR)

    # Print top topics
    print("\n" + "=" * 70, flush=True)
    print(f"TOP TOPICS FOR {YEAR} (by website count)", flush=True)
    print("=" * 70, flush=True)

    topic_summary = pl.read_csv(OUTPUT_DIR / str(YEAR) / "topic_summary.csv")
    top = (
        topic_summary.filter(pl.col("topic_id") != -1)
        .sort("website_count", descending=True)
        .head(10)
    )

    for row in top.iter_rows(named=True):
        print(
            f"\n   Topic {row['topic_id']} ({row['website_count']} websites, {row['page_count']} pages)",
            flush=True,
        )
        print(f"   {row['name']}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"DONE! Results saved to: {OUTPUT_DIR / str(YEAR)}", flush=True)
    print("=" * 70, flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
