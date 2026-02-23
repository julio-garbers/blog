"""
Luxembourg Website Topic Analysis - Step 2: Global BERTopic
=============================================================
Runs BERTopic on ALL website-years at once (global model), then uses
topics_over_time() for consistent cross-year tracking.

This script:
1. Loads pre-computed embeddings and texts for all years
2. Runs BERTopic globally (UMAP + HDBSCAN + c-TF-IDF)
3. Calls topics_over_time() for evolution tracking
4. Saves results: website_topics.parquet, topic_info.csv,
   topics_over_time.csv, visualizations, and stats.json

Using a global model means topic IDs are consistent across years —
no need for heuristic cross-year matching.

Author: Julio Garbers with contributions from Claude
Date: February 2026
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from umap import UMAP

warnings.filterwarnings("ignore")


# =============================================================================
# Stopwords
# =============================================================================

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

DATA_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/data/yearly")
EMBED_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/output/embeddings")
OUTPUT_DIR = Path("/project/home/p200812/blog/bert_topic_websites_lux/output")

# BERTopic Parameters
MIN_TOPIC_SIZE = 50
TOP_N_WORDS = 10

# Max text length for c-TF-IDF (can be longer than embedding input)
TFIDF_MAX_LENGTH = 50000

# Years to include
YEARS = list(range(2013, 2025))

# Visualization subsampling
MAX_VIZ_POINTS = 20000


# =============================================================================
# Data Loading
# =============================================================================


def load_all_data() -> tuple[np.ndarray, list[str], list[str], pl.DataFrame]:
    """Load embeddings, texts, and timestamps for all years.

    Returns:
        embeddings: (N, 1024) array of all website embeddings
        texts: List of clean_text strings (truncated for c-TF-IDF)
        timestamps: List of year strings for topics_over_time
        metadata_df: DataFrame with website_url, year, n_pages columns
    """
    print("\n[LOAD] Loading all years...", flush=True)

    all_embeddings = []
    all_texts = []
    all_timestamps = []
    all_meta = []

    for year in YEARS:
        embed_file = EMBED_DIR / f"embeddings_{year}.npy"
        order_file = EMBED_DIR / f"order_{year}.parquet"
        data_file = DATA_DIR / f"websites_{year}.parquet"

        if not embed_file.exists():
            print(f"   [SKIP] {year}: embeddings not found", flush=True)
            continue

        # Load embeddings
        embeddings = np.load(embed_file)

        # Load order (website_url, year) to align with embeddings
        order_df = pl.read_parquet(order_file)

        # Load full text data
        data_df = pl.read_parquet(data_file)

        # Align text with embedding order
        aligned = order_df.join(data_df, on=["website_url", "year"], how="left")

        texts = aligned["clean_text"].str.slice(0, TFIDF_MAX_LENGTH).to_list()
        timestamps = [str(year)] * len(texts)

        all_embeddings.append(embeddings)
        all_texts.extend(texts)
        all_timestamps.extend(timestamps)
        all_meta.append(aligned.select(["website_url", "year", "n_pages"]))

        print(
            f"   {year}: {len(texts):,} websites, embeddings {embeddings.shape}",
            flush=True,
        )

    embeddings_all = np.vstack(all_embeddings)
    metadata_df = pl.concat(all_meta)

    print(f"\n   Total: {len(all_texts):,} website-years", flush=True)
    print(
        f"   Embeddings: {embeddings_all.shape} ({embeddings_all.nbytes / 1e6:.0f} MB)",
        flush=True,
    )

    return embeddings_all, all_texts, all_timestamps, metadata_df


# =============================================================================
# BERTopic Analysis
# =============================================================================


def run_bertopic(
    embeddings: np.ndarray,
    texts: list[str],
    timestamps: list[str],
) -> tuple[BERTopic, list[int], pl.DataFrame]:
    print("\n" + "=" * 70, flush=True)
    print("[BERTOPIC] Running Global BERTopic Analysis", flush=True)
    print("=" * 70, flush=True)
    print(f"   Documents: {len(texts):,}", flush=True)
    print(f"   Embedding dim: {embeddings.shape[1]}", flush=True)
    print(f"   Min topic size: {MIN_TOPIC_SIZE}", flush=True)

    # UMAP: 1024D -> 5D
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    # HDBSCAN clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words=ALL_STOPWORDS,
        min_df=3,
        ngram_range=(1, 2),
        max_features=10000,
    )

    # Create BERTopic model (no embedding model - we pass pre-computed)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=TOP_N_WORDS,
        verbose=True,
        calculate_probabilities=False,
    )

    # Fit on pre-computed embeddings, full text for c-TF-IDF
    print("\n   Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)...", flush=True)
    start = time.time()
    topics, _ = topic_model.fit_transform(texts, embeddings)
    elapsed = time.time() - start
    print(f"   [OK] BERTopic fit in {elapsed:.1f}s", flush=True)

    # Topic info
    topic_info_pd = topic_model.get_topic_info()
    topic_info = pl.from_pandas(topic_info_pd)

    n_topics = len(topic_info) - 1  # Exclude outlier topic -1
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_pct = n_outliers / len(topics) * 100

    print(f"\n   Discovered {n_topics} topics", flush=True)
    print(f"   Outliers: {n_outliers:,} ({outlier_pct:.1f}%)", flush=True)

    # Topics over time
    print("\n   Computing topics_over_time()...", flush=True)
    start = time.time()
    tot_df = topic_model.topics_over_time(
        texts,
        timestamps,
        nr_bins=None,
        evolution_tuning=True,
        global_tuning=True,
    )
    elapsed = time.time() - start
    print(f"   [OK] topics_over_time in {elapsed:.1f}s", flush=True)

    topics_over_time = pl.from_pandas(tot_df)
    print(f"   Topics over time rows: {len(topics_over_time):,}", flush=True)

    return topic_model, topics, topics_over_time


# =============================================================================
# Save Results
# =============================================================================


def save_results(
    topic_model: BERTopic,
    topics: list[int],
    topics_over_time: pl.DataFrame,
    metadata_df: pl.DataFrame,
    timestamps: list[str],
    embeddings: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] Saving results to {OUTPUT_DIR}", flush=True)

    # 1. Website topics (per website-year)
    website_topics = metadata_df.with_columns(pl.Series("topic", topics))
    website_topics.write_parquet(
        OUTPUT_DIR / "website_topics.parquet",
        compression="zstd",
        compression_level=10,
    )
    print(f"   [OK] website_topics.parquet ({len(website_topics):,} rows)", flush=True)

    # 2. Topic info with representative docs
    topic_info_pd = topic_model.get_topic_info()
    topic_info = pl.from_pandas(topic_info_pd)

    topic_summary = []
    for row in topic_info.iter_rows(named=True):
        topic_id = row["Topic"]
        count = row["Count"]

        # Top words
        if topic_id != -1:
            try:
                top_words = ", ".join(
                    [word for word, _ in topic_model.get_topic(topic_id)[:TOP_N_WORDS]]
                )
            except Exception:
                top_words = ""
        else:
            top_words = "outliers"

        # Representative docs
        rep_docs_str = ""
        if topic_id != -1 and count >= 3:
            try:
                rep_docs = topic_model.get_representative_docs(topic_id)[:3]
                rep_docs_str = "\n---\n".join(
                    [doc[:500] + "..." if len(doc) > 500 else doc for doc in rep_docs]
                )
            except Exception:
                pass

        topic_summary.append(
            {
                "topic_id": topic_id,
                "count": count,
                "name": row["Name"],
                "top_words": top_words,
                "representative_docs": rep_docs_str,
            }
        )

    topic_summary_df = pl.DataFrame(topic_summary)
    topic_summary_df.write_csv(OUTPUT_DIR / "topic_info.csv")
    print(f"   [OK] topic_info.csv ({len(topic_summary_df)} topics)", flush=True)

    # 3. Topics over time
    topics_over_time.write_csv(OUTPUT_DIR / "topics_over_time.csv")
    print(f"   [OK] topics_over_time.csv ({len(topics_over_time):,} rows)", flush=True)

    # 4. Metadata
    n_total = len(topics)
    n_outliers = sum(1 for t in topics if t == -1)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)

    years_present = sorted(set(timestamps))
    yearly_meta = {}
    for yr_str in years_present:
        yr_mask = [t for t, ts in zip(topics, timestamps) if ts == yr_str]
        yr_outliers = sum(1 for t in yr_mask if t == -1)
        yearly_meta[yr_str] = {
            "n_websites": len(yr_mask),
            "n_outliers": yr_outliers,
            "outlier_pct": round(yr_outliers / len(yr_mask) * 100, 1) if yr_mask else 0,
            "n_classified": len(yr_mask) - yr_outliers,
        }

    metadata = {
        "total_website_years": n_total,
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "outlier_pct": round(n_outliers / n_total * 100, 1),
        "min_topic_size": MIN_TOPIC_SIZE,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dim": int(embeddings.shape[1]),
        "years": years_present,
        "yearly": yearly_meta,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("   [OK] metadata.json", flush=True)


def build_stats_json(
    topic_model: BERTopic,
    topics: list[int],
    topics_over_time: pl.DataFrame,
    metadata_df: pl.DataFrame,
    timestamps: list[str],
) -> None:
    """Build stats.json for the blog post visualization."""
    print("\n[STATS] Building stats.json for blog post...", flush=True)

    website_topics = metadata_df.with_columns(pl.Series("topic", topics))

    # --- Yearly stats ---
    yearly_stats = []
    years = sorted(set(timestamps))

    for yr_str in years:
        year = int(yr_str)
        yr_df = website_topics.filter(pl.col("year") == year)
        n_websites = len(yr_df)
        n_outliers = yr_df.filter(pl.col("topic") == -1).height
        n_classified = n_websites - n_outliers
        n_pages = yr_df["n_pages"].sum()

        # Top 5 topics for this year
        top = (
            yr_df.filter(pl.col("topic") != -1)
            .group_by("topic")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(5)
        )

        top_topics = []
        for row in top.iter_rows(named=True):
            tid = row["topic"]
            try:
                words = ", ".join(
                    [w for w, _ in topic_model.get_topic(tid)[:TOP_N_WORDS]]
                )
            except Exception:
                words = ""
            top_topics.append(
                {
                    "topic_id": tid,
                    "count": row["count"],
                    "top_words": words,
                }
            )

        yearly_stats.append(
            {
                "year": year,
                "n_websites": n_websites,
                "n_pages": int(n_pages),
                "n_topics_active": yr_df.filter(pl.col("topic") != -1)[
                    "topic"
                ].n_unique(),
                "n_outliers": n_outliers,
                "outlier_pct": round(n_outliers / n_websites * 100, 1),
                "n_classified": n_classified,
                "top_topics": top_topics,
            }
        )

    # --- Topic info for latest year (2024) treemap ---
    latest_year = max(int(y) for y in years)
    latest_df = website_topics.filter(
        (pl.col("year") == latest_year) & (pl.col("topic") != -1)
    )
    topic_counts = (
        latest_df.group_by("topic")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    topics_latest = []
    for row in topic_counts.iter_rows(named=True):
        tid = row["topic"]
        try:
            words = [w for w, _ in topic_model.get_topic(tid)[:TOP_N_WORDS]]
        except Exception:
            words = []
        topics_latest.append(
            {
                "topic_id": tid,
                "count": row["count"],
                "top_words": words,
            }
        )

    # --- Topics over time (for evolution chart) ---
    # Filter to topics with at least 10 websites in some year
    tot = topics_over_time.clone()

    # Get top N topics by total frequency across all years
    top_n = 20
    if "Topic" in tot.columns and "Frequency" in tot.columns:
        top_topic_ids = (
            tot.filter(pl.col("Topic") != -1)
            .group_by("Topic")
            .agg(pl.col("Frequency").sum().alias("total"))
            .sort("total", descending=True)
            .head(top_n)["Topic"]
            .to_list()
        )

        evolution = []
        for row in tot.filter(pl.col("Topic").is_in(top_topic_ids)).iter_rows(
            named=True
        ):
            tid = row["Topic"]
            try:
                words = ", ".join([w for w, _ in topic_model.get_topic(tid)[:5]])
            except Exception:
                words = f"Topic {tid}"
            # Timestamp can be string or datetime — convert to year
            ts = row["Timestamp"]
            if hasattr(ts, "year"):
                year_val = ts.year
            else:
                year_val = int(str(ts)[:4])

            evolution.append(
                {
                    "topic_id": tid,
                    "year": year_val,
                    "frequency": row["Frequency"],
                    "label": words,
                }
            )
    else:
        evolution = []

    # --- Summary ---
    n_total = len(topics)
    n_outliers_total = sum(1 for t in topics if t == -1)
    n_topics_total = len(set(topics)) - (1 if -1 in topics else 0)

    stats = {
        "summary": {
            "first_year": int(min(years)),
            "last_year": int(max(years)),
            "years_covered": len(years),
            "total_website_years": n_total,
            "n_topics": n_topics_total,
            "outlier_pct": round(n_outliers_total / n_total * 100, 1),
            "embedding_model": "BAAI/bge-m3",
            "method": "global_bertopic_with_topics_over_time",
        },
        "yearly": yearly_stats,
        "topics_latest": topics_latest,
        "evolution": evolution,
    }

    stats_file = OUTPUT_DIR / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   [OK] stats.json saved: {stats_file}", flush=True)


# =============================================================================
# Visualizations
# =============================================================================


def create_visualizations(
    topic_model: BERTopic,
    topics: list[int],
    embeddings: np.ndarray,
) -> None:
    print("\n[VIZ] Creating visualizations...", flush=True)

    topic_info = topic_model.get_topic_info()
    topic_info_filtered = topic_info[topic_info["Topic"] != -1].head(20)

    # 1. Topic distribution bar chart
    try:
        _fig, ax = plt.subplots(figsize=(12, 10))
        topics_sorted = topic_info_filtered.sort_values("Count", ascending=True).tail(
            20
        )

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(topics_sorted)))
        ax.barh(range(len(topics_sorted)), topics_sorted["Count"], color=colors)
        ax.set_yticks(range(len(topics_sorted)))
        ax.set_yticklabels(topics_sorted["Name"], fontsize=9)
        ax.set_xlabel("Number of Website-Years", fontsize=11)
        ax.set_title("Top 20 Topics (Global Model)", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "topic_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_distribution.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_distribution: {e}", flush=True)

    # 2. UMAP 2D scatter (subsampled)
    try:
        n_points = len(embeddings)
        if n_points > MAX_VIZ_POINTS:
            print(
                f"   Subsampling {MAX_VIZ_POINTS:,} of {n_points:,} for UMAP scatter...",
                flush=True,
            )
            rng = np.random.RandomState(42)
            idx = rng.choice(n_points, MAX_VIZ_POINTS, replace=False)
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

        _fig, ax = plt.subplots(figsize=(14, 10))

        # Outliers in gray
        outlier_mask = viz_topics == -1
        if outlier_mask.any():
            ax.scatter(
                embeddings_2d[outlier_mask, 0],
                embeddings_2d[outlier_mask, 1],
                c="lightgray",
                alpha=0.15,
                s=8,
                label="Outliers",
            )

        # Top topics with colors
        non_outlier = [t for t in sorted(set(viz_topics)) if t != -1]
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(non_outlier))))
        for i, tid in enumerate(non_outlier[:20]):
            mask = viz_topics == tid
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i % 20]],
                alpha=0.5,
                s=12,
                label=f"Topic {tid}",
            )

        ax.set_title(
            "Website Topics (UMAP) - Global Model", fontsize=14, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "topic_umap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_umap.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_umap: {e}", flush=True)

    # 3. Top words per topic grid
    try:
        n_topics = min(12, len(topic_info_filtered))
        n_cols = 3
        n_rows = (n_topics + n_cols - 1) // n_cols

        _fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5))
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

        for idx in range(n_topics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Top Words per Topic (Global Model)", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "topic_words.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("   [OK] topic_words.png", flush=True)
    except Exception as e:
        print(f"   [WARN] Skipped topic_words: {e}", flush=True)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70, flush=True)
    print("Luxembourg Website Topic Analysis - Global BERTopic", flush=True)
    print("=" * 70, flush=True)
    print("\nConfiguration:", flush=True)
    print(f"   Years: {YEARS[0]}-{YEARS[-1]}", flush=True)
    print(f"   Min topic size: {MIN_TOPIC_SIZE}", flush=True)
    print(f"   c-TF-IDF max text: {TFIDF_MAX_LENGTH:,} chars", flush=True)

    # Load all data
    embeddings, texts, timestamps, metadata_df = load_all_data()

    # Run BERTopic globally
    topic_model, topics, topics_over_time = run_bertopic(embeddings, texts, timestamps)

    # Save results
    save_results(
        topic_model,
        topics,
        topics_over_time,
        metadata_df,
        timestamps,
        embeddings,
    )

    # Build blog stats.json
    build_stats_json(
        topic_model,
        topics,
        topics_over_time,
        metadata_df,
        timestamps,
    )

    # Create visualizations
    create_visualizations(topic_model, topics, embeddings)

    # Print top topics
    print("\n" + "=" * 70, flush=True)
    print("TOP 15 TOPICS (by website-year count)", flush=True)
    print("=" * 70, flush=True)

    topic_info = topic_model.get_topic_info()
    top = topic_info[topic_info["Topic"] != -1].head(15)

    for _, row in top.iterrows():
        tid = row["Topic"]
        try:
            words = ", ".join([w for w, _ in topic_model.get_topic(tid)[:8]])
        except Exception:
            words = ""
        print(f"\n   Topic {tid} ({row['Count']:,} website-years)", flush=True)
        print(f"   {words}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("DONE!", flush=True)
    print(f"Results: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
