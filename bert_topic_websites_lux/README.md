# What Does Luxembourg's Web Talk About?

Topic modeling of Luxembourg (.lu) websites from 2013 to 2024 using BERTopic and CommonCrawl archives.

## Overview

This project applies unsupervised topic modeling to discover what themes Luxembourg websites cover and how the thematic landscape evolves over twelve years. Using the same CommonCrawl sample as the [language analysis](https://github.com/julio-garbers/blog/tree/main/website_languages_lux), we run BERTopic on ~81,000 website-years with a global model for consistent cross-year topic tracking.

**Key design choices:**
- **Long-context embeddings** (BAAI/bge-m3, 8,192 tokens) capture full website content instead of just the first ~100 words
- **Paragraph-level deduplication** removes repeated navigation, footers, and cookie banners within each website
- **Global BERTopic** model across all years produces consistent topic IDs — no heuristic cross-year matching needed
- **`topics_over_time()`** built-in method for tracking topic evolution

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      00_prepare_data.py                         │
│   Aggregate pages per website-year with paragraph deduplication │
│   Output: websites_{YEAR}.parquet                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        01_embed.py                              │
│                  (SLURM array job: 1 per year, GPU)             │
│                                                                 │
│  Encode website text with BAAI/bge-m3 (1024-dim, 8192 tokens)  │
│  Output: embeddings_{YEAR}.npy                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      02_bert_topic.py                           │
│              (Single job, CPU/largemem)                          │
│                                                                 │
│  1. Load ALL embeddings + texts (~81,000 website-years)         │
│  2. UMAP dimensionality reduction (1024D → 5D)                 │
│  3. HDBSCAN clustering                                          │
│  4. c-TF-IDF on full text (up to 50,000 chars)                 │
│  5. topics_over_time() for evolution tracking                   │
│  6. Visualizations + stats.json                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      03_blog_stats.py                            │
│              (Lightweight, CPU)                                   │
│                                                                  │
│  Reads pipeline output + sector_mapping.json                     │
│  Aggregates topics into sectors, computes shares over time       │
│  Output: blog_stats.json (copy to blog directory)                │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
blog/
├── bert_topic_websites_lux/
│   ├── script/
│   │   ├── 00_prepare_data.py      # Aggregate + paragraph dedup
│   │   ├── 00_prepare_data.sh      # SLURM job script (CPU)
│   │   ├── 01_embed.py             # GPU embedding with bge-m3
│   │   ├── 01_embed.sh             # SLURM array job (GPU, 0-11)
│   │   ├── 02_bert_topic.py        # Global BERTopic + topics_over_time
│   │   ├── 02_bert_topic.sh        # SLURM job script (largemem)
│   │   ├── 03_blog_stats.py        # Generate blog stats from output + mapping
│   │   ├── 03_blog_stats.sh        # SLURM job script (CPU)
│   │   └── slurm/                  # SLURM output logs
│   ├── sector_mapping.json             # Manual topic → sector assignments
│   ├── data/
│   │   ├── metadata.json           # Year/website counts
│   │   └── yearly/                 # Per-year parquet files (on HPC)
│   │       └── websites_{YEAR}.parquet  # Deduplicated website text
│   ├── output/
│   │   ├── embeddings/             # Pre-computed embeddings
│   │   │   ├── embeddings_{YEAR}.npy
│   │   │   └── order_{YEAR}.parquet
│   │   ├── website_topics.parquet  # Global topic assignments
│   │   ├── topic_info.csv          # Topic names, keywords, counts
│   │   ├── topics_over_time.csv    # Built-in evolution tracking
│   │   ├── metadata.json           # Run metadata
│   │   ├── stats.json              # Pipeline summary stats
│   │   ├── blog_stats.json         # Blog-ready data (from 03_blog_stats.py)
│   │   ├── topic_distribution.png
│   │   ├── topic_words.png
│   │   └── topic_umap.png
│   └── models/                     # Cached embedding model (on HPC)
├── pyproject.toml                  # Python dependencies
└── uv.lock                        # Dependency lock file
```

## Usage

### 1. Prepare the data

Aggregate pages per website-year with paragraph deduplication:

```bash
sbatch bert_topic_websites_lux/script/00_prepare_data.sh
```

### 2. Generate embeddings (GPU array job)

Encode website text with BAAI/bge-m3 (12 parallel GPU jobs):

```bash
sbatch bert_topic_websites_lux/script/01_embed.sh
```

### 3. Run global BERTopic

Run BERTopic on all years at once with `topics_over_time()`:

```bash
sbatch bert_topic_websites_lux/script/02_bert_topic.sh
```

### 4. Generate blog statistics

Aggregate topics into sectors and compute shares over time:

```bash
sbatch bert_topic_websites_lux/script/03_blog_stats.sh
```

Then copy `output/blog_stats.json` to the blog directory as `stats.json`.

## Data

- **~81,000** website-years (2013-2024)
- **12 years** of topic evolution with consistent topic IDs
- Paragraph-level deduplication removes boilerplate text

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | `BAAI/bge-m3` (1024-dim, 8192-token context) |
| Max embed length | 30,000 characters |
| Max c-TF-IDF length | 50,000 characters |
| Min topic size | 50 |
| UMAP dimensions | 5 |
| UMAP neighbors | 15 |
| Stopwords | EN + FR + DE + PT + NL + Luxembourgish |

## Author

**Julio Garbers**
[julio.garbers@liser.lu](mailto:julio.garbers@liser.lu)
