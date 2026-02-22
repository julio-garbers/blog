# What Does Luxembourg's Web Talk About?

Topic modeling of Luxembourg (.lu) websites from 2016 to 2024 using BERTopic and CommonCrawl archives.

📊 **[View the interactive visualization](https://juliogarbers.com/posts/bert_topic_websites_lux/)**

## Overview

This project applies unsupervised topic modeling to discover what themes Luxembourg websites cover and how the thematic landscape evolves over nine years. Using the same CommonCrawl sample as the [language analysis](https://github.com/julio-garbers/blog/tree/main/website_languages_lux), we run BERTopic on individual web pages independently for each year, then aggregate results back to the website-year level.

Unlike traditional approaches that assign each website a single topic, this **page-level analysis** allows a website to span multiple topics through its different pages — providing a richer, more accurate picture of each site's content.

**Key findings:**
- Restaurants dominated Luxembourg's web from 2017–2020, peaking at 574 websites in 2019
- Real estate rose from a niche topic in 2016 to the #1 topic by 2024
- Healthcare and wellness websites emerged as a major category starting in 2022
- Financial services, automotive, and HVAC/construction maintain stable presence throughout
- Cookie consent and privacy regulation text became visible topics from 2021 onward
- Municipal government websites ("commune", "administration communale") appear consistently across all years

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      00_prepare_data.py                         │
│   Extract individual pages per website-year from language sample │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       01_bert_topic.py                          │
│                    (SLURM array job: 1 per year)                │
│                                                                 │
│  1. Sentence embeddings per page (multilingual-MiniLM-L12-v2)  │
│  2. UMAP dimensionality reduction (384D → 5D)                  │
│  3. HDBSCAN clustering (min cluster = 10)                      │
│  4. c-TF-IDF topic representation                              │
│  5. Aggregate page topics → website-level distributions         │
│  6. Visualizations (distribution, UMAP, similarity)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     02_combine_results.py                       │
│    Aggregate yearly results, find recurring topics, stats.json  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
blog/
├── bert_topic_websites_lux/
│   ├── script/
│   │   ├── 00_prepare_data.py      # Extract individual pages per website-year
│   │   ├── 00_prepare_data.sh      # SLURM job script
│   │   ├── 01_bert_topic.py        # BERTopic analysis (per year, page-level)
│   │   ├── 01_bert_topic.sh        # SLURM array job (0-11 = 2013-2024)
│   │   ├── 02_combine_results.py   # Combine results across years
│   │   ├── 02_combine_results.sh   # SLURM job script
│   │   └── slurm/                  # SLURM output logs
│   ├── data/
│   │   ├── metadata.json           # Year/page counts (pipeline artifact)
│   │   └── yearly/                 # Per-year parquet files (on HPC)
│   │       └── pages_{YEAR}.parquet  # Individual pages
│   ├── output/
│   │   ├── {year}/                 # Per-year results (2013-2024)
│   │   │   ├── page_topics.parquet   # Page-level topic assignments
│   │   │   ├── website_topics.parquet # Website-level aggregation
│   │   │   ├── topic_summary.csv   # Topic names, keywords, page/website counts
│   │   │   ├── topic_distribution.png
│   │   │   ├── topic_words.png
│   │   │   ├── topic_umap.png
│   │   │   ├── topic_similarity.png
│   │   │   ├── embeddings.npy
│   │   │   └── metadata.json
│   │   └── stats.json              # Combined stats for blog visualization
│   └── models/                     # Cached embedding model (on HPC)
├── pyproject.toml                  # Python dependencies
└── uv.lock                        # Dependency lock file
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Access to CommonCrawl data (via previous language analysis sample)
- HPC cluster with SLURM

### Dependencies

Main packages:
- `bertopic` — Topic modeling (UMAP + HDBSCAN + c-TF-IDF)
- `sentence-transformers` — Multilingual sentence embeddings
- `polars` — Fast DataFrame operations

Install dependencies:
```bash
uv sync
```

## Usage

### 1. Prepare the data

Extract individual pages from the language analysis sample:

```bash
sbatch bert_topic_websites_lux/script/00_prepare_data.sh
```

### 2. Run BERTopic (array job)

Run page-level topic modeling independently for each year (12 parallel jobs):

```bash
sbatch bert_topic_websites_lux/script/01_bert_topic.sh
```

Each job processes one year (array index 0 = 2013, ..., 11 = 2024). Topics are discovered at the page level, then aggregated back to website-year level.

### 3. Combine results

Aggregate all yearly results into a single stats.json:

```bash
sbatch bert_topic_websites_lux/script/02_combine_results.sh
```

## Data

The analysis covers:
- **75,904** website-years analyzed (2016–2024, after dropping 2013–2015)
- **9 years** of topic evolution
- Topics discovered at the **page level**, then aggregated per website
- Websites can span **multiple topics** through their different pages

### BERTopic Configuration

| Parameter | Value |
|-----------|-------|
| Analysis unit | Individual web pages |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Max text per page | 10,000 characters |
| Min topic size | 10 |
| UMAP dimensions | 5 |
| UMAP neighbors | 15 |
| N-gram range | (1, 2) |
| Stopwords | EN + FR + DE + PT + NL + Luxembourgish |

## Citation

```bibtex
@misc{garbers2026topics,
  author = {Garbers, Julio},
  title = {What Does Luxembourg's Web Talk About?},
  url = {https://github.com/julio-garbers/blog/tree/main/bert_topic_websites_lux},
  year = {2026}
}
```

## License

MIT

## Author

**Julio Garbers**
[julio.garbers@liser.lu](mailto:julio.garbers@liser.lu)
