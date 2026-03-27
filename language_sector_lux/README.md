# Lost in Translation: Digital Language Gaps in Luxembourg

Cross-referencing language availability with website sectors to analyze whether essential services are accessible in the languages Luxembourg's diverse population speaks.

## Overview

This project joins two previous analyses — [language detection](https://github.com/julio-garbers/blog/tree/main/website_languages_lux) and [topic modeling](https://github.com/julio-garbers/blog/tree/main/bert_topic_websites_lux) — to answer a policy-relevant question: **are essential online services (government, healthcare, childcare) as linguistically accessible as market-driven sectors (real estate, restaurants, retail)?**

Luxembourg has the EU's highest share of foreign-born residents (48%), yet the previous language analysis showed that only 2.4% of websites offer Portuguese — despite Portuguese speakers making up 14.5% of the population. By crossing sector classifications with language data, this analysis reveals *where* those gaps are concentrated.

**Key design choices:**
- **Inner join** of language and topic datasets on `(website_url, year)` — same CommonCrawl sample underlies both
- **Sector-level aggregation** using the 15-category taxonomy from the topic analysis
- **Essential vs. market comparison** to test whether market forces drive multilingualism more than public services
- **Portuguese gap analysis** comparing web availability to population demographics

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   00_language_by_sector.py                       │
│                    (Single job, CPU)                             │
│                                                                 │
│  1. Load language flags (regex + LLM + FastText fallback)       │
│  2. Load BERTopic assignments + sector mapping                  │
│  3. Inner join on (website_url, year)                           │
│  4. Compute language availability per sector per year            │
│  5. Essential vs market comparison                              │
│  6. Portuguese gap analysis                                     │
│  Output: stats.json                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
language_sector_lux/
├── README.md
├── script/
│   ├── 00_language_by_sector.py   # Join + aggregate analysis
│   ├── 00_language_by_sector.sh   # SLURM job script (CPU)
│   └── slurm/                     # SLURM output logs
└── output/
    └── stats.json                 # Blog-ready statistics
```

## Usage

This script depends on outputs from both upstream pipelines:
- `website_languages_lux/data/lux_sample_with_languages.parquet`
- `bert_topic_websites_lux/output/website_topics.parquet`
- `bert_topic_websites_lux/sector_mapping.json`

Run the analysis:

```bash
sbatch language_sector_lux/script/00_language_by_sector.sh
```

Then copy `output/stats.json` to the blog directory.

## Data

- **Input:** ~81,000 website-years with both language flags and topic assignments
- **Analysis window:** 2016-2024 (9 years, matching topic analysis)
- **Sectors:** 15 categories from manual topic-to-sector mapping
- **Languages:** French, German, English, Luxembourgish, Portuguese, Dutch

## Author

**Julio Garbers**
[julio.garbers@liser.lu](mailto:julio.garbers@liser.lu)
