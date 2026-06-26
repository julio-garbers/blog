# Project Guidelines

## Workflow

- Scripts are run on a supercomputer (MeluXina) via SLURM - do not run scripts locally
- User will push code to supercomputer, run with `sbatch`, and pull results back
- When results are pulled, user will let you know so you can review outputs
- Do not attempt to run Python scripts or lint/check them locally

## SLURM Partitions

Choose the appropriate partition based on script requirements:

| Partition | Use Case |
|-----------|----------|
| `cpu` | Default choice. Sufficient for most scripts with early filtering, few columns, streaming |
| `largemem` | Large joins, loading multiple full tables, high memory operations |
| `gpu` | GPU-accelerated workloads (e.g., LLM inference with vLLM) |

**Rule of thumb:** Start with `cpu`, use `largemem` only when joining large unfiltered datasets

## Script Organization

- Each Python script has a matching `.sh` SLURM submission file
- SLURM output logs go to `script/slurm/` subdirectory
- Use section separators: `# ===...===` with section titles

## Naming Conventions

- Input directories in CAPS ending with `_DIR` (e.g., `DATA_DIR`)
- Input files in CAPS ending with `_FILE` (e.g., `LANGUAGE_SAMPLE_FILE`)
- Output directories also in CAPS with `_DIR`

## Data Paths

- **Website data:** `/project/home/p201125/firm_websites/data/clean/luxembourg/`
- **Blog project space:** `/project/home/p200812/blog/`
  - `bert_topic_websites_lux/` - This project (topic analysis)
  - `website_languages_lux/` - Previous project (language analysis)
- **Virtual environment:** `/project/home/p200812/blog/.venv/`

## Python Style

- All print statements must have `flush=True`
- Always write parquet with `compression="zstd", compression_level=10`
- Use `collect(engine="streaming")` for large Polars operations
- All imports must be at the top of the script, never in the middle
- Use `matplotlib.use("Agg")` before importing pyplot (non-interactive backend)

## Blog Posts

- Website built with Quarto, hosted at juliogarbers.github.io
- Blog posts go in `posts/<slug>/index.qmd` with a `stats.json` data file
- Interactive charts use Plotly via JavaScript in the .qmd file
- Previous post: "The Linguistic Web of Luxembourg" (language analysis)
- Current post: Topic modeling of Luxembourg websites (BERTopic)

## Current Pipeline: bert_topic_websites_lux

1. `00_prepare_data.py` - Aggregate text by website-year from language sample
2. `01_bert_topic.py` - Run BERTopic per year (SLURM array job, 0-11 = 2013-2024)
3. `02_combine_results.py` - Combine yearly results into stats.json
