### Luxembourg Linguistic Shift Analysis (2013-2024)
### =================================================
### Analyzes language distribution across .lu websites over time using CommonCrawl data.
### 
### Author: Julio Garbers with contributions from Claude
### Date: June 2024

import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick    # type: ignore

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/project/home/p201125/firm_websites/data/clean/luxembourg")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Minimum confidence threshold for language detection (optional filter)
MIN_CONFIDENCE = 0.5

# Languages to track explicitly (others grouped as "Other")
MAIN_LANGUAGES = ["fr", "de", "en", "lb", "pt", "nl", "it"]

# Color palette for languages (Luxembourg-appropriate choices)
LANGUAGE_COLORS = {
    "fr": "#0055A4",      # French - blue
    "de": "#DD0000",      # German - red  
    "en": "#2E8B57",      # English - sea green
    "lb": "#00A1DE",      # Luxembourgish - light blue (Lux flag)
    "pt": "#006600",      # Portuguese - green
    "nl": "#FF6600",      # Dutch - orange
    "it": "#009246",      # Italian - green
    "Other": "#888888",   # Other - gray
}

LANGUAGE_LABELS = {
    "fr": "French",
    "de": "German",
    "en": "English",
    "lb": "Luxembourgish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "it": "Italian",
    "Other": "Other",
}

# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_dir: Path) -> pl.LazyFrame:
    """Load all parquet files from directory as a lazy frame."""
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Scan all parquet files lazily
    lf = pl.scan_parquet(parquet_files)
    
    return lf


# =============================================================================
# Data Processing
# =============================================================================

def process_data(lf: pl.LazyFrame, min_confidence: float = 0.5) -> pl.DataFrame:
    """
    Process the data:
    1. Filter to .lu domains only
    2. Filter by language confidence threshold (>= min_confidence)
    3. Aggregate to website_url + year level (mode of language)
    4. Compute language shares by year
    """
    
    # Step 1: Filter to .lu domains and apply confidence threshold
    lf_filtered = (
        lf
        .filter(pl.col("website_url").str.ends_with(".lu"))
        .filter(pl.col("confidence_fasttext") >= min_confidence)
    )

    # Step 2: Aggregate at website_url + year level
    # For each (website_url, year), find the mode of language_fasttext
    # If tie, take the first one (alphabetically due to sort)
    df_page_level = (
        lf_filtered
        .group_by(["website_url", "year"])
        .agg([
            # Count occurrences of each language, then get the most frequent
            pl.col("language_fasttext").mode().first().alias("language"),
            pl.len().alias("n_observations"),
        ])
        .collect(engine="streaming")
    )

    print(f"Total unique website-year combinations: {len(df_page_level):,}")

    return df_page_level


def compute_language_shares(df: pl.DataFrame, main_languages: list) -> pl.DataFrame:
    """
    Compute language shares by year.
    Groups minor languages into 'Other'.
    """
    
    # Map languages not in main_languages to "Other"
    df_with_groups = df.with_columns(
        pl.when(pl.col("language").is_in(main_languages))
        .then(pl.col("language"))
        .otherwise(pl.lit("Other"))
        .alias("language_group")
    )
    
    # Count pages by year and language group
    counts = (
        df_with_groups
        .group_by(["year", "language_group"])
        .agg(pl.len().alias("n_websites"))
    )
    
    # Compute total websites per year
    totals = (
        counts
        .group_by("year")
        .agg(pl.col("n_websites").sum().alias("total_websites"))
    )
    
    # Join and compute shares
    shares = (
        counts
        .join(totals, on="year")
        .with_columns(
            (pl.col("n_websites") / pl.col("total_websites") * 100).alias("share_pct")
        )
        .sort(["year", "language_group"])
    )
    
    return shares


def pivot_for_plotting(shares: pl.DataFrame) -> pl.DataFrame:
    """Pivot the shares dataframe for easier plotting."""
    
    pivoted = (
        shares
        .pivot(
            on="language_group",
            index="year", 
            values="share_pct"
        )
        .sort("year")
        .fill_null(0)
    )
    
    return pivoted


# =============================================================================
# Visualization
# =============================================================================

def create_stacked_area_chart(
    pivoted: pl.DataFrame,
    main_languages: list,
    colors: dict,
    labels: dict,
    output_path: Path
):
    """Create a stacked area chart of language shares over time."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = pivoted["year"].to_numpy()
    
    # Determine which languages are actually in the data
    available_langs = [lang for lang in main_languages + ["Other"] 
                       if lang in pivoted.columns]
    
    # Stack the data
    data_stack = []
    color_list = []
    label_list = []
    
    for lang in available_langs:
        if lang in pivoted.columns:
            data_stack.append(pivoted[lang].to_numpy())
            color_list.append(colors.get(lang, "#888888"))
            label_list.append(labels.get(lang, lang))
    
    # Create stacked area plot
    ax.stackplot(years, data_stack, labels=label_list, colors=color_list, alpha=0.85)
    
    # Styling
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of Websites (%)", fontsize=12)
    ax.set_title("Language Distribution on Luxembourg (.lu) Websites\n2013-2024", 
                 fontsize=14, fontweight="bold")
    
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    # Set x-axis ticks to show all years
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    
    # Legend outside the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    print(f"Saved stacked area chart to {output_path}")


def create_line_chart(
    pivoted: pl.DataFrame,
    main_languages: list,
    colors: dict,
    labels: dict,
    output_path: Path
):
    """Create a line chart showing individual language trends."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = pivoted["year"].to_numpy()
    
    # Determine which languages are actually in the data
    available_langs = [lang for lang in main_languages + ["Other"] 
                       if lang in pivoted.columns]
    
    for lang in available_langs:
        if lang in pivoted.columns:
            values = pivoted[lang].to_numpy()
            ax.plot(years, values, 
                   label=labels.get(lang, lang),
                   color=colors.get(lang, "#888888"),
                   linewidth=2.5,
                   marker="o",
                   markersize=5)
    
    # Styling
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of Websites (%)", fontsize=12)
    ax.set_title("Language Trends on Luxembourg (.lu) Websites\n2013-2024", 
                 fontsize=14, fontweight="bold")
    
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    
    # Set x-axis ticks
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    
    # Legend
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    print(f"Saved line chart to {output_path}")


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_stats(df: pl.DataFrame, shares: pl.DataFrame):
    """Print summary statistics about the data."""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Total websites by year
    websites_by_year = (
        df
        .group_by("year")
        .agg(pl.len().alias("n_websites"))
        .sort("year")
    )
    
    print("\nWebsites per year:")
    print(websites_by_year)
    
    # Language distribution overall
    print("\nOverall language distribution:")
    overall = (
        df
        .group_by("language")
        .agg(pl.len().alias("n_websites"))
        .sort("n_websites", descending=True)
        .head(15)
    )
    print(overall)
    
    # First and last year comparison for main languages
    print("\nLanguage share changes (first vs last year):")
    
    first_year = shares.filter(pl.col("year") == shares["year"].min())
    last_year = shares.filter(pl.col("year") == shares["year"].max())
    
    comparison = (
        first_year
        .select(["language_group", "share_pct"])
        .rename({"share_pct": "first_year_pct"})
        .join(
            last_year.select(["language_group", "share_pct"]).rename({"share_pct": "last_year_pct"}),
            on="language_group",
            how="full"
        )
        .fill_null(0)
        .with_columns(
            (pl.col("last_year_pct") - pl.col("first_year_pct")).alias("change_pp")
        )
        .sort("change_pp", descending=True)
    )
    
    print(comparison)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Luxembourg Linguistic Shift Analysis")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    lf = load_data(DATA_DIR)
    
    # Process data
    print("\n2. Processing data...")
    df_website_level = process_data(lf, min_confidence=MIN_CONFIDENCE)
    
    # Compute shares
    print("\n3. Computing language shares...")
    shares = compute_language_shares(df_website_level, MAIN_LANGUAGES)
    pivoted = pivot_for_plotting(shares)
    
    print("\nLanguage shares by year:")
    print(pivoted)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    create_stacked_area_chart(
        pivoted, MAIN_LANGUAGES, LANGUAGE_COLORS, LANGUAGE_LABELS,
        OUTPUT_DIR / "lux_languages_stacked_area.png"
    )
    
    create_line_chart(
        pivoted, MAIN_LANGUAGES, LANGUAGE_COLORS, LANGUAGE_LABELS,
        OUTPUT_DIR / "lux_languages_line.png"
    )
    
    # Print summary
    print_summary_stats(df_website_level, shares)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()