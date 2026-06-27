# The Consent Illusion — Who Really Runs Luxembourg's Web

Measuring the hidden infrastructure of Luxembourg (.lu) websites from 2013 to
2024 using CommonCrawl archives: the third-party trackers they embed, the
cookie-consent banners they show, and which companies (and countries) the
local web silently depends on.

## Overview

The first three posts in this series asked what Luxembourg's web *says* — its
[languages](https://github.com/julio-garbers/blog/tree/main/website_languages_lux),
its [topics](https://github.com/julio-garbers/blog/tree/main/bert_topic_websites_lux),
and [who gets served in which language](https://github.com/julio-garbers/blog/tree/main/language_sector_lux).
This post asks how the web is *built* and who controls it.

For every website-year we parse the raw HTML and record:

- **Third-party requests** — every external host the page loads resources from
  (`<script>`, `<img>`, `<iframe>`, `<link>`, `<source>` and absolute URLs in
  *executable* inline scripts), reduced to registrable domains different from the
  site's own. JSON-LD / structured-data `<script>` blocks and spec/profile
  namespaces (schema.org, gmpg.org, …) are excluded — they are metadata, not
  loaded resources, and would otherwise inflate the per-site domain count and the
  unidentified long tail.
- **Cookie-consent banners** — whether a Consent Management Platform (Cookiebot,
  OneTrust, Didomi, Usercentrics, …) or generic cookie-banner markup is present.
- **HTTPS** — whether the site is served over a secure connection.

Each third party is mapped to its owning entity, owner country (US / EU / LU /
Other), and category, letting us separate **trackers** (analytics, advertising,
social, tag managers, marketing) from **functional infrastructure** (CDNs,
fonts, payments, maps, video, support).

**The story:** after GDPR (2018) cookie banners went from rare to near-universal
— but did the actual amount of tracking fall, or did we just bolt a banner on
top of it? And when you open a `.lu` site — even your commune's or your doctor's
— whose servers are quietly watching, and where do they sit?

**Key design choices:**
- **Fully offline & reproducible** — no live WHOIS/DNS/geolocation. Ownership and
  country come from a curated, version-controlled entity map shipped in the repo.
- **Same universe as the other posts** — inner-joined to the language sample, so
  results are directly comparable across the series.
- **Sector reuse** — joins the BERTopic `sector_mapping.json` to ask which
  *sectors* track you most, with the same essential-vs-market cut as post #3.
- **Conservative tracking counts** — only positively identified domains count as
  trackers; the unknown long tail is never assumed to be tracking.

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 00_extract_third_parties.py                      │
│              (SLURM array job: 1 per year, CPU)                  │
│                                                                  │
│  Parse raw HTML -> per website-year:                            │
│   • set of third-party registrable domains                      │
│   • has_cmp (consent banner) / has_cookie_text / any_https      │
│  Inner-join to the language sample universe.                    │
│  Output: data/yearly/thirdparty_{YEAR}.parquet                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        01_classify.py                            │
│                          (CPU)                                   │
│                                                                  │
│  Explode third-party sets and join the curated entity map.      │
│  .lu third parties -> local/sovereign automatically.            │
│  Output: data/classified_requests.parquet (long)                │
│          data/site_summary.parquet (per website-year)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       02_blog_stats.py                           │
│                          (CPU)                                   │
│                                                                  │
│  Join BERTopic sectors. Aggregate into stats.json:              │
│   • over_time      (consent illusion: banners vs trackers)      │
│   • sovereignty    (owner-country mix, Google/Meta reach)       │
│   • top_entities   (most-embedded companies)                    │
│   • by_sector + essential_vs_market                             │
│  Output: output/stats.json  (copy to the blog directory)        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
web_trackers_lux/
├── reference/
│   ├── entity_map.json        # domain -> entity / country / category / tracker
│   └── cmp_signatures.json    # consent-manager fingerprints
├── script/
│   ├── 00_extract_third_parties.py   # parse raw HTML (array job, CPU)
│   ├── 00_extract_third_parties.sh
│   ├── 01_classify.py                # join entity map, per-site metrics
│   ├── 01_classify.sh
│   ├── 02_blog_stats.py              # aggregate to stats.json
│   ├── 02_blog_stats.sh
│   └── slurm/                        # SLURM output logs
├── data/                      # intermediate parquet (on HPC, not tracked)
│   ├── yearly/
│   ├── classified_requests.parquet
│   └── site_summary.parquet
└── output/
    └── stats.json             # blog-ready data
```

## Usage

```bash
# 1. Extract third parties from raw HTML (12 parallel CPU jobs, one per year)
sbatch web_trackers_lux/script/00_extract_third_parties.sh

# 2. Classify against the entity map
sbatch web_trackers_lux/script/01_classify.sh

# 3. Generate blog statistics
sbatch web_trackers_lux/script/02_blog_stats.sh
```

Then copy `output/stats.json` to the blog post directory as `stats.json`.

## Data

- Same **~80k website-years** (2013–2024) as the rest of the series
- Raw HTML source: `/project/home/p201125/firm_websites/data/raw/luxembourg/`
- Curated entity map covers ~112 dominant third-party domains; the long tail
  is reported as `country = "Other"`. The identified-entity coverage rate is
  printed by `01_classify.py` as a transparency check.

### Configuration

| Parameter | Value |
|-----------|-------|
| Years | 2013–2024 |
| First-party resolution | registrable domain (eTLD+1) via `tldextract`, offline snapshot |
| Tracker categories | advertising, analytics, social, tag-manager, marketing |
| Country buckets | US, EU, LU, Other |
| Max third parties / site-year | 200 |

## Author

**Julio Garbers**
[julio.garbers@liser.lu](mailto:julio.garbers@liser.lu)
