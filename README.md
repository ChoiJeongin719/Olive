# Olive – App Store Review Crawling & Analysis

Crawl user reviews from **Google Play** and the **Apple App Store**, then automatically analyse them to surface sentiment trends and pain points.

---

## Features

| Module | Description |
|---|---|
| `crawler/google_play_crawler.py` | Fetches reviews from Google Play via `google-play-scraper` |
| `crawler/app_store_crawler.py` | Fetches reviews from the Apple App Store via `app-store-scraper` |
| `analysis/sentiment_analysis.py` | VADER-based sentiment scoring & distribution charts |
| `analysis/pain_point_extractor.py` | Keyword frequency analysis of low-rating reviews + word cloud |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Crawl Google Play reviews and run analysis

```bash
python main.py --platform google_play --app-id com.example.app --count 500
```

### Crawl Apple App Store reviews and run analysis

```bash
python main.py --platform app_store --app-id 1234567890 --app-name "My App" --count 500
```

### Analyse an existing CSV file (skip crawling)

```bash
python main.py --csv data/google_play_reviews.csv
```

### All options

```
--platform        google_play | app_store
--app-id          App package name (Google Play) or numeric ID (App Store)
--app-name        Human-readable app name (App Store only)
--count           Number of reviews to crawl (default: 500)
--lang            Review language for Google Play (default: en)
--country         Country code (default: us)
--rating-threshold  Star rating treated as negative, ≤ N (default: 2)
--top-n           Top N pain-point keywords to extract (default: 20)
--csv             Path to an existing CSV to analyse without crawling
```

---

## Output files (saved to `data/`)

| File | Description |
|---|---|
| `google_play_reviews.csv` / `app_store_reviews.csv` | Raw crawled reviews |
| `sentiment_distribution.png` | Bar chart of positive / neutral / negative counts |
| `pain_points.png` | Horizontal bar chart of top pain-point keywords |
| `pain_points_wordcloud.png` | Word cloud of pain-point keywords |

---

## Project structure

```
Olive/
├── main.py                          # CLI entry point
├── requirements.txt
├── crawler/
│   ├── google_play_crawler.py
│   └── app_store_crawler.py
├── analysis/
│   ├── sentiment_analysis.py
│   └── pain_point_extractor.py
├── data/                            # Output CSVs & charts (git-ignored)
└── tests/
    └── test_analysis.py
```
