"""Olive – App Store Review Crawling & Analysis.

Usage examples
--------------
Crawl Google Play reviews and run full analysis::

    python main.py --platform google_play --app-id com.example.app --count 500

Crawl App Store reviews::

    python main.py --platform app_store --app-id 1234567890 --app-name "My App" --count 500

Analyse an existing CSV file (skip crawling)::

    python main.py --csv data/google_play_reviews.csv
"""

import argparse
import os
import sys

import pandas as pd

from analysis.sentiment_analysis import SentimentAnalyzer
from analysis.pain_point_extractor import PainPointExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crawl_google_play(app_id: str, count: int, lang: str, country: str) -> pd.DataFrame:
    from crawler.google_play_crawler import GooglePlayCrawler

    crawler = GooglePlayCrawler(app_id, lang=lang, country=country)
    df = crawler.crawl(count=count)
    out = f"data/google_play_reviews.csv"
    os.makedirs("data", exist_ok=True)
    crawler.save(df, out)
    return df


def _crawl_app_store(app_id: str, app_name: str, count: int, country: str) -> pd.DataFrame:
    from crawler.app_store_crawler import AppStoreCrawler

    crawler = AppStoreCrawler(app_name=app_name, app_id=app_id, country=country)
    df = crawler.crawl(count=count)
    out = f"data/app_store_reviews.csv"
    os.makedirs("data", exist_ok=True)
    crawler.save(df, out)
    return df


def _run_analysis(df: pd.DataFrame, rating_threshold: int, top_n: int) -> None:
    os.makedirs("data", exist_ok=True)

    # --- Sentiment analysis ---
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze(df)

    print("\n=== Sentiment Summary ===")
    print(analyzer.summary(df).to_string(index=False))
    analyzer.plot_distribution(df)

    # --- Pain point extraction ---
    extractor = PainPointExtractor(rating_threshold=rating_threshold, top_n=top_n)
    extractor.summarize(df)

    keywords_df = extractor.extract(df)
    if not keywords_df.empty:
        extractor.plot_keywords(keywords_df)
        extractor.plot_wordcloud(keywords_df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawl app store reviews and analyse user pain points.",
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        metavar="PATH",
        help="Skip crawling and analyse an existing CSV file.",
    )
    source.add_argument(
        "--platform",
        choices=["google_play", "app_store"],
        help="App store platform to crawl.",
    )

    p.add_argument("--app-id", help="App package name (Google Play) or numeric ID (App Store).")
    p.add_argument("--app-name", default="", help="App name (required for App Store).")
    p.add_argument("--count", type=int, default=500, help="Number of reviews to crawl.")
    p.add_argument("--lang", default="en", help="Review language (Google Play only).")
    p.add_argument("--country", default="us", help="Country code.")
    p.add_argument(
        "--rating-threshold",
        type=int,
        default=2,
        help="Star rating at or below which a review is treated as negative.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top pain-point keywords to extract.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.csv:
        print(f"Loading reviews from {args.csv} …")
        df = pd.read_csv(args.csv)
    else:
        if not args.app_id:
            parser.error("--app-id is required when using --platform.")
        if args.platform == "google_play":
            df = _crawl_google_play(args.app_id, args.count, args.lang, args.country)
        else:
            if not args.app_name:
                parser.error("--app-name is required for App Store crawling.")
            df = _crawl_app_store(args.app_id, args.app_name, args.count, args.country)

    _run_analysis(df, rating_threshold=args.rating_threshold, top_n=args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
