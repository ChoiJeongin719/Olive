"""Apple App Store review crawler using app-store-scraper."""

import pandas as pd
from tqdm import tqdm
from app_store_scraper import AppStore


class AppStoreCrawler:
    """Crawl user reviews from the Apple App Store."""

    def __init__(self, app_name: str, app_id: str, country: str = "us"):
        """
        Args:
            app_name: Human-readable app name (used in the scraper).
            app_id:   Numeric App Store app ID (e.g. '1234567890').
            country:  Two-letter country code (default 'us').
        """
        self.app_name = app_name
        self.app_id = app_id
        self.country = country

    def crawl(self, count: int = 1000) -> pd.DataFrame:
        """Fetch reviews and return them as a DataFrame.

        Args:
            count: Total number of reviews to fetch.

        Returns:
            DataFrame with columns: review, rating, date, userName, title.
        """
        app = AppStore(
            country=self.country,
            app_name=self.app_name,
            app_id=self.app_id,
        )

        with tqdm(total=count, desc="App Store") as pbar:
            app.review(how_many=count)
            pbar.update(count)

        df = pd.DataFrame(app.reviews)
        rename_map = {
            "review": "content",
            "rating": "score",
            "date": "at",
            "userName": "userName",
            "title": "title",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        keep = [c for c in ["content", "score", "at", "userName", "title"] if c in df.columns]
        df = df[keep].copy()
        df["source"] = "app_store"
        return df

    def save(
        self,
        df: pd.DataFrame,
        path: str = "data/app_store_reviews.csv",
    ) -> None:
        """Save the crawled reviews to a CSV file."""
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} reviews → {path}")
