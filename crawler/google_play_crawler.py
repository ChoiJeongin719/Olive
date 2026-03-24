"""Google Play Store review crawler using google-play-scraper."""

import time
import pandas as pd
from tqdm import tqdm
from google_play_scraper import reviews, Sort


class GooglePlayCrawler:
    """Crawl user reviews from the Google Play Store."""

    def __init__(self, app_id: str, lang: str = "en", country: str = "us"):
        """
        Args:
            app_id: Google Play app package name (e.g. 'com.example.app').
            lang:    Language code for reviews (default 'en').
            country: Country code (default 'us').
        """
        self.app_id = app_id
        self.lang = lang
        self.country = country

    def crawl(
        self,
        count: int = 1000,
        sort: Sort = Sort.NEWEST,
        sleep_seconds: float = 0.5,
    ) -> pd.DataFrame:
        """Fetch reviews and return them as a DataFrame.

        Args:
            count:         Total number of reviews to fetch.
            sort:          Sort order (Sort.NEWEST or Sort.MOST_RELEVANT).
            sleep_seconds: Seconds to sleep between paginated requests.

        Returns:
            DataFrame with columns: reviewId, userName, score, at, content.
        """
        all_reviews = []
        continuation_token = None

        with tqdm(total=count, desc="Google Play") as pbar:
            while len(all_reviews) < count:
                batch_size = min(200, count - len(all_reviews))
                result, continuation_token = reviews(
                    self.app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=sort,
                    count=batch_size,
                    continuation_token=continuation_token,
                )
                if not result:
                    break
                all_reviews.extend(result)
                pbar.update(len(result))
                if continuation_token is None:
                    break
                time.sleep(sleep_seconds)

        df = pd.DataFrame(all_reviews)
        df = df[["reviewId", "userName", "score", "at", "content"]].copy()
        df["source"] = "google_play"
        return df

    def save(
        self,
        df: pd.DataFrame,
        path: str = "data/google_play_reviews.csv",
    ) -> None:
        """Save the crawled reviews to a CSV file."""
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} reviews → {path}")
