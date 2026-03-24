"""Pain point extractor for app store reviews.

Identifies frequently mentioned negative themes and keywords to surface the
main user experience issues reported in low-rating reviews.
"""

import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

_STOP_WORDS = set(stopwords.words("english"))


def _clean_text(text: str) -> str:
    """Lowercase, remove punctuation, and strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* and remove stop words and single-character tokens."""
    tokens = word_tokenize(_clean_text(text))
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class PainPointExtractor:
    """Extract pain points from negative / low-rating reviews."""

    def __init__(self, rating_threshold: int = 2, top_n: int = 20):
        """
        Args:
            rating_threshold: Reviews with a star rating *at or below* this
                              value are treated as negative.
            top_n:            Number of top keywords to surface.
        """
        self.rating_threshold = rating_threshold
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        score_col: str = "score",
    ) -> pd.DataFrame:
        """Return a DataFrame of the top pain-point keywords.

        Args:
            df:        Reviews DataFrame.
            text_col:  Name of the review-text column.
            score_col: Name of the star-rating column (1–5).

        Returns:
            DataFrame with columns: keyword, count, frequency.
        """
        negative_df = df[df[score_col] <= self.rating_threshold].copy()
        if negative_df.empty:
            return pd.DataFrame(columns=["keyword", "count", "frequency"])

        all_tokens: list[str] = []
        for text in negative_df[text_col].fillna(""):
            all_tokens.extend(_tokenize(text))

        counts = Counter(all_tokens).most_common(self.top_n)
        result = pd.DataFrame(counts, columns=["keyword", "count"])
        result["frequency"] = (result["count"] / len(negative_df)).round(4)
        return result

    def plot_keywords(
        self,
        keywords_df: pd.DataFrame,
        save_path: str = "data/pain_points.png",
    ) -> None:
        """Save a horizontal bar chart of top pain-point keywords.

        Args:
            keywords_df: Output of :meth:`extract`.
            save_path:   File path for the output PNG.
        """
        if keywords_df.empty:
            print("No pain-point data to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, max(4, len(keywords_df) * 0.4)))
        keywords_df_sorted = keywords_df.sort_values("count")
        ax.barh(keywords_df_sorted["keyword"], keywords_df_sorted["count"], color="#F44336")
        ax.set_title("Top Pain-Point Keywords (Low-Rating Reviews)")
        ax.set_xlabel("Occurrences")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved pain-point chart → {save_path}")

    def plot_wordcloud(
        self,
        keywords_df: pd.DataFrame,
        save_path: str = "data/pain_points_wordcloud.png",
    ) -> None:
        """Save a word cloud of pain-point keywords.

        Args:
            keywords_df: Output of :meth:`extract`.
            save_path:   File path for the output PNG.
        """
        if keywords_df.empty:
            print("No pain-point data to plot.")
            return

        freq_dict = dict(zip(keywords_df["keyword"], keywords_df["count"]))
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Reds",
            max_words=self.top_n,
        ).generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Pain-Point Word Cloud")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved word cloud → {save_path}")

    def summarize(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        score_col: str = "score",
    ) -> None:
        """Print a plain-text summary of the top pain points.

        Args:
            df:        Reviews DataFrame.
            text_col:  Name of the review-text column.
            score_col: Name of the star-rating column.
        """
        keywords_df = self.extract(df, text_col=text_col, score_col=score_col)
        if keywords_df.empty:
            print("No negative reviews found.")
            return

        total_neg = int((df[score_col] <= self.rating_threshold).sum())
        print(
            f"\n=== Pain Point Summary ===\n"
            f"Negative reviews (≤{self.rating_threshold}★): {total_neg} / {len(df)}\n"
            f"Top {self.top_n} pain-point keywords:\n"
        )
        for _, row in keywords_df.iterrows():
            print(f"  {row['keyword']:20s}  {row['count']:>5}  ({row['frequency']:.1%})")
