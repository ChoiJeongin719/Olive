"""Sentiment analysis for app store reviews using VADER (vaderSentiment)."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Score review sentiment with vaderSentiment and summarise results."""

    # Compound score thresholds used to assign a sentiment label
    POS_THRESHOLD = 0.05
    NEG_THRESHOLD = -0.05

    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
        """Add sentiment columns to *df* and return the updated DataFrame.

        New columns added:
            compound  – VADER compound score  (-1.0 … +1.0)
            pos       – positive sub-score
            neu       – neutral sub-score
            neg       – negative sub-score
            sentiment – 'positive' | 'neutral' | 'negative'

        Args:
            df:       Reviews DataFrame (must contain *text_col*).
            text_col: Name of the column that holds the review text.

        Returns:
            Copy of *df* with the new sentiment columns appended.
        """
        df = df.copy()
        scores = df[text_col].fillna("").apply(self._sia.polarity_scores)
        df["compound"] = scores.apply(lambda s: s["compound"])
        df["pos"] = scores.apply(lambda s: s["pos"])
        df["neu"] = scores.apply(lambda s: s["neu"])
        df["neg"] = scores.apply(lambda s: s["neg"])
        df["sentiment"] = df["compound"].apply(self._label)
        return df

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a sentiment distribution summary table.

        Args:
            df: DataFrame that already contains a 'sentiment' column
                (i.e. the output of :meth:`analyze`).

        Returns:
            DataFrame with columns: sentiment, count, percentage.
        """
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        counts["percentage"] = (counts["count"] / len(df) * 100).round(2)
        return counts

    def plot_distribution(
        self,
        df: pd.DataFrame,
        save_path: str = "data/sentiment_distribution.png",
    ) -> None:
        """Save a bar chart of sentiment distribution.

        Args:
            df:        DataFrame with a 'sentiment' column.
            save_path: File path for the output PNG.
        """
        summary = self.summary(df)
        palette = {"positive": "#4CAF50", "neutral": "#9E9E9E", "negative": "#F44336"}

        fig, ax = plt.subplots(figsize=(7, 4))
        summary["color"] = summary["sentiment"].map(lambda s: palette.get(s, "#90A4AE"))
        sns.barplot(
            data=summary,
            x="sentiment",
            y="count",
            hue="sentiment",
            palette=dict(zip(summary["sentiment"], summary["color"])),
            legend=False,
            ax=ax,
        )
        for bar, pct in zip(ax.patches, summary["percentage"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved sentiment chart → {save_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _label(self, compound: float) -> str:
        if compound >= self.POS_THRESHOLD:
            return "positive"
        if compound <= self.NEG_THRESHOLD:
            return "negative"
        return "neutral"
