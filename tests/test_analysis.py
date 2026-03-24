"""Unit tests for analysis modules (no external API calls required)."""

import pandas as pd
import pytest

from analysis.sentiment_analysis import SentimentAnalyzer
from analysis.pain_point_extractor import PainPointExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_reviews() -> pd.DataFrame:
    """Small synthetic review dataset."""
    return pd.DataFrame(
        {
            "content": [
                "I love this app, it is absolutely wonderful!",
                "Terrible experience. App crashes every time I open it.",
                "It is okay, nothing special.",
                "Great features but the login keeps failing.",
                "Worst app ever. Keeps freezing and crashing.",
                "Pretty good overall. Would recommend.",
                "Cannot login at all. Very frustrating bug.",
            ],
            "score": [5, 1, 3, 2, 1, 4, 2],
            "source": ["google_play"] * 7,
        }
    )


# ---------------------------------------------------------------------------
# SentimentAnalyzer tests
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def test_analyze_adds_expected_columns(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        for col in ("compound", "pos", "neu", "neg", "sentiment"):
            assert col in result.columns, f"Missing column: {col}"

    def test_sentiment_labels_are_valid(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        valid_labels = {"positive", "neutral", "negative"}
        assert set(result["sentiment"]).issubset(valid_labels)

    def test_positive_review_is_positive(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        positive_row = result[result["content"].str.startswith("I love")]
        assert positive_row["sentiment"].iloc[0] == "positive"

    def test_negative_review_is_negative(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        negative_row = result[result["content"].str.startswith("Terrible")]
        assert negative_row["sentiment"].iloc[0] == "negative"

    def test_compound_score_range(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        assert result["compound"].between(-1.0, 1.0).all()

    def test_summary_returns_dataframe(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        summary = analyzer.summary(result)
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == ["sentiment", "count", "percentage"]

    def test_summary_percentages_sum_to_100(self, sample_reviews):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(sample_reviews)
        summary = analyzer.summary(result)
        assert abs(summary["percentage"].sum() - 100.0) < 0.1

    def test_analyze_handles_empty_text(self):
        analyzer = SentimentAnalyzer()
        df = pd.DataFrame({"content": ["", None, "   "], "score": [3, 3, 3]})
        result = analyzer.analyze(df)
        assert len(result) == 3
        assert result["sentiment"].notna().all()


# ---------------------------------------------------------------------------
# PainPointExtractor tests
# ---------------------------------------------------------------------------

class TestPainPointExtractor:
    def test_extract_returns_dataframe(self, sample_reviews):
        extractor = PainPointExtractor(rating_threshold=2, top_n=10)
        result = extractor.extract(sample_reviews)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["keyword", "count", "frequency"]

    def test_extract_respects_rating_threshold(self, sample_reviews):
        extractor_strict = PainPointExtractor(rating_threshold=1, top_n=10)
        extractor_loose = PainPointExtractor(rating_threshold=3, top_n=10)
        result_strict = extractor_strict.extract(sample_reviews)
        result_loose = extractor_loose.extract(sample_reviews)
        # Looser threshold covers more reviews → generally more / equal keywords
        assert result_loose["count"].sum() >= result_strict["count"].sum()

    def test_extract_top_n_respected(self, sample_reviews):
        extractor = PainPointExtractor(rating_threshold=2, top_n=3)
        result = extractor.extract(sample_reviews)
        assert len(result) <= 3

    def test_extract_no_negative_reviews(self):
        df = pd.DataFrame(
            {"content": ["Great app!", "Excellent service."], "score": [5, 4]}
        )
        extractor = PainPointExtractor(rating_threshold=2)
        result = extractor.extract(df)
        assert result.empty

    def test_keywords_are_lowercase(self, sample_reviews):
        extractor = PainPointExtractor(rating_threshold=2, top_n=20)
        result = extractor.extract(sample_reviews)
        assert result["keyword"].str.islower().all()

    def test_known_pain_keyword_appears(self, sample_reviews):
        extractor = PainPointExtractor(rating_threshold=2, top_n=20)
        result = extractor.extract(sample_reviews)
        keywords = result["keyword"].tolist()
        # "crashing" or "crashes" should appear in low-rating reviews
        assert any("crash" in kw for kw in keywords)

    def test_frequency_between_0_and_1(self, sample_reviews):
        extractor = PainPointExtractor(rating_threshold=2, top_n=20)
        result = extractor.extract(sample_reviews)
        assert result["frequency"].between(0, 1).all()
