import polars as pl
from timeit_decorator import timeit_sync
from typing import Optional

DATA_URL = "hf://datasets/Exorde/exorde-social-media-one-month-2024/**/*.parquet"

SELECTED_COLUMNS = [
    "date",
    "original_text",
    "author_hash",
    "language",
    "primary_theme",
    "english_keywords",
    "sentiment",
]

def load_scan(
    url: str = DATA_URL,
    columns: Optional[list[str]] = None,
    language: Optional[str] = None,
    theme: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Returns a lazy frame with optional column projection and filters.
    No data is read until .collect() is called.
    """
    lf = pl.scan_parquet(url, hive_partitioning=False)

    if columns:
        lf = lf.select(columns)
    if language:
        lf = lf.filter(pl.col("language") == language)
    if theme:
        lf = lf.filter(pl.col("primary_theme") == theme)

    return lf


def add_frequency_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Adds author_post_frequency and content_frequency window columns."""
    return lf.with_columns([
        pl.col("original_text")
          .count()
          .over(["original_text", "author_hash"])
          .alias("author_post_frequency"),
        pl.col("original_text")
          .count()
          .over("original_text")
          .alias("content_frequency"),
    ])


def add_row_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_row_index(name="id", offset=0)


def load_sample(
    n: int = 1_000_000,
    seed: int = 42,
    url: str = DATA_URL,
    columns: Optional[list[str]] = None,
    language: Optional[str] = None,
    theme: Optional[str] = None,
) -> pl.DataFrame:
    """
    Collects a stratified sample as an eager DataFrame.
    This is the entry point for the sampling pipeline.
    """
    lf = load_scan(url=url, columns=columns, language=language, theme=theme)
    lf = add_frequency_features(lf)
    lf = add_row_id(lf)

    return (
        lf
        .filter(pl.col("original_text").is_not_null())
        .collect(streaming=True)
        .sample(n=n, seed=seed)
    )
