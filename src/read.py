from huggingface_hub import HfFileSystem
import json
import os
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

CACHE_PATH = "../cache/hf_files.json"

def get_files(url: str = DATA_URL) -> list[str]:
    if os.path.exists(CACHE_PATH):
        print(f"Loading HF files from {CACHE_PATH}")
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    
    fs = HfFileSystem()
    files = [f"hf://{p}" for p in fs.glob(url)]
    
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(files, f)
    
    print(f"Found and cached {len(files)} files")
    return files

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
    use_head: bool=True,
) -> pl.DataFrame:
    """
    Collects a stratified sample as an eager DataFrame.
    This is the entry point for the sampling pipeline.
    """
    lf = pl.scan_parquet(get_files(url), hive_partitioning=False)
    if columns:
        lf = lf.select(columns)

    if use_head:
        sample = lf.head(n).collect()
    else:
        sample = (
            lf
            .collect(streaming=True)
            .sample(n=n, seed=seed)
        )

    sample = sample.filter(pl.col("original_text").is_not_null())
    if language:
        sample = sample.filter(pl.col("language") == language)
    if theme:
        sample = sample.filter(pl.col("primary_theme") == theme)
    sample = add_frequency_features(sample.lazy()).collect()
    sample = add_row_id(sample.lazy()).collect()
    return sample
