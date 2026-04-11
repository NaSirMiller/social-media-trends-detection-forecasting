from __future__ import annotations

import os
import numpy as np
from scipy.sparse import spmatrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

_TFIDF_VECTORIZER = TfidfVectorizer(
    max_features=100_000,
    min_df=5,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
)

def fit_tfidf(
    X: list[str],
    vectorizer: TfidfVectorizer = _TFIDF_VECTORIZER,
) -> TfidfVectorizer:
    """Fit vectorizer on corpus."""
    return vectorizer.fit(X)

def to_tfidf_vector(
    X: list[str],
    vectorizer: TfidfVectorizer = _TFIDF_VECTORIZER,
) -> spmatrix:
    """Transform texts to TF-IDF matrix."""
    return vectorizer.transform(X)

_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-m3", device="mps", token=os.getenv("HF_TOKEN"))
    return _EMBEDDING_MODEL

def to_embedding(
    X: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts to dense BGE-M3 embeddings. Shape: (N, 1024)."""
    return get_embedding_model().encode(
        X,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )