
import os
import re
import joblib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@dataclass
class Candidate:
    question: str
    distance: float  # FAISS L2 distance (smaller=better)

class _Feat:
    def __init__(self):
        # we fit per-query to keep things simple and robust
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)

    @staticmethod
    def _clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        A, B = set(a.split()), set(b.split())
        if not A and not B:
            return 0.0
        inter = len(A & B); union = len(A | B)
        return inter / union if union else 0.0

    def build(self, user_query: str, candidates: List[Candidate]) -> np.ndarray:
        uq = self._clean(user_query)
        cq = [self._clean(c.question) for c in candidates]

        # TF‑IDF over [query + cands]; cosine via dot (L2 normalized by default)
        corpus = [uq] + cq
        mat = self.tfidf.fit_transform(corpus)
        q_vec, C = mat[0:1], mat[1:]
        tfidf_cos = (C @ q_vec.T).toarray().ravel()

        jacc = np.array([self._jaccard(uq, c) for c in cq], dtype=float)

        d = np.array([c.distance for c in candidates], dtype=float)
        d = np.nan_to_num(d, nan=np.nanmax(d) + 1.0)
        inv1 = 1.0 / (1.0 + d)
        expd = np.exp(-d)

        qlen = len(uq.split())
        clen = np.array([len(c.split()) for c in cq], dtype=float)
        abs_len_diff = np.abs(clen - qlen)

        # shape: [n, n_features]
        X = np.vstack([tfidf_cos, jacc, inv1, expd, abs_len_diff]).T
        return X

class NextQuestionLR:
    """
    Logistic Regression re-ranker.
    Train offline to create models/nextq_lr.joblib.
    At inference, if the model file isn’t found -> fallback to FAISS order.
    """
    def __init__(self, model_path: str = "models/nextq_lr.joblib"):
        self.model_path = model_path
        self.model: LogisticRegression | None = None
        self.fb = _Feat()

    def _load(self):
        if self.model is None and os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def rerank(self, user_query: str, candidates: List[Candidate], top_k: int = 6) -> List[int]:
        self._load()
        if not candidates:
            return []

        # No trained model? fall back to FAISS order (distance ascending)
        if self.model is None:
            order = np.argsort([c.distance for c in candidates])  # smaller is better
            return order[:top_k].tolist()

        X = self.fb.build(user_query, candidates)
        # probability of class 1 == “clicked/likely”
        scores = self.model.predict_proba(X)[:, 1]
        order = np.argsort(-scores)
        return order[:top_k].tolist()
