import os, sys, re, pickle
sys.path.append(".")

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from scripts.next_question_ranker import Candidate, _Feat
from knowledge_base.knowledge_base_helper import VectorDBHelper
from knowledge_base.kb_builder import VectorDbBuilder
from utils.constants import Constants

N_CANDS = 10
MODEL_PATH = "models/nextq_lr.joblib"
FAQ_META_PATH = "data/generated/faq_metadata.pkl"


# ---------- helpers ----------
def _as_meta(it):
    if isinstance(it, dict):
        return it
    # many vector DB wrappers expose .metadata or attrs
    if hasattr(it, "metadata") and isinstance(it.metadata, dict):
        return it.metadata
    return {}

_CAND_KEYS = ("question", "Question", "q", "title", "heading", "prompt")
_NESTED_PARENTS = ("faq", "qa", "meta", "metadata")

def _extract_question(meta: dict):
    # 1) direct keys
    for k in _CAND_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 2) common nested dicts
    for parent in _NESTED_PARENTS:
        v = meta.get(parent)
        if isinstance(v, dict):
            for k in _CAND_KEYS:
                vv = v.get(k)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    # 3) heuristic from text/content
    text = meta.get("text") or meta.get("content") or meta.get("answer") or ""
    if isinstance(text, str) and text:
        m = re.search(r"(.{5,}?\?)", text)
        if m:
            return m.group(1).strip()
    return None


def _load_questions_from_vdb(vdb):
    qs = []
    items = getattr(vdb, "knowledge_base_meta_data", None) or []
    for it in items:
        meta = _as_meta(it)
        q = _extract_question(meta)
        if q:
            qs.append(q)
    return qs


def _load_questions_from_pkl(path=FAQ_META_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            recs = pickle.load(f)
    except Exception:
        return []
    qs = []
    for r in recs if isinstance(recs, list) else []:
        if isinstance(r, dict):
            q = r.get("question") or _extract_question(r)
            if isinstance(q, str) and q.strip():
                qs.append(q.strip())
    return qs


def _dedup_keep_order(seq):
    seen = set()
    out = []
    for s in seq:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out
# ---------- end helpers ----------


def main():
    os.makedirs("models", exist_ok=True)

    # ensure KB is built with normalized metadata if configured
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()

    vdb = VectorDBHelper()
    vdb.load_kb()

    # get seed questions
    kb_questions = _load_questions_from_vdb(vdb)
    if not kb_questions:
        # fallback to the saved metadata file created by VectorDbBuilder
        kb_questions = _load_questions_from_pkl(FAQ_META_PATH)

    kb_questions = _dedup_keep_order(kb_questions)

    if not kb_questions:
        raise RuntimeError(
            "No questions found in KB metadata. "
            "Rebuild your KB (VectorDbBuilder.build_db()) and ensure your CSV has Question/Answer columns."
        )

    # neighbor helper using your existing retriever
    def neighbors(q: str, k: int):
        items = vdb.get_faq_response_from_kb(q, top_k=k)  # expects .question and .confidence
        return [Candidate(question=i.question, distance=i.confidence) for i in items]

    feat = _Feat()
    X_all, y_all = [], []

    for q in kb_questions:
        cands = neighbors(q, N_CANDS)
        if len(cands) < 2:
            continue
        X = feat.build(q, cands)
        y = np.zeros(len(cands), dtype=int)
        y[0] = 1  # nearest neighbor as weak positive
        X_all.append(X)
        y_all.append(y)

    if not X_all:
        print("No training data assembled. Is your KB loaded and returning neighbors?")
        return

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_all, y_all)
    joblib.dump(lr, MODEL_PATH)
    print(f"Saved Logistic Regression model to {MODEL_PATH} with X shape {X_all.shape}")


if __name__ == "__main__":
    main()
