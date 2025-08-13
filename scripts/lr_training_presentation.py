"""
lr_training_presentation.py

Builds a presentation-ready training run for your Logistic Regression next-question
re-ranker using the EXACT pipeline you trained with:

- Collects questions from the KB metadata
- For each question, retrieves top-N neighbors (VectorDBHelper)
- Builds features via scripts.next_question_ranker._Feat (TF-IDF cosine, Jaccard,
  distance transforms, length diff)
- Creates weak labels (top-1 neighbor = positive, rest = negative)
- Trains Logistic Regression via SGD (log-loss) so we can log per-epoch metrics
- Saves plots:
    lr_train_curve.png      (log-loss per epoch, train + val)
    lr_accuracy_curve.png   (accuracy per epoch, train + val)
    lr_prob_hist.png        (val predicted-prob hist by class)
    lr_confusion.png        (val confusion matrix)
    lr_roc.png              (ROC curve) if both classes present in val
    lr_pr.png               (Precision-Recall curve) if both classes present in val
"""

import os, re, pickle, sys, logging
from collections import Counter
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    log_loss, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ---- Make sure project modules resolve (run from repo root) ----
sys.path.append(".")

from scripts.next_question_ranker import Candidate, _Feat  # uses your exact features
from knowledge_base.knowledge_base_helper import VectorDBHelper
from knowledge_base.kb_builder import VectorDbBuilder
from utils.constants import Constants

# ------------------------- Config -------------------------
N_CANDS = 10                 # neighbors per query
VAL_SIZE = 0.25              # validation split
EPOCHS   = 60                # SGD epochs
BATCH_SZ = 64                # mini-batch size for SGD
ALPHA    = 1e-3              # L2 regularization (~1/C)
SEED     = 123
FAQ_META_PATH = "data/generated/faq_metadata.pkl"
# ---------------------------------------------------------

logger = logging.getLogger("lr_training_presentation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ------------------------ KB helpers ------------------------
def _as_meta(it):
    if isinstance(it, dict):
        return it
    if hasattr(it, "metadata") and isinstance(getattr(it, "metadata"), dict):
        return getattr(it, "metadata")
    return {}

_CAND_KEYS = ("question", "Question", "q", "title", "heading", "prompt")
_NESTED_PARENTS = ("faq", "qa", "meta", "metadata")

def _extract_question(meta: dict):
    # direct keys
    for k in _CAND_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested dicts
    for parent in _NESTED_PARENTS:
        v = meta.get(parent)
        if isinstance(v, dict):
            for k in _CAND_KEYS:
                vv = v.get(k)
                if isinstance(vv, str) and vv.strip():
                    return vv.strip()
    # heuristic from text
    text = meta.get("text") or meta.get("content") or meta.get("answer") or ""
    if isinstance(text, str) and text:
        m = re.search(r"(.{5,}?\?)", text)
        if m:
            return m.group(1).strip()
    return None

def _load_questions_from_vdb(vdb: VectorDBHelper) -> List[str]:
    qs = []
    items = getattr(vdb, "knowledge_base_meta_data", None) or []
    for it in items:
        meta = _as_meta(it)
        q = _extract_question(meta)
        if q:
            qs.append(q)
    return qs

def _load_questions_from_pkl(path=FAQ_META_PATH) -> List[str]:
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

def _dedup_keep_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if not isinstance(s, str):
            continue
        t = s.strip()
        if not t or t in seen: 
            continue
        seen.add(t)
        out.append(t)
    return out


# ---------------------- Data assembly ----------------------
def assemble_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """Build X,y using your exact KB → neighbors → features pipeline."""
    # Ensure KB exists
    if Constants.should_build_vector_db():
        VectorDbBuilder.build_db()
    vdb = VectorDBHelper()
    vdb.load_kb()

    kb_questions = _load_questions_from_vdb(vdb)
    if not kb_questions:
        kb_questions = _load_questions_from_pkl(FAQ_META_PATH)
    kb_questions = _dedup_keep_order(kb_questions)

    if not kb_questions:
        raise RuntimeError(
            "No questions found in KB metadata. Rebuild KB and ensure your CSV has Question/Answer columns."
        )

    # neighbor helper using your retriever
    def neighbors(q: str, k: int):
        items = vdb.get_faq_response_from_kb(q, top_k=k)  # has .question and .confidence
        return [Candidate(question=i.question, distance=i.confidence) for i in items]

    feat = _Feat()
    X_all, y_all = [], []
    skipped = 0

    for q in kb_questions:
        cands = neighbors(q, N_CANDS)
        if len(cands) < 2:
            skipped += 1
            continue

        # Build features for this query's candidates
        X = feat.build(q, cands)

        # Weak labels: assume the nearest candidate is the "clicked" one
        y = np.zeros(len(cands), dtype=int)
        y[0] = 1

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No training data assembled. Is your KB loaded and returning neighbors?")

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    logger.info(f"Assembled training matrix: X={X_all.shape}, positives={y_all.sum()}, negatives={(y_all==0).sum()}, skipped_queries={skipped}")
    return X_all, y_all


# ------------------------ Plot helpers ------------------------
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs_axis = np.arange(1, len(train_losses) + 1)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_axis, train_losses, label="Train Log-Loss")
    if not np.isnan(val_losses).all():
        plt.plot(epochs_axis, val_losses, label="Val Log-Loss")
    plt.xlabel("Epoch"); plt.ylabel("Log-Loss")
    plt.title("Logistic Regression (SGD) — Training Curve (Loss per Epoch)")
    plt.legend(); plt.tight_layout()
    plt.savefig("lr_train_curve.png", dpi=200)
    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_axis, train_accs, label="Train Accuracy")
    if not np.isnan(val_accs).all():
        plt.plot(epochs_axis, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Logistic Regression (SGD) — Accuracy per Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig("lr_accuracy_curve.png", dpi=200)
    plt.show()

def plot_prob_hist(y_true, y_prob, title="Predicted Probabilities (Validation)"):
    plt.figure(figsize=(8,5))
    plt.hist(y_prob[y_true==1], bins=25, alpha=0.6, label="Positives")
    plt.hist(y_prob[y_true==0], bins=25, alpha=0.6, label="Negatives")
    plt.xlabel("Predicted probability (class 1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig("lr_prob_hist.png", dpi=200)
    plt.show()

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Validation)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white")
    fig.tight_layout(); fig.savefig("lr_confusion.png", dpi=200); plt.show()

def plot_roc_pr(y_true, y_prob):
    # Only meaningful if val has both classes
    if len(np.unique(y_true)) < 2:
        logger.warning("Validation set has a single class; skipping ROC/PR plots.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC (Validation)")
    plt.legend(); plt.tight_layout(); plt.savefig("lr_roc.png", dpi=200); plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall (Validation)")
    plt.legend(); plt.tight_layout(); plt.savefig("lr_pr.png", dpi=200); plt.show()


# ---------------------------- Main ----------------------------
def main():
    # 1) Build dataset from KB
    X, y = assemble_training_data()

    # 2) Balanced split (avoid the 1-positive issue)
    class_counts = Counter(y)
    stratify_opt = y if min(class_counts.values()) >= 2 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=SEED, stratify=stratify_opt
    )

    # 3) Scale features (helps SGD convergence)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # 4) SGD-based Logistic Regression (so we can log per-epoch metrics)
    sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=ALPHA,
        learning_rate="optimal",
        random_state=SEED
    )

    classes = np.array([0, 1], dtype=int)
    sgd.partial_fit(X_train_s, y_train, classes=classes)

    rng = np.random.default_rng(SEED)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        # Mini-batch SGD
        idx = np.arange(len(X_train_s))
        rng.shuffle(idx)
        for start in range(0, len(idx), BATCH_SZ):
            batch = idx[start:start+BATCH_SZ]
            sgd.partial_fit(X_train_s[batch], y_train[batch])

        # probs (or sigmoid(decision_function) fallback)
        if hasattr(sgd, "predict_proba"):
            p_tr = sgd.predict_proba(X_train_s)[:, 1]
            p_va = sgd.predict_proba(X_val_s)[:, 1]
        else:
            sigm = lambda z: 1 / (1 + np.exp(-z))
            p_tr = sigm(sgd.decision_function(X_train_s))
            p_va = sigm(sgd.decision_function(X_val_s))

        train_losses.append(log_loss(y_train, p_tr, labels=[0,1]))
        val_losses.append(log_loss(y_val, p_va, labels=[0,1]))
        train_accs.append(accuracy_score(y_train, (p_tr >= 0.5).astype(int)))
        val_accs.append(accuracy_score(y_val, (p_va >= 0.5).astype(int)))

    # 5) Plots for slides
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_prob_hist(y_val, p_va)
    plot_confusion(y_val, (p_va >= 0.5).astype(int))
    plot_roc_pr(y_val, p_va)

    print("Saved figures:")
    print("  lr_train_curve.png, lr_accuracy_curve.png, lr_prob_hist.png, lr_confusion.png")
    print("  (Optional) lr_roc.png, lr_pr.png")

if __name__ == "__main__":
    main()
