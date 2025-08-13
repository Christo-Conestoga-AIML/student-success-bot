# lr_next_question_demo.py
# Demonstrates Logistic Regression re-ranking for "next questions"
# + adds a TRAINING CURVE using SGD-based logistic regression (per-epoch metrics).
# Outputs:
#   - lr_demo_rank.png       (bar chart of predicted probabilities)
#   - lr_demo_pca.png        (2D PCA scatter colored by prob)
#   - lr_training_curve.png  (train/val log-loss & accuracy vs epochs)

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

# --------------------------
# 1) Sample data (EDIT ME)
# --------------------------
user_query = "How do I pay my tuition fees?"

candidate_questions = [
    "Fees and payments",
    "Opting out of optional fees",
    "Tuition fee payment default",
    "Student invoice and fee information",
    "Paying fees with OSAP",
    "Tuition rate and compulsory ancillary fees for part-time courses",
]

# ----------------------------------------
# 2) Feature builder (same as in your app)
# ----------------------------------------
def _clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 0.0
    inter = len(A & B); union = len(A | B)
    return inter / union if union else 0.0

def build_features(user_query: str, candidates: list[str]):
    uq = _clean(user_query)
    cqs = [_clean(c) for c in candidates]

    # TF-IDF over [query + candidates]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    mat = tfidf.fit_transform([uq] + cqs)
    q_vec, C = mat[0:1], mat[1:]
    # Cosine ~ dot product (vectors are L2-normalized by default)
    tfidf_cos = (C @ q_vec.T).toarray().ravel()

    # Jaccard overlap
    jacc = np.array([_jaccard(uq, c) for c in cqs], dtype=float)

    # Simulated FAISS L2 distance (smaller is better) ~ correlated with 1 - cosine
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, size=len(cqs))
    distance = np.clip(1.0 - tfidf_cos + noise, 0, None)

    # Distance transforms
    inv1 = 1.0 / (1.0 + distance)
    expd = np.exp(-distance)

    # Length features
    qlen = len(uq.split())
    clen = np.array([len(c.split()) for c in cqs], dtype=float)
    abs_len_diff = np.abs(clen - qlen)

    # Feature matrix (n_candidates, n_features)
    X = np.vstack([tfidf_cos, jacc, inv1, expd, abs_len_diff]).T
    return X, distance, tfidf_cos

# --------------------------------
# 3) Weak labels & LR (as before)
# --------------------------------
X, distance, tfidf_cos = build_features(user_query, candidate_questions)

# Weak label: most similar by tfidf_cos is positive (clicked); others negative
pos_idx = int(np.argmax(tfidf_cos))
y = np.zeros(len(candidate_questions), dtype=int)
y[pos_idx] = 1

# Classic LR (for ranking output only)
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(X, y)
probs = lr.predict_proba(X)[:, 1]
order = np.argsort(-probs)

# --------------------------------
# 4) Pretty output + plots (same)
# --------------------------------
print("User query:", user_query)
print("\nRanked ‘next question’ suggestions (by LR probability):\n")
for rank, idx in enumerate(order, start=1):
    print(f"{rank:>2}. p_click={probs[idx]:.3f}  |  {candidate_questions[idx]}")

# --- Bar chart (ranked) ---
labels_sorted = [candidate_questions[i] for i in order]
scores_sorted = probs[order]
plt.figure(figsize=(10, 5))
plt.barh(range(len(labels_sorted)), scores_sorted)
plt.gca().invert_yaxis()
plt.yticks(range(len(labels_sorted)), [l if len(l) < 55 else l[:52] + "..." for l in labels_sorted])
plt.xlabel("Predicted probability (Logistic Regression)")
plt.title("Next-Question Prediction — LR Probabilities (Higher is Better)")
plt.tight_layout()
plt.savefig("lr_demo_rank.png", dpi=200)
plt.show()

# --- PCA scatter (2D) ---
pca = PCA(n_components=2, random_state=0)
X2 = pca.fit_transform(X)
plt.figure(figsize=(6, 5))
plt.scatter(X2[:, 0], X2[:, 1], s=80, c=probs)  # colormap left to default (no explicit colors)
for i, _ in enumerate(candidate_questions):
    plt.annotate(str(i+1), (X2[i, 0], X2[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Feature Space (PCA) — Color = LR Probability")
plt.tight_layout()
plt.savefig("lr_demo_pca.png", dpi=200)
plt.show()

print("\nSaved figures: lr_demo_rank.png, lr_demo_pca.png")

# ---------------------------------------------------------
# 5) TRAINING CURVE (Logistic Regression trained via SGD)
# ---------------------------------------------------------
# Safe stratify logic for small datasets
class_counts = Counter(y)
if min(class_counts.values()) < 2:
    stratify_opt = None
else:
    stratify_opt = y

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.33,
    random_state=123,
    stratify=stratify_opt
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# Initialize SGD-based logistic regression
sgd = SGDClassifier(
    loss="log_loss",       # logistic regression objective
    penalty="l2",
    alpha=1e-3,            # L2 regularization strength (~1/C)
    learning_rate="optimal",
    random_state=0
)

# partial_fit requires class labels up front
classes = np.array([0, 1], dtype=int)
epochs = 60

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

# First partial_fit to initialize
sgd.partial_fit(X_train_s, y_train, classes=classes)

for epoch in range(epochs):
    # Shuffle each epoch
    idx = np.arange(len(X_train_s))
    np.random.shuffle(idx)
    for i in idx:
        sgd.partial_fit(X_train_s[i:i+1], y_train[i:i+1])

    # Predict proba or fallback to sigmoid(decision_function)
    if hasattr(sgd, "predict_proba"):
        p_train = sgd.predict_proba(X_train_s)[:, 1]
        p_val   = sgd.predict_proba(X_val_s)[:, 1] if len(X_val_s) else np.array([])
    else:
        def _sigmoid(z): return 1 / (1 + np.exp(-z))
        p_train = _sigmoid(sgd.decision_function(X_train_s))
        p_val   = _sigmoid(sgd.decision_function(X_val_s)) if len(X_val_s) else np.array([])

    train_losses.append(log_loss(y_train, p_train, labels=[0,1]))
    train_accs.append(accuracy_score(y_train, (p_train >= 0.5).astype(int)))

    if len(X_val_s):
        val_losses.append(log_loss(y_val, p_val, labels=[0,1]))
        val_accs.append(accuracy_score(y_val, (p_val >= 0.5).astype(int)))
    else:
        val_losses.append(np.nan)
        val_accs.append(np.nan)

# --- Plot training loss curve ---
epochs_axis = np.arange(1, epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_axis, train_losses, label="Train Log-Loss")
if not np.isnan(val_losses).all():
    plt.plot(epochs_axis, val_losses, label="Val Log-Loss")
plt.xlabel("Epoch")
plt.ylabel("Log-Loss")
plt.title("Logistic Regression (SGD) — Training Curve (Loss per Epoch)")
plt.legend()
plt.tight_layout()
plt.savefig("lr_training_curve.png", dpi=200)
plt.show()

# --- Accuracy curve ---
plt.figure(figsize=(10, 5))
plt.plot(epochs_axis, train_accs, label="Train Accuracy")
if not np.isnan(val_accs).all():
    plt.plot(epochs_axis, val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Logistic Regression (SGD) — Accuracy per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("lr_accuracy_curve.png", dpi=200)
plt.show()

print("Saved training figures: lr_training_curve.png, lr_accuracy_curve.png")
