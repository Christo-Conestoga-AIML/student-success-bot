# lr_next_question_demo.py
# Demonstrates Logistic Regression re-ranking for "next questions" using the same features as your app.
# Saves two figures: lr_demo_rank.png (bar chart) and lr_demo_pca.png (2D PCA scatter).

import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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
# ----------------------------------------``
def _clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def build_features(user_query: str, candidates: list[str]):
    uq = _clean(user_query)
    cqs = [_clean(c) for c in candidates]

    # TF‑IDF over [query + candidates]
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    mat = tfidf.fit_transform([uq] + cqs)
    q_vec, C = mat[0:1], mat[1:]
    # Cosine ~ dot product because vectors are L2-normalized by default
    tfidf_cos = (C @ q_vec.T).toarray().ravel()

    # Jaccard overlap
    jacc = np.array([_jaccard(uq, c) for c in cqs], dtype=float)

    # Simulated FAISS L2 distance (smaller is better) correlated with 1 - cosine
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
# 3) Weak labels & model training
# --------------------------------
X, distance, tfidf_cos = build_features(user_query, candidate_questions)

# Weak label: assume the most similar (highest tfidf_cos) is the one the user would click next
pos_idx = int(np.argmax(tfidf_cos))
y = np.zeros(len(candidate_questions), dtype=int)
y[pos_idx] = 1

# Train LR (tiny demo set; in reality you'd train across many queries)
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(X, y)

# Predict probabilities (our ranking score)
probs = lr.predict_proba(X)[:, 1]
order = np.argsort(-probs)

# --------------------------------
# 4) Pretty output + plots
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
plt.scatter(X2[:, 0], X2[:, 1], s=80, c=probs)  # no explicit colors per your constraints
for i, txt in enumerate(range(len(candidate_questions))):
    plt.annotate(str(i+1), (X2[i, 0], X2[i, 1]), xytext=(5, 5), textcoords="offset points", fontsize=9)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Feature Space (PCA) — Color = LR Probability")
plt.tight_layout()
plt.savefig("lr_demo_pca.png", dpi=200)
plt.show()

print("\nSaved figures: lr_demo_rank.png, lr_demo_pca.png")
