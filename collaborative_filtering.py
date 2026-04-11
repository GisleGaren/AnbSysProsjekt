"""
collaborative_filtering.py
==========================
Item-Based Collaborative Filtering for the MIND dataset.

Two articles are "similar" if the same users tend to click both.
To score candidates for a user, we compare each candidate's click-vector
against the user's history articles via cosine similarity.

Usage:
    python collaborative_filtering.py
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from evaluation import evaluate

# ─── Paths ───────────────────────────────────────────────────────────────────

TRAIN_BEHAVIORS = "smallDataset/MINDsmall_train/behaviors.tsv"
DEV_BEHAVIORS   = "smallDataset/MINDsmall_dev/behaviors.tsv"

COLS = ["impression_id", "user_id", "time", "history", "impressions"]

# ─── Step 1: Collect user-article clicks from training data ──────────────────

print("Loading training clicks...")

train = pd.read_csv(TRAIN_BEHAVIORS, sep="\t", header=None, names=COLS)

user_clicks: dict[str, set[str]] = defaultdict(set)

for _, row in train.iterrows():
    uid = row["user_id"]

    if isinstance(row["history"], str) and row["history"].strip():
        for nid in row["history"].split():
            user_clicks[uid].add(nid)

    if isinstance(row["impressions"], str):
        for item in row["impressions"].split():
            nid, label = item.rsplit("-", 1)
            if label == "1":
                user_clicks[uid].add(nid)

print(f"  Unique users: {len(user_clicks)}")
print(f"  Total (user, article) clicks: {sum(len(v) for v in user_clicks.values())}")

# ─── Step 2: Build sparse user-article interaction matrix ────────────────────

print("Building interaction matrix...")

user_to_idx    = {u: i for i, u in enumerate(user_clicks.keys())}
all_articles   = sorted({nid for clicks in user_clicks.values() for nid in clicks})
article_to_idx = {a: i for i, a in enumerate(all_articles)}

n_users    = len(user_to_idx)
n_articles = len(article_to_idx)

rows, cols = [], []
for uid, clicks in user_clicks.items():
    u_idx = user_to_idx[uid]
    for nid in clicks:
        rows.append(u_idx)
        cols.append(article_to_idx[nid])

data = np.ones(len(rows), dtype=np.float32)
interaction = csr_matrix((data, (rows, cols)), shape=(n_users, n_articles))

print(f"  Shape: {interaction.shape} | Non-zeros: {interaction.nnz} | "
      f"Density: {100 * interaction.nnz / (n_users * n_articles):.4f}%")

# ─── Step 3: Normalise for cosine similarity ─────────────────────────────────

print("Normalising article vectors...")

# Shape: (n_articles, n_users), each row L2-normalised
article_user_norm = normalize(interaction.T.tocsr(), norm="l2", axis=1, copy=False)

print("  Done. Dot products now yield cosine similarities.")

# ─── Step 4: Scoring function ────────────────────────────────────────────────

def cf_score(history: list[str], candidates: list[str]) -> list[float]:
    """
    Score candidates by average cosine similarity to the user's history articles.
    Unknown articles (cold-start) receive score 0.
    """
    hist_idx = [article_to_idx[h] for h in history if h in article_to_idx]

    if not hist_idx:
        return [0.0] * len(candidates)

    hist_vecs = article_user_norm[hist_idx]  # (h, n_users)

    scores = []
    for cand in candidates:
        c_idx = article_to_idx.get(cand)
        if c_idx is None:
            scores.append(0.0)
            continue
        cand_vec = article_user_norm[c_idx]
        sims = cand_vec @ hist_vecs.T                    # (1, h)
        sims = np.asarray(sims.todense()).ravel()
        scores.append(float(sims.mean()))

    return scores

# ─── Step 5: Evaluate ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nEvaluating on dev set...")
    metrics = evaluate(DEV_BEHAVIORS, cf_score)

    print("\nItem-Based CF — Dev Set Results")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")
