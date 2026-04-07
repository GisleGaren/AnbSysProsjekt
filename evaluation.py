"""
evaluation.py
=============
Central evaluation module for MIND-dataset recommender systems.

Usage in a model file:
    from evaluation import evaluate

    def my_score_fn(history: list[str], candidates: list[str]) -> list[float]:
        # return a score per candidate (higher = more relevant)
        ...

    metrics = evaluate("path/to/behaviors.tsv", my_score_fn)
    print(metrics)  # {"AUC": ..., "MRR": ..., "nDCG@5": ..., "nDCG@10": ...}

The score_fn receives:
  - history:    list of news IDs the user clicked before this impression
  - candidates: list of news IDs shown in the impression (order matches labels)
And must return a list of floats of the same length as candidates.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Callable


# ─── Metric helpers ───────────────────────────────────────────────────────────

def dcg_at_k(relevance: list, k: int) -> float:
    relevance = np.array(relevance[:k], dtype=np.float32)
    if relevance.size == 0:
        return 0.0
    return float((relevance / np.log2(np.arange(2, relevance.size + 2))).sum())


def ndcg_at_k(relevance: list, k: int) -> float:
    ideal = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(relevance, k) / idcg if idcg > 0 else 0.0


def mrr(relevance: list) -> float:
    for i, r in enumerate(relevance):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0


# ─── Central evaluation loop ──────────────────────────────────────────────────

def evaluate(
    behaviors_path: str,
    score_fn: Callable[[list[str], list[str]], list[float]],
    verbose: bool = True,
) -> dict[str, float]:
    """
    Parse behaviors.tsv, call score_fn for each impression, compute metrics.

    Parameters
    ----------
    behaviors_path : str
        Path to a MIND behaviors.tsv file.
    score_fn : callable
        score_fn(history, candidates) -> list of floats
        where history and candidates are lists of news IDs.
    verbose : bool
        Print progress summary when done.

    Returns
    -------
    dict with keys: AUC, MRR, nDCG@5, nDCG@10
    """
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    skipped = 0

    with open(behaviors_path, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue

            history = cols[3].strip().split() if cols[3].strip() else []
            impressions = cols[4].strip().split()

            if not impressions:
                continue

            candidates, labels = [], []
            for imp in impressions:
                parts = imp.rsplit("-", 1)
                if len(parts) != 2:
                    continue
                nid, label = parts
                candidates.append(nid)
                labels.append(int(label))

            # Skip impressions where AUC is undefined
            if sum(labels) == 0 or sum(labels) == len(labels):
                skipped += 1
                continue

            scores = score_fn(history, candidates)

            ranked = [lbl for _, lbl in sorted(zip(scores, labels), reverse=True)]

            aucs.append(roc_auc_score(labels, scores))
            mrrs.append(mrr(ranked))
            ndcg5s.append(ndcg_at_k(ranked, 5))
            ndcg10s.append(ndcg_at_k(ranked, 10))

    if verbose:
        print(f"  Evaluated {len(aucs)} impressions | Skipped (degenerate): {skipped}")

    return {
        "AUC":     float(np.mean(aucs)),
        "MRR":     float(np.mean(mrrs)),
        "nDCG@5":  float(np.mean(ndcg5s)),
        "nDCG@10": float(np.mean(ndcg10s)),
    }
