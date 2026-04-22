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
from typing import Callable, Optional


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


def ild_at_k(top_k_ids: list[str], news_vecs: dict, k: int) -> float:
    """
    Intra-List Diversity at k: average pairwise cosine distance among top-k items.
    Items not in news_vecs are skipped. Returns 0.0 if fewer than 2 known items.
    """
    vecs = [news_vecs[nid] for nid in top_k_ids[:k] if nid in news_vecs]
    if len(vecs) < 2:
        return 0.0
    mat = np.stack(vecs)                      # (m, d)
    # cosine similarity matrix (vectors are pre-normalised)
    sim = mat @ mat.T                         # (m, m)
    m = len(vecs)
    # sum of upper triangle (excluding diagonal)
    total_sim = (sim.sum() - np.trace(sim)) / 2
    n_pairs = m * (m - 1) / 2
    return float(1.0 - total_sim / n_pairs)   # distance = 1 - similarity


def novelty_at_k(top_k_ids: list[str], global_pop: dict, total_clicks: int, k: int) -> float:
    """
    Novelty at k: mean -log2(pop(i) / total_clicks) for top-k items.
    Items with no recorded clicks are treated as pop=1 (maximum novelty).
    """
    scores = []
    for nid in top_k_ids[:k]:
        pop = global_pop.get(nid, 1)
        scores.append(-np.log2(pop / total_clicks))
    return float(np.mean(scores)) if scores else 0.0


# ─── Central evaluation loop ──────────────────────────────────────────────────

def evaluate(
    behaviors_path: str,
    score_fn: Callable[[list[str], list[str]], list[float]],
    news_vecs: Optional[dict] = None,
    global_pop: Optional[dict] = None,
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
    news_vecs : dict, optional
        news_id -> L2-normalised np.array. When provided, ILD@5 and ILD@10
        are computed over the top-k ranked candidates.
    global_pop : dict, optional
        news_id -> click count from training. When provided, Novelty@5 and
        Novelty@10 are computed over the top-k ranked candidates.
    verbose : bool
        Print progress summary when done.

    Returns
    -------
    dict with keys: AUC, MRR, nDCG@5, nDCG@10
    and optionally: ILD@5, ILD@10, Novelty@5, Novelty@10
    """
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    ild5s, ild10s, nov5s, nov10s = [], [], [], []
    skipped = 0

    total_clicks = sum(global_pop.values()) if global_pop else 1

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

            ranked_pairs = sorted(zip(scores, candidates, labels), reverse=True)
            ranked_labels = [lbl for _, _, lbl in ranked_pairs]
            ranked_ids    = [nid for _, nid, _  in ranked_pairs]

            aucs.append(roc_auc_score(labels, scores))
            mrrs.append(mrr(ranked_labels))
            ndcg5s.append(ndcg_at_k(ranked_labels, 5))
            ndcg10s.append(ndcg_at_k(ranked_labels, 10))

            if news_vecs is not None:
                ild5s.append(ild_at_k(ranked_ids, news_vecs, 5))
                ild10s.append(ild_at_k(ranked_ids, news_vecs, 10))

            if global_pop is not None:
                nov5s.append(novelty_at_k(ranked_ids, global_pop, total_clicks, 5))
                nov10s.append(novelty_at_k(ranked_ids, global_pop, total_clicks, 10))

    if verbose:
        print(f"  Evaluated {len(aucs)} impressions | Skipped (degenerate): {skipped}")

    result = {
        "AUC":     float(np.mean(aucs)),
        "MRR":     float(np.mean(mrrs)),
        "nDCG@5":  float(np.mean(ndcg5s)),
        "nDCG@10": float(np.mean(ndcg10s)),
    }
    if ild5s:
        result["ILD@5"]  = float(np.mean(ild5s))
        result["ILD@10"] = float(np.mean(ild10s))
    if nov5s:
        result["Novelty@5"]  = float(np.mean(nov5s))
        result["Novelty@10"] = float(np.mean(nov10s))
    return result
