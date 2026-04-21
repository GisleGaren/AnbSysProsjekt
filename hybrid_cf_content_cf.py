"""
hybrid_cf_content_cf.py
=======================
Two-stage Content → CF pipeline for the MIND dataset.

  Stage 1 – Content filter:
      Score ALL candidates with the SBERT + TransE entity content model
      (identical to mind_entity_recsys).  Keep the top TOP_FRAC fraction.

  Stage 2 – CF final:
      Score the content shortlist with item-based collaborative filtering
      (identical to collaborative_filtering.py).  CF scores are the final
      ranking.  Eliminated candidates receive score 0.

Cold-start handling
-------------------
  - No content history → skip content filter, keep all candidates for CF
  - No CF signal       → fall back to content scores for surviving candidates
  - Both absent        → fall back to global popularity

Usage
-----
    python hybrid_cf_content_cf.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from evaluation import evaluate

# ─── Config ──────────────────────────────────────────────────────────────────

TOP_FRAC   = 0.5    # fraction of candidates kept after the content filter
USE_ENTITY = False  # include confidence-weighted TransE entity vectors

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE      = "smallDataset"
TRAIN_DIR = os.path.join(BASE, "MINDsmall_train")
DEV_DIR   = os.path.join(BASE, "MINDsmall_dev")
CACHE_DIR = os.path.join(BASE, "sbert_cache")

SBERT_DIM       = 768
TOTAL_SBERT_DIM = SBERT_DIM * 2
ENTITY_DIM      = 100
TOTAL_DIM       = TOTAL_SBERT_DIM + (ENTITY_DIM if USE_ENTITY else 0)


# =============================================================================
# Stage 1 – Content model  (from mind_entity_recsys)
# =============================================================================

def load_sbert_cache(npz_path: str) -> dict:
    data     = np.load(npz_path)
    all_vecs = np.concatenate([data["title_vecs"], data["abstract_vecs"]], axis=1)
    embeddings = dict(zip(data["ids"], all_vecs))
    print(f"  Loaded {len(embeddings)} SBERT embeddings from {npz_path}")
    return embeddings


def load_entity_embeddings(vec_path: str) -> dict:
    entity_vecs = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            entity_vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(entity_vecs)} entity embeddings from {vec_path}")
    return entity_vecs


def parse_news(news_tsv: str, entity_vecs: dict) -> dict:
    news_entity_vecs = {}
    zero = np.zeros(ENTITY_DIM, dtype=np.float32)
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 7:
                continue
            news_id = cols[0]
            vecs, weights = [], []
            for col_idx in (6, 7):
                if col_idx >= len(cols):
                    continue
                try:
                    entities = json.loads(cols[col_idx])
                except (json.JSONDecodeError, ValueError):
                    continue
                for ent in entities:
                    qid  = ent.get("WikidataId", "")
                    conf = float(ent.get("Confidence", 1.0))
                    if qid in entity_vecs:
                        vecs.append(entity_vecs[qid])
                        weights.append(conf)
            news_entity_vecs[news_id] = (
                np.average(vecs, axis=0, weights=weights) if vecs else zero
            )
    print(f"  Parsed {len(news_entity_vecs)} news articles from {news_tsv}")
    return news_entity_vecs


def build_news_vecs(sbert_vecs: dict, entity_news_vecs: dict) -> dict:
    zero_sbert  = np.zeros(TOTAL_SBERT_DIM, dtype=np.float32)
    zero_entity = np.zeros(ENTITY_DIM,      dtype=np.float32)
    all_ids = sorted(set(sbert_vecs) | set(entity_news_vecs))
    if USE_ENTITY:
        mat = np.stack([
            np.concatenate([sbert_vecs.get(nid, zero_sbert),
                            entity_news_vecs.get(nid, zero_entity)])
            for nid in all_ids
        ])
    else:
        mat = np.stack([sbert_vecs.get(nid, zero_sbert) for nid in all_ids])

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    print(f"  Built {len(all_ids)} content vectors ({TOTAL_DIM}-dim, pre-normalised)")
    return dict(zip(all_ids, mat))


def build_global_popularity(behaviors_path: str) -> dict:
    pop = defaultdict(int)
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            for nid in cols[3].split():
                pop[nid] += 1
    print(f"  Global popularity: {len(pop)} articles")
    return pop


def _content_scores_raw(history: list[str], candidates: list[str],
                         news_vecs: dict, global_pop: dict) -> list[float]:
    hist_vecs = [news_vecs[h] for h in history if h in news_vecs]
    if not hist_vecs:
        return [float(global_pop.get(nid, 0)) for nid in candidates]

    user_vec = np.mean(hist_vecs, axis=0)
    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return [float(global_pop.get(nid, 0)) for nid in candidates]
    user_vec /= norm

    zero     = np.zeros(TOTAL_DIM, dtype=np.float32)
    cand_mat = np.stack([news_vecs.get(nid, zero) for nid in candidates])
    return list(cand_mat @ user_vec)


# =============================================================================
# Stage 2 – Item-based CF model  (from collaborative_filtering.py)
# =============================================================================

def build_cf_model(behaviors_path: str):
    """
    Returns (article_user_norm, article_to_idx).
    article_user_norm : csr_matrix (n_articles, n_users), L2-normalised rows.
    """
    COLS = ["impression_id", "user_id", "time", "history", "impressions"]
    train = pd.read_csv(behaviors_path, sep="\t", header=None, names=COLS)

    user_clicks: dict[str, set] = defaultdict(set)

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

    user_to_idx    = {u: i for i, u in enumerate(user_clicks)}
    all_articles   = sorted({nid for clicks in user_clicks.values() for nid in clicks})
    article_to_idx = {a: i for i, a in enumerate(all_articles)}

    rows, cols = [], []
    for uid, clicks in user_clicks.items():
        u = user_to_idx[uid]
        for nid in clicks:
            rows.append(u)
            cols.append(article_to_idx[nid])

    interaction = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_to_idx), len(all_articles)),
    )
    print(f"  CF matrix: {interaction.shape[0]} users × {interaction.shape[1]} articles")

    article_user_norm = normalize(interaction.T.tocsr(), norm="l2", axis=1, copy=False)
    return article_user_norm, article_to_idx


def _cf_scores_raw(history: list[str], candidates: list[str],
                   article_user_norm, article_to_idx: dict) -> list[float]:
    hist_idx = [article_to_idx[h] for h in history if h in article_to_idx]
    if not hist_idx:
        return [0.0] * len(candidates)

    hist_vecs = article_user_norm[hist_idx]   # (h, n_users)

    scores = []
    for cand in candidates:
        c_idx = article_to_idx.get(cand)
        if c_idx is None:
            scores.append(0.0)
            continue
        cand_vec = article_user_norm[c_idx]
        sims = np.asarray((cand_vec @ hist_vecs.T).todense()).ravel()
        scores.append(float(sims.mean()))
    return scores


# =============================================================================
# Two-stage hybrid scoring
# =============================================================================

def _top_k_indices(scores: list[float], frac: float) -> list[int]:
    """Indices of the top-frac fraction; always at least 1."""
    k = max(1, math.ceil(len(scores) * frac))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def hybrid_score_fn(
    history: list[str],
    candidates: list[str],
    article_user_norm,
    article_to_idx: dict,
    news_vecs: dict,
    global_pop: dict,
) -> list[float]:
    """
    Stage 1 – content scores all candidates, keeps top TOP_FRAC.
    Stage 2 – CF scores the shortlist; survivors' CF scores are the final output.

    CF scores on binary interaction matrices are always in [0, 1].
    Eliminated candidates receive -1 so they always rank below every survivor,
    including cold-start survivors whose CF score is 0.
    """
    n = len(candidates)

    # ── Stage 1: content filter ───────────────────────────────────────────────
    content_raw = _content_scores_raw(history, candidates, news_vecs, global_pop)

    if any(s != content_raw[0] for s in content_raw):   # at least some variation
        shortlist_local = _top_k_indices(content_raw, TOP_FRAC)
    else:
        # No content signal (all identical) → pass all candidates to CF
        shortlist_local = list(range(n))

    shortlist_ids = [candidates[i] for i in shortlist_local]

    # ── Stage 2: CF final ranking ─────────────────────────────────────────────
    # Eliminated candidates start at -1 (below all CF scores which are >= 0)
    final_scores = [-1.0] * n

    cf_raw = _cf_scores_raw(history, shortlist_ids, article_user_norm, article_to_idx)

    cf_nonzero = any(s > 0 for s in cf_raw)

    for j, orig_idx in enumerate(shortlist_local):
        if cf_nonzero:
            final_scores[orig_idx] = cf_raw[j]
        else:
            # No CF signal → fall back to content score for this candidate
            final_scores[orig_idx] = content_raw[orig_idx]

    return final_scores


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\nContent→CF hybrid config: TOP_FRAC={TOP_FRAC}  USE_ENTITY={USE_ENTITY}")

    for fname in ("sbert_v2_train.npz", "sbert_v2_dev.npz"):
        path = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                "Run 'python generate_sbert_cache.py' first."
            )

    print("\n[1] Building item-based CF model from training behaviors...")
    article_user_norm, article_to_idx = build_cf_model(
        os.path.join(TRAIN_DIR, "behaviors.tsv")
    )

    print("\n[2] Building global popularity fallback...")
    global_pop = build_global_popularity(os.path.join(TRAIN_DIR, "behaviors.tsv"))

    entity_vecs = {}
    if USE_ENTITY:
        print("\n[3] Loading entity embeddings...")
        entity_vecs = load_entity_embeddings(
            os.path.join(DEV_DIR, "entity_embedding.vec")
        )

    print("\n[4] Loading SBERT embeddings...")
    sbert_vecs = load_sbert_cache(os.path.join(CACHE_DIR, "sbert_v2_dev.npz"))

    print("\n[5] Parsing news (confidence-weighted entity vecs)...")
    entity_news_vecs = parse_news(os.path.join(DEV_DIR, "news.tsv"), entity_vecs)

    print("\n[6] Building content vectors...")
    news_vecs = build_news_vecs(sbert_vecs, entity_news_vecs)

    print("\n[7] Evaluating on DEV set...")

    def score_fn(history, candidates,
                 _au=article_user_norm, _ai=article_to_idx,
                 _nv=news_vecs, _pop=global_pop):
        return hybrid_score_fn(history, candidates, _au, _ai, _nv, _pop)

    metrics = evaluate(
        os.path.join(DEV_DIR, "behaviors.tsv"),
        score_fn,
        news_vecs=news_vecs,
        global_pop=global_pop,
    )

    print(f"\nContent→CF Results (DEV)  TOP_FRAC={TOP_FRAC}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
