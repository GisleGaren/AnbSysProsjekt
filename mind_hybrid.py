"""
MIND Hybrid Recommender: Item-Based CF + Content (SBERT + TransE entity)
=========================================================================
Combines two signals per impression:
  - CF score      : item-based collaborative filtering via cosine similarity on a
                    normalised user-article interaction matrix (history + impression clicks)
  - Content score : cosine similarity between user profile and candidate content vectors
                    (title_sbert[768] | abstract_sbert[768] | entity[100] — optional)

Both score lists are min-max normalised per impression before blending so their
scales are comparable regardless of how many candidates are in the impression.

  hybrid = ALPHA * cf_norm + (1 - ALPHA) * content_norm

CF pipeline:
  1. Build binary user-article matrix from history AND impression clicks (label=1)
  2. Transpose to article-user matrix, L2-normalise each article row
  3. At score time: for each candidate, compute cosine sim against all history articles,
     take the mean — articles similar to what the user read score higher

Fallback chain:
  - Candidate not in CF matrix   -> CF score = 0 for that candidate
  - User has no CF history        -> CF scores all 0, weight shifts entirely to content
  - User has no content history   -> content scores all 0, fall back to global popularity

Run generate_sbert_cache.py first to produce the SBERT v2 cache.

Tuning:
  ALPHA         : blend weight for CF (0 = content only, 1 = CF only)
  CF_COMPONENTS : SVD latent dimensions (higher = more capacity, slower)
  USE_ENTITY    : include TransE entity embeddings in content vector
"""

import os
import json
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import pandas as pd
from evaluation import evaluate

# --- Config ------------------------------------------------------------------
ALPHA      = 0.5   # CF weight; (1 - ALPHA) goes to content
USE_ENTITY = False  # set True to append confidence-weighted TransE entity (100-dim)

# --- Paths -------------------------------------------------------------------
BASE      = "smallDataset"
TRAIN_DIR = os.path.join(BASE, "MINDsmall_train")
DEV_DIR   = os.path.join(BASE, "MINDsmall_dev")
CACHE_DIR = os.path.join(BASE, "sbert_cache")

SBERT_DIM       = 768
TOTAL_SBERT_DIM = SBERT_DIM * 2           # title + abstract
ENTITY_DIM      = 100
TOTAL_DIM       = TOTAL_SBERT_DIM + (ENTITY_DIM if USE_ENTITY else 0)


# =============================================================================
# CF model
# =============================================================================

def build_cf_model(behaviors_path):
    """
    Builds an item-based CF model from training behaviors.

    Interaction matrix includes both history clicks and impression clicks
    (label=1) from train — both are legitimate signals with no leakage risk.

    Pipeline:
      binary user-article matrix -> transpose to article-user -> L2-normalise rows
      -> dot products between article rows give cosine similarities

    Returns (article_user_norm, article_to_idx).
    """
    COLS = ["impression_id", "user_id", "time", "history", "impressions"]
    train = pd.read_csv(behaviors_path, sep="\t", header=None, names=COLS)

    user_clicks = defaultdict(set)

    # History clicks
    hist = train[["user_id", "history"]].dropna(subset=["history"])
    for uid, h in zip(hist["user_id"], hist["history"]):
        if isinstance(h, str) and h.strip():
            user_clicks[uid].update(h.split())

    # Impression clicks (label=1) — legitimate training signal
    for uid, impressions in zip(train["user_id"], train["impressions"]):
        if not isinstance(impressions, str):
            continue
        for item in impressions.split():
            if item.endswith("-1"):
                user_clicks[uid].add(item[:-2])

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
        shape=(len(user_to_idx), len(article_to_idx)),
    )
    print(f"  Interaction matrix: {interaction.shape[0]} users x {interaction.shape[1]} articles")

    # Article-user matrix, L2-normalised per row -> dot products = cosine similarities
    article_user_norm = normalize(interaction.T.tocsr(), norm="l2", axis=1, copy=False)

    print(f"  Article-user matrix: {article_user_norm.shape}, L2-normalised")
    return article_user_norm, article_to_idx


def cf_scores(history, candidates, article_user_norm, article_to_idx):
    """Item-based CF: score candidates by mean cosine similarity to history articles."""
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
        sims = np.asarray((cand_vec @ hist_vecs.T).todense()).ravel()
        scores.append(float(sims.mean()))
    return scores


# =============================================================================
# Content model (from mind_entity_recsys)
# =============================================================================

def load_sbert_cache(npz_path):
    """Returns dict: news_id -> np.array (TOTAL_SBERT_DIM,)"""
    data = np.load(npz_path)
    all_vecs = np.concatenate([data["title_vecs"], data["abstract_vecs"]], axis=1)
    embeddings = dict(zip(data["ids"], all_vecs))
    print(f"  Loaded {len(embeddings)} SBERT embeddings (2x{SBERT_DIM}-dim) from {npz_path}")
    return embeddings


def load_entity_embeddings(vec_path):
    entity_vecs = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            entity_vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(entity_vecs)} entity embeddings from {vec_path}")
    return entity_vecs


def parse_news(news_tsv, entity_vecs):
    """Returns dict: news_id -> confidence-weighted mean TransE vector (100,)."""
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


def build_news_vecs(sbert_vecs, entity_news_vecs):
    """Concatenates SBERT + (optional) entity vectors, then L2-normalises each row."""
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

    # Pre-normalise so scoring is just a dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    vecs = dict(zip(all_ids, mat))
    print(f"  Built {len(vecs)} content vectors ({TOTAL_DIM}-dim, pre-normalised)")
    return vecs


def content_scores(history, candidates, news_vecs, global_pop):
    """Content-based scores: dot product against pre-normalised mean user profile."""
    hist_vecs = [news_vecs[h] for h in history if h in news_vecs]
    if not hist_vecs:
        return [float(global_pop.get(nid, 0)) for nid in candidates]

    user_vec = np.mean(hist_vecs, axis=0)
    norm = np.linalg.norm(user_vec)
    if norm == 0:
        return [float(global_pop.get(nid, 0)) for nid in candidates]
    user_vec /= norm

    zero = np.zeros(TOTAL_DIM, dtype=np.float32)
    cand_mat = np.stack([news_vecs.get(nid, zero) for nid in candidates])
    return list(cand_mat @ user_vec)  # single batched dot product


def build_global_popularity(behaviors_path):
    pop = defaultdict(int)
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            for nid in cols[3].split():
                pop[nid] += 1
    print(f"  Popularity: {len(pop)} articles")
    return pop


# =============================================================================
# Hybrid scoring
# =============================================================================

def minmax(scores):
    """Min-max normalise a list of floats to [0, 1]. Returns list unchanged if all equal."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def hybrid_score_fn(history, candidates,
                    article_factors, article_to_idx,
                    news_vecs, global_pop):
    cf  = cf_scores(history, candidates, article_factors, article_to_idx)
    cnt = content_scores(history, candidates, news_vecs, global_pop)

    cf_n  = minmax(cf)
    cnt_n = minmax(cnt)

    return [ALPHA * c + (1 - ALPHA) * s for c, s in zip(cf_n, cnt_n)]


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\nHybrid config: ALPHA={ALPHA}  USE_ENTITY={USE_ENTITY}")

    # Check cache files exist before doing any heavy work
    for fname in ("sbert_v2_dev.npz",):
        path = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                "Run 'python generate_sbert_cache.py' first to generate SBERT embeddings."
            )

    print("\n[0] Building item-based CF model from training behaviors...")
    article_user_norm, article_to_idx = build_cf_model(
        os.path.join(TRAIN_DIR, "behaviors.tsv")
    )

    print("\n[0b] Building global popularity fallback...")
    global_pop = build_global_popularity(os.path.join(TRAIN_DIR, "behaviors.tsv"))

    entity_vecs = {}
    if USE_ENTITY:
        print("\n[0c] Loading entity embeddings...")
        entity_vecs = load_entity_embeddings(
            os.path.join(DEV_DIR, "entity_embedding.vec")
        )

    print("\n[1] Loading SBERT embeddings...")
    sbert_vecs = load_sbert_cache(os.path.join(CACHE_DIR, "sbert_v2_dev.npz"))

    print("\n[2] Parsing news (entity vecs)...")
    entity_news_vecs = parse_news(os.path.join(DEV_DIR, "news.tsv"), entity_vecs)

    print("\n[3] Building content vectors...")
    news_vecs = build_news_vecs(sbert_vecs, entity_news_vecs)

    print("\n[4] Evaluating hybrid on DEV...")
    def score_fn(history, candidates,
                 _au=article_user_norm, _ai=article_to_idx,
                 _nv=news_vecs, _pop=global_pop):
        return hybrid_score_fn(history, candidates, _au, _ai, _nv, _pop)

    metrics = evaluate(os.path.join(DEV_DIR, "behaviors.tsv"), score_fn,
                       news_vecs=news_vecs, global_pop=global_pop)

    print(f"\n  Results on DEV (ALPHA={ALPHA}):")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
