"""
MIND Hybrid Recommender v2: Separate SBERT fields + Confidence-weighted TransE entities
========================================================================================
News vector = concat(title_sbert[768], abstract_sbert[768], cat_sbert[768], entity[100])
  - title_sbert    : SBERT embedding of article title alone
  - abstract_sbert : SBERT embedding of article abstract (falls back to title if empty)
  - cat_sbert      : SBERT embedding of "category: {cat}, subcategory: {subcat}"
  - entity         : confidence-weighted mean of TransE entity embeddings (100-dim)

Category and subcategory are encoded semantically through SBERT rather than as
sparse one-hot vectors, so similar categories share representation space.
No vocabulary leakage risk since the category text is self-contained.

Run generate_sbert_cache.py first to produce:
  smallDataset/sbert_cache/sbert_v2_train.npz
  smallDataset/sbert_cache/sbert_v2_dev.npz

Pipeline:
  1. Load sbert_v2_{split}.npz  -> news_id -> 2304-dim (3 x 768)
  2. Load entity_embedding.vec  -> wikidata_id -> 100-dim vector
  3. Parse news.tsv entities    -> news_id -> confidence-weighted mean 100-dim vec
  4. Concatenate                -> news_id -> 2404-dim vector
  5. User profile = mean of clicked news vectors
  6. Score candidates by cosine similarity; fallback to global popularity
  7. Evaluate with AUC, MRR, nDCG@5, nDCG@10

Expected folder layout (MIND Small):
  smallDataset/
    sbert_cache/
      sbert_v2_train.npz
      sbert_v2_dev.npz
    MINDsmall_train/
      behaviors.tsv
      news.tsv
      entity_embedding.vec
    MINDsmall_dev/
      behaviors.tsv
      news.tsv
      entity_embedding.vec
"""

import os
import json
import numpy as np
from collections import defaultdict
from evaluation import evaluate

# --- Paths -------------------------------------------------------------------
BASE       = "smallDataset"
TRAIN_DIR  = os.path.join(BASE, "MINDsmall_train")
DEV_DIR    = os.path.join(BASE, "MINDsmall_dev")
CACHE_DIR  = os.path.join(BASE, "sbert_cache")

USE_ENTITY = False   # set False to run SBERT-only (no TransE entity concat)

SBERT_DIM       = 768
SBERT_FIELDS    = 2                          # title (with category prefix), abstract
TOTAL_SBERT_DIM = SBERT_DIM * SBERT_FIELDS  # 1536
ENTITY_DIM      = 100
TOTAL_DIM       = TOTAL_SBERT_DIM + (ENTITY_DIM if USE_ENTITY else 0)


# --- 1. Load v2 SBERT cache --------------------------------------------------
def load_sbert_cache(npz_path):
    """
    Returns dict: news_id -> np.array (2304,)  [title | abstract | category]
    Cache must have keys: ids, title_vecs, abstract_vecs, cat_vecs
    """
    data     = np.load(npz_path)
    all_vecs = np.concatenate([data["title_vecs"], data["abstract_vecs"]], axis=1)
    embeddings = dict(zip(data["ids"], all_vecs))
    print(f"  Loaded {len(embeddings)} SBERT embeddings (2x{SBERT_DIM}-dim) from {npz_path}")
    return embeddings


# --- 2. Load TransE entity embeddings ----------------------------------------
def load_entity_embeddings(vec_path):
    """Returns dict: wikidata_id -> np.array (100,)"""
    entity_vecs = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            entity_vecs[parts[0]] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(entity_vecs)} entity embeddings from {vec_path}")
    return entity_vecs


# --- 3. Build per-news confidence-weighted entity vectors --------------------
def parse_news(news_tsv, entity_vecs):
    """
    Returns dict: news_id -> confidence-weighted mean TransE vector (100,).
    Entity confidence scores from both title entities (col 6) and abstract
    entities (col 7) are used as weights in the mean aggregation.
    """
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
            if vecs:
                news_entity_vecs[news_id] = np.average(vecs, axis=0, weights=weights)
            else:
                news_entity_vecs[news_id] = zero
    print(f"  Parsed {len(news_entity_vecs)} news articles from {news_tsv}")
    return news_entity_vecs


# --- 4. Concatenate SBERT (+ optional entity) vectors, pre-normalised -------
def build_hybrid_vecs(sbert_vecs, entity_news_vecs):
    """Returns dict: news_id -> L2-normalised np.array (TOTAL_DIM,)"""
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

    hybrid = dict(zip(all_ids, mat))
    print(f"  Built {len(hybrid)} vectors ({TOTAL_DIM}-dim, pre-normalised, entity={'on' if USE_ENTITY else 'off'})")
    return hybrid


# --- 6. Global popularity fallback -------------------------------------------
def build_global_popularity(behaviors_path):
    """Returns dict: news_id -> click_count from training set."""
    pop_counts = defaultdict(int)
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            for nid in cols[3].split():
                pop_counts[nid] += 1
    print(f"  Calculated popularity for {len(pop_counts)} unique news items.")
    return pop_counts




# --- Main --------------------------------------------------------------------
def main():
    for fname in ("sbert_v2_train.npz", "sbert_v2_dev.npz"):
        path = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                "Run 'python generate_sbert_cache.py' first to generate SBERT embeddings."
            )

    print("\n[0] Building Global Popularity from Training data...")
    global_pop = build_global_popularity(os.path.join(TRAIN_DIR, "behaviors.tsv"))

    entity_vecs = {}
    if USE_ENTITY:
        print("\n[0b] Loading entity embeddings (shared across splits)...")
        entity_vecs = load_entity_embeddings(os.path.join(DEV_DIR, "entity_embedding.vec"))

    for split, data_dir, cache_file in [
        ("TRAIN", TRAIN_DIR, "sbert_v2_train.npz"),
        ("DEV",   DEV_DIR,   "sbert_v2_dev.npz"),
    ]:
        print(f"\n{'='*50}")
        print(f" Split: {split}  ({data_dir})")
        print(f"{'='*50}")

        print("\n[1] Loading SBERT embeddings (title + abstract + category)...")
        sbert_vecs = load_sbert_cache(os.path.join(CACHE_DIR, cache_file))

        print("\n[2] Parsing news (confidence-weighted entity vecs)...")
        entity_news_vecs = parse_news(os.path.join(data_dir, "news.tsv"), entity_vecs)

        print("\n[3] Concatenating SBERT + entity vectors...")
        news_vecs = build_hybrid_vecs(sbert_vecs, entity_news_vecs)

        print("\n[4] Evaluating...")
        def score_fn(history, candidates, _news_vecs=news_vecs, _pop=global_pop):
            hist_vecs = [_news_vecs[h] for h in history if h in _news_vecs]
            if not hist_vecs:
                return [float(_pop.get(nid, 0)) for nid in candidates]
            user_vec = np.mean(hist_vecs, axis=0)
            norm = np.linalg.norm(user_vec)
            if norm == 0:
                return [float(_pop.get(nid, 0)) for nid in candidates]
            user_vec /= norm
            zero = np.zeros(TOTAL_DIM, dtype=np.float32)
            cand_mat = np.stack([_news_vecs.get(nid, zero) for nid in candidates])
            return list(cand_mat @ user_vec)

        metrics = evaluate(os.path.join(data_dir, "behaviors.tsv"), score_fn,
                           news_vecs=news_vecs, global_pop=global_pop)

        print(f"\n  Results on {split}:")
        for k, v in metrics.items():
            print(f"    {k:10s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
