"""
MIND Hybrid Recommender: SBERT (768) + TransE Entity (100) + Category one-hot
==============================================================================
News vector = concat(SBERT, mean_entity_TransE, one_hot_category, one_hot_subcategory)
Category vocab is built from training news.tsv only (no leakage).
Unknown categories in dev get a zero vector for that slice.

Pipeline:
  1. Load sbert_train.npz / sbert_dev.npz  -> news_id -> 768-dim vector
  2. Load entity_embedding.vec             -> wikidata_id -> 100-dim vector
  3. Parse news.tsv entities               -> news_id -> mean 100-dim entity vec
  4. Build category vocab from train       -> category/subcategory -> index
  5. One-hot encode category + subcategory -> news_id -> (n_cat + n_subcat)-dim vec
  6. Concatenate all parts                 -> news_id -> full-dim vector
  7. User profile = mean of clicked news vectors
  8. Score candidates by cosine similarity
  9. Evaluate with AUC, MRR, nDCG@5, nDCG@10

Expected folder layout (MIND Small):
  smallDataset/
    sbert_cache/
      sbert_train.npz
      sbert_dev.npz
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

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = "smallDataset"
TRAIN_DIR  = os.path.join(BASE, "MINDsmall_train")
DEV_DIR    = os.path.join(BASE, "MINDsmall_dev")
CACHE_DIR  = os.path.join(BASE, "sbert_cache")

SBERT_DIM  = 768
ENTITY_DIM = 100

# ─── 1. Load cached SBERT embeddings ─────────────────────────────────────────
def load_sbert_cache(npz_path):
    """Returns dict: news_id -> np.array (768,)"""
    data = np.load(npz_path)
    ids  = data["ids"]
    vecs = data["vecs"]
    embeddings = {nid: vecs[i] for i, nid in enumerate(ids)}
    print(f"  Loaded {len(embeddings)} SBERT embeddings from {npz_path}")
    return embeddings

# ─── 2. Load TransE entity embeddings ────────────────────────────────────────
def load_entity_embeddings(vec_path):
    """Returns dict: wikidata_id -> np.array (100,)"""
    entity_vecs = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qid = parts[0]
            entity_vecs[qid] = np.array(parts[1:], dtype=np.float32)
    print(f"  Loaded {len(entity_vecs)} entity embeddings from {vec_path}")
    return entity_vecs

# ─── 3. Build per-news entity vectors + categories from news.tsv ─────────────
def parse_news(news_tsv, entity_vecs):
    """
    Returns:
      news_entity_vecs: news_id -> mean TransE vector (100,)
      news_categories:  news_id -> (category, subcategory)
    """
    news_entity_vecs = {}
    news_categories  = {}
    zero = np.zeros(ENTITY_DIM, dtype=np.float32)
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 7:
                continue
            news_id  = cols[0]
            category = cols[1]
            subcat   = cols[2]
            news_categories[news_id] = (category, subcat)

            vecs = []
            for col_idx in (6, 7):
                if col_idx >= len(cols):
                    continue
                try:
                    entities = json.loads(cols[col_idx])
                except (json.JSONDecodeError, ValueError):
                    continue
                for ent in entities:
                    qid = ent.get("WikidataId", "")
                    if qid in entity_vecs:
                        vecs.append(entity_vecs[qid])
            news_entity_vecs[news_id] = np.mean(vecs, axis=0) if vecs else zero
    print(f"  Parsed {len(news_entity_vecs)} news articles from {news_tsv}")
    return news_entity_vecs, news_categories

# ─── 4. Build category vocabularies from training news only ──────────────────
def build_category_vocab(news_categories):
    """
    Returns cat_vocab (category -> index) and subcat_vocab (subcategory -> index).
    Built from training data only so dev unknowns map to zero vectors.
    """
    cats    = sorted({cat for cat, _   in news_categories.values()})
    subcats = sorted({sub for _,   sub in news_categories.values()})
    cat_vocab    = {c: i for i, c in enumerate(cats)}
    subcat_vocab = {s: i for i, s in enumerate(subcats)}
    print(f"  Category vocab: {len(cat_vocab)} categories, {len(subcat_vocab)} subcategories")
    return cat_vocab, subcat_vocab

# ─── 5. Build one-hot category vectors ───────────────────────────────────────
def build_category_vecs(news_categories, cat_vocab, subcat_vocab):
    """
    Returns dict: news_id -> np.array (n_cat + n_subcat,).
    Unknown categories (e.g. in dev) get zeros for that slice.
    """
    n_cat    = len(cat_vocab)
    n_subcat = len(subcat_vocab)
    cat_vecs = {}
    for news_id, (cat, subcat) in news_categories.items():
        vec = np.zeros(n_cat + n_subcat, dtype=np.float32)
        if cat in cat_vocab:
            vec[cat_vocab[cat]] = 1.0
        if subcat in subcat_vocab:
            vec[n_cat + subcat_vocab[subcat]] = 1.0
        cat_vecs[news_id] = vec
    return cat_vecs

# ─── 6. Concatenate all vectors ───────────────────────────────────────────────
def build_hybrid_vecs(sbert_vecs, entity_news_vecs, cat_vecs, cat_dim):
    """Returns dict: news_id -> np.array (768 + 100 + cat_dim,)"""
    zero_sbert  = np.zeros(SBERT_DIM,  dtype=np.float32)
    zero_entity = np.zeros(ENTITY_DIM, dtype=np.float32)
    zero_cat    = np.zeros(cat_dim,    dtype=np.float32)
    all_ids = set(sbert_vecs) | set(entity_news_vecs) | set(cat_vecs)
    hybrid = {}
    for nid in all_ids:
        s = sbert_vecs.get(nid, zero_sbert)
        e = entity_news_vecs.get(nid, zero_entity)
        c = cat_vecs.get(nid, zero_cat)
        hybrid[nid] = np.concatenate([s, e, c])
    total_dim = SBERT_DIM + ENTITY_DIM + cat_dim
    print(f"  Built {len(hybrid)} hybrid vectors ({total_dim}-dim)")
    return hybrid, total_dim

# ─── 7. Build user vectors ────────────────────────────────────────────────────
def build_user_vector(history_ids, news_vecs, dim):
    """Average of clicked news vectors. Returns zero vec if history is empty."""
    vecs = [news_vecs[nid] for nid in history_ids if nid in news_vecs]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

# ─── 8. Global popularity fallback ───────────────────────────────────────────
def build_global_popularity(behaviors_path):
    """Returns dict: news_id -> click_count from the training set."""
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

# ─── 9. Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n[0] Building Global Popularity from Training data...")
    global_pop = build_global_popularity(os.path.join(TRAIN_DIR, "behaviors.tsv"))

    print("\n[0b] Loading entity embeddings (shared across splits)...")
    entity_vecs = load_entity_embeddings(os.path.join(DEV_DIR, "entity_embedding.vec"))

    print("\n[0c] Building category vocab from training news...")
    _, train_categories = parse_news(os.path.join(TRAIN_DIR, "news.tsv"), entity_vecs)
    cat_vocab, subcat_vocab = build_category_vocab(train_categories)
    cat_dim = len(cat_vocab) + len(subcat_vocab)

    for split, data_dir, cache_file in [
        ("TRAIN", TRAIN_DIR, "sbert_train.npz"),
        ("DEV",   DEV_DIR,   "sbert_dev.npz"),
    ]:
        print(f"\n{'='*50}")
        print(f" Split: {split}  ({data_dir})")
        print(f"{'='*50}")

        print("\n[1] Loading SBERT embeddings...")
        sbert_vecs = load_sbert_cache(os.path.join(CACHE_DIR, cache_file))

        print("\n[2] Parsing news (entity vecs + categories)...")
        entity_news_vecs, news_categories = parse_news(
            os.path.join(data_dir, "news.tsv"), entity_vecs
        )

        print("\n[3] One-hot encoding categories...")
        cat_vecs = build_category_vecs(news_categories, cat_vocab, subcat_vocab)

        print("\n[4] Concatenating SBERT + entity + category vectors...")
        news_vecs, total_dim = build_hybrid_vecs(sbert_vecs, entity_news_vecs, cat_vecs, cat_dim)

        print("\n[5] Evaluating...")
        def score_fn(history, candidates, _news_vecs=news_vecs, _pop=global_pop, _dim=total_dim):
            user_vec = build_user_vector(history, _news_vecs, _dim)
            if np.all(user_vec == 0):
                return [_pop.get(nid, 0) for nid in candidates]
            zero = np.zeros(_dim, dtype=np.float32)
            return [cosine_sim(user_vec, _news_vecs.get(nid, zero)) for nid in candidates]

        metrics = evaluate(os.path.join(data_dir, "behaviors.tsv"), score_fn)

        print(f"\n  Results on {split}:")
        for k, v in metrics.items():
            print(f"    {k:10s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
