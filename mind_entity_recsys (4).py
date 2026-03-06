"""
MIND Entity Embedding Recommender System
=========================================
Uses only entity embeddings (TransE, 100-dim) from the MIND dataset.

Pipeline:
  1. Load entity_embedding.vec
  2. For each news article, aggregate entity embeddings from title+abstract → news vector
  3. For each user, average their clicked news vectors → user vector
  4. Score candidates by cosine similarity (user vec vs candidate vec)
  5. Evaluate with AUC, MRR, nDCG@5, nDCG@10 (standard MIND metrics)

Expected folder layout (MIND Small):
  smallDataset/
    MINDsmall_train/
      news.tsv
      behaviors.tsv
      entity_embedding.vec
    MINDSmall_dev/
      news.tsv
      behaviors.tsv
      entity_embedding.vec
"""

import json
import os
import numpy as np
from collections import defaultdict
from evaluation import evaluate

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "smallDataset"
TRAIN_DIR = os.path.join(BASE, "MINDsmall_train")
DEV_DIR   = os.path.join(BASE, "MINDsmall_dev")

# ─── 1. Load entity embeddings ────────────────────────────────────────────────
def load_entity_embeddings(path):
    """Returns dict: wikidata_id -> np.array (100,)"""
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split() # Splits by any whitespace (tabs or spaces)
            if len(parts) != 101:        # 1 ID + 100 dimensions
                continue
            entity_id = parts[0]
            # Take everything after the ID as the vector
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[entity_id] = vec
    print(f"  Loaded {len(embeddings)} entity embeddings from {path}")
    return embeddings


# ─── 2. Build news vectors from entity embeddings ─────────────────────────────
def parse_entities(entity_str):
    """Parse the JSON entity field from news.tsv, return list of WikidataIDs."""
    if not entity_str or entity_str.strip() in ("", "[]"):
        return []
    try:
        entities = json.loads(entity_str)
        return [e["WikidataId"] for e in entities if "WikidataId" in e]
    except (json.JSONDecodeError, KeyError):
        return []


def build_news_vectors(news_path, entity_embeddings, dim=100):
    """
    Returns dict: news_id -> np.array (dim,)
    News with no entity coverage get a zero vector.
    """
    news_vecs = {}
    no_coverage = 0

    # Diagnostic: show sample entity IDs from both sides
    emb_sample = list(entity_embeddings.keys())[:5]
    print(f"  Sample embedding IDs : {emb_sample}")

    news_entity_sample = []
    all_news_entity_ids = set()

    with open(news_path, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 8:
                continue
            news_id      = cols[0]
            title_ents   = parse_entities(cols[6])
            abstract_ents= parse_entities(cols[7])
            all_ents     = list(set(title_ents + abstract_ents))
            all_news_entity_ids.update(all_ents)

            if len(news_entity_sample) < 5 and all_ents:
                news_entity_sample.extend(all_ents[:2])

            vecs = [entity_embeddings[e] for e in all_ents
                    if e in entity_embeddings and entity_embeddings[e].shape == (dim,)]
            if vecs:
                news_vecs[news_id] = np.mean(vecs, axis=0)
            else:
                news_vecs[news_id] = np.zeros(dim, dtype=np.float32)
                no_coverage += 1

    overlap = all_news_entity_ids & set(entity_embeddings.keys())
    print(f"  Sample news entity IDs: {news_entity_sample[:5]}")
    print(f"  Unique entity IDs in news.tsv : {len(all_news_entity_ids)}")
    print(f"  Overlap with embedding vocab  : {len(overlap)} ({100*len(overlap)/max(len(all_news_entity_ids),1):.1f}%)")
    print(f"  Built {len(news_vecs)} news vectors | No entity coverage: {no_coverage} "
          f"({100*no_coverage/max(len(news_vecs),1):.1f}%)")
    return news_vecs


# ─── 3. Build user vectors ────────────────────────────────────────────────────
def build_user_vector(history_ids, news_vecs, dim=100):
    """Average of clicked news vectors. Returns zero vec if history is empty."""
    vecs = [news_vecs[nid] for nid in history_ids
            if nid in news_vecs and news_vecs[nid].shape == (dim,)]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def build_global_popularity(behaviors_path):
    """Returns dict: news_id -> click_count from the training set."""
    pop_counts = defaultdict(int)
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 4: continue
            
            # History is column index 3 (0-indexed)
            history = cols[3].split()
            for nid in history:
                pop_counts[nid] += 1
    print(f"  Calculated popularity for {len(pop_counts)} unique news items.")
    return pop_counts

# ─── 4. Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    dim = 100

    train_behaviors = os.path.join(TRAIN_DIR, "behaviors.tsv")
    print("\n[0] Building Global Popularity from Training data...")
    global_pop = build_global_popularity(train_behaviors)

    for split, data_dir in [("TRAIN", TRAIN_DIR), ("DEV", DEV_DIR)]:
        print(f"\n{'='*50}")
        print(f" Split: {split}  ({data_dir})")
        print(f"{'='*50}")

        # Always use train entity embeddings — broader vocab coverage
        entity_emb_path = os.path.join(TRAIN_DIR, "entity_embedding.vec")
        news_path       = os.path.join(data_dir, "news.tsv")
        behaviors_path  = os.path.join(data_dir, "behaviors.tsv")

        print("\n[1] Loading entity embeddings...")
        entity_embeddings = load_entity_embeddings(entity_emb_path)

        print("\n[2] Building news vectors...")
        news_vecs = build_news_vectors(news_path, entity_embeddings, dim)

        print("\n[3] Evaluating...")
        def score_fn(history, candidates, _news_vecs=news_vecs, _pop=global_pop, _dim=dim):
            user_vec = build_user_vector(history, _news_vecs, _dim)
            if np.all(user_vec == 0):
                return [_pop.get(nid, 0) for nid in candidates]
            return [cosine_sim(user_vec, _news_vecs.get(nid, np.zeros(_dim))) for nid in candidates]

        metrics = evaluate(behaviors_path, score_fn)

        print(f"\n  Results on {split}:")
        for k, v in metrics.items():
            print(f"    {k:10s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
