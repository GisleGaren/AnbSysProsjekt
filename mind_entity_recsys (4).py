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
from sklearn.metrics import roc_auc_score

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "smallDataset"
TRAIN_DIR = os.path.join(BASE, "MINDsmall_train")
DEV_DIR   = os.path.join(BASE, "MINDSmall_dev")

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


# ─── 4. Cosine similarity ─────────────────────────────────────────────────────
def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── 5. Evaluation metrics ───────────────────────────────────────────────────
def dcg_at_k(relevance, k):
    relevance = np.array(relevance[:k], dtype=np.float32)
    if relevance.size == 0:
        return 0.0
    gains = relevance / np.log2(np.arange(2, relevance.size + 2))
    return gains.sum()


def ndcg_at_k(relevance, k):
    ideal = sorted(relevance, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(relevance, k) / idcg if idcg > 0 else 0.0


def mrr(relevance):
    for i, r in enumerate(relevance):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(behaviors_path, news_vecs, dim=100):
    """
    Parse behaviors.tsv, score each impression, compute metrics.
    Returns dict of metric -> float.
    """
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    skipped = 0

    with open(behaviors_path, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue

            history_str    = cols[3].strip()
            impressions_str= cols[4].strip()

            history = history_str.split() if history_str else []
            impressions = impressions_str.split()

            if not impressions:
                continue

            # Parse impression list: "N12345-1" or "N12345-0"
            candidates, labels = [], []
            for imp in impressions:
                parts = imp.rsplit("-", 1)
                if len(parts) != 2:
                    continue
                nid, label = parts
                candidates.append(nid)
                labels.append(int(label))

            if sum(labels) == 0 or sum(labels) == len(labels):
                skipped += 1
                continue  # can't compute AUC for degenerate cases

            user_vec = build_user_vector(history, news_vecs, dim)
            scores   = [cosine_sim(user_vec, news_vecs.get(nid, np.zeros(dim)))
                        for nid in candidates]

            # Sort by score descending → ranked list of labels
            ranked = [lbl for _, lbl in sorted(zip(scores, labels), reverse=True)]

            aucs.append(roc_auc_score(labels, scores))
            mrrs.append(mrr(ranked))
            ndcg5s.append(ndcg_at_k(ranked, 5))
            ndcg10s.append(ndcg_at_k(ranked, 10))

    print(f"  Evaluated {len(aucs)} impressions | Skipped (degenerate): {skipped}")
    return {
        "AUC":      np.mean(aucs),
        "MRR":      np.mean(mrrs),
        "nDCG@5":   np.mean(ndcg5s),
        "nDCG@10":  np.mean(ndcg10s),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    dim = 100

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
        metrics = evaluate(behaviors_path, news_vecs, dim)

        print(f"\n  Results on {split}:")
        for k, v in metrics.items():
            print(f"    {k:10s}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
