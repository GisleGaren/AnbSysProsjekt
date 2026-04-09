"""
Generate SBERT v2 cache: category-prefixed title + abstract embeddings.
========================================================================
Output: smallDataset/sbert_cache/sbert_v2_{train,dev}.npz
Keys:   ids, title_vecs, abstract_vecs  (all float32, shape [N, 768])

- title is encoded as "{category} {subcategory}: {title}" so category
  context informs the semantic encoding without adding extra dimensions
- abstract falls back to title text if the abstract field is empty

Usage:
    python generate_sbert_cache.py
"""

import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

BASE      = "smallDataset"
CACHE_DIR = os.path.join(BASE, "sbert_cache")
MODEL     = "all-mpnet-base-v2"   # 768-dim; change if you used a different model

SPLITS = [
    ("train", os.path.join(BASE, "MINDsmall_train", "news.tsv"), "sbert_v2_train.npz"),
    ("dev",   os.path.join(BASE, "MINDsmall_dev",   "news.tsv"), "sbert_v2_dev.npz"),
]


def parse_news_texts(news_tsv):
    """Returns list of (news_id, title_text, abstract_text)."""
    records = []
    with open(news_tsv, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            news_id  = cols[0]
            category = cols[1].strip()
            subcat   = cols[2].strip()
            title    = cols[3].strip()
            abstract = cols[4].strip()
            # Bake category context directly into the title encoding
            title_text = f"{category} {subcat}: {title}"
            abstract_text = abstract if abstract else title_text
            records.append((news_id, title_text, abstract_text))
    return records


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SBERT model: {MODEL}  (device: {device})")
    model = SentenceTransformer(MODEL, device=device)

    encode_kwargs = dict(batch_size=256, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=False)

    for split, news_tsv, cache_file in SPLITS:
        print(f"\n=== {split.upper()} ===")
        records = parse_news_texts(news_tsv)
        print(f"  {len(records)} articles")

        ids       = [r[0] for r in records]
        titles    = [r[1] for r in records]
        abstracts = [r[2] for r in records]

        print("  Encoding titles (with category prefix)...")
        title_vecs = model.encode(titles, **encode_kwargs)

        print("  Encoding abstracts...")
        abstract_vecs = model.encode(abstracts, **encode_kwargs)

        out_path = os.path.join(CACHE_DIR, cache_file)
        np.savez_compressed(
            out_path,
            ids=np.array(ids),
            title_vecs=title_vecs.astype(np.float32),
            abstract_vecs=abstract_vecs.astype(np.float32),
        )
        print(f"  Saved -> {out_path}")
        print(f"  Shapes: title={title_vecs.shape}  abstract={abstract_vecs.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
