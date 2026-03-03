from pathlib import Path
import os
import sys
import pickle
from csv import QUOTE_NONE
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # only show error messages

from recommenders.datasets.mind import word_tokenize, download_and_extract_glove, load_glove_matrix

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# -------------------------------------------------------------------
# 1) Root + split folders
# -------------------------------------------------------------------
LOCAL_DATA_ROOT = Path(os.environ.get("MIND_DATA_ROOT", "./smallDataset")).expanduser().resolve()
WORD_EMBEDDING_DIM = int(os.environ.get("WORD_EMBEDDING_DIM", "300"))
UTILS_DIR = LOCAL_DATA_ROOT / "utils"
MIND_TYPE = "small"  # "small" or "large"

train_dir = LOCAL_DATA_ROOT / f"MIND{MIND_TYPE}_train"
valid_dir = LOCAL_DATA_ROOT / f"MIND{MIND_TYPE}_dev"

def must_exist(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)

# -------------------------------------------------------------------
# 2) Core data files (paths)
# -------------------------------------------------------------------
train_news_path      = must_exist(train_dir / "news.tsv",      "train news.tsv")
train_behaviors_path = must_exist(train_dir / "behaviors.tsv", "train behaviors.tsv")
valid_news_path      = must_exist(valid_dir / "news.tsv",      "valid news.tsv")
valid_behaviors_path = must_exist(valid_dir / "behaviors.tsv", "valid behaviors.tsv")

# -------------------------------------------------------------------
# 3) Load TSVs into DataFrames used throughout the notebook
# -------------------------------------------------------------------
NEWS_COLUMNS = [
    "nid",
    "vertical",
    "subvertical",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

def load_tsv(path_str: str, columns):
    return pd.read_table(
        path_str,
        header=None,
        names=columns,
        sep="\t",
        quoting=QUOTE_NONE,
        dtype=object,
        na_filter=False,
    )

train_news_df = load_tsv(train_news_path, NEWS_COLUMNS)
valid_news_df = load_tsv(valid_news_path, NEWS_COLUMNS)
train_behaviors_df = load_tsv(train_behaviors_path, BEHAVIOR_COLUMNS)
valid_behaviors_df = load_tsv(valid_behaviors_path, BEHAVIOR_COLUMNS)

# expose the DataFrames with the variable names used later
train_news_file = train_news_df
valid_news_file = valid_news_df
train_behaviors_file = train_behaviors_df
valid_behaviors_file = valid_behaviors_df

print(f"Loaded {len(train_news_file)} train news rows and {len(valid_news_file)} validation news rows from {LOCAL_DATA_ROOT}")
print(f"Loaded {len(train_behaviors_file)} train behaviors rows and {len(valid_behaviors_file)} validation behaviors rows")

# -------------------------------------------------------------------
# 4) Build a unified "news" table (train + valid) for vocab/category stats
#    (Safe: item text/features only, no labels)
# -------------------------------------------------------------------
news = (
    pd.concat([train_news_file, valid_news_file], ignore_index=True)
    .drop_duplicates(subset="nid")
    .reset_index(drop=True)
)

# Category dictionaries (1-indexed; 0 reserved for UNK if needed later)
news_vertical = news["vertical"].drop_duplicates().reset_index(drop=True)
vert_dict_inv = news_vertical.to_dict()
vert_dict = {v: k + 1 for k, v in vert_dict_inv.items()}

news_subvertical = news["subvertical"].drop_duplicates().reset_index(drop=True)
subvert_dict_inv = news_subvertical.to_dict()
subvert_dict = {v: k + 1 for k, v in subvert_dict_inv.items()}

# -------------------------------------------------------------------
# 5) Tokenize and build vocab from BOTH title + abstract
# -------------------------------------------------------------------
news["title"] = news["title"].apply(word_tokenize)
news["abstract"] = news["abstract"].apply(word_tokenize)

word_cnt_all = Counter()
for i in tqdm(range(len(news)), desc="Counting tokens (title+abstract)"):
    word_cnt_all.update(news.loc[i, "title"])
    word_cnt_all.update(news.loc[i, "abstract"])

# 1-indexed (0 reserved; often used as PAD/UNK depending on downstream)
word_dict = {token: idx + 1 for idx, token in enumerate(word_cnt_all.keys())}

# -------------------------------------------------------------------
# 6) Build TRAIN-ONLY user_dict and map unknown validation users to 0/UNK
# -------------------------------------------------------------------
def _is_valid_identifier(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True

def _build_index(values, start_idx: int = 1):
    index = {}
    next_idx = start_idx
    for value in values:
        if not _is_valid_identifier(value):
            continue
        if value not in index:
            index[value] = next_idx
            next_idx += 1
    return index

# Train-only user dict:
train_user_series = train_behaviors_file["user_id"]
user_dict = _build_index(train_user_series.tolist(), start_idx=1)

# Optional: create mapped user indices for train/valid (unknown -> 0)
# This is useful if you later want a precomputed numeric column.
def map_user_to_index(user_id: str) -> int:
    if not _is_valid_identifier(user_id):
        return 0
    return user_dict.get(user_id, 0)

train_behaviors_file["user_idx"] = train_behaviors_file["user_id"].apply(map_user_to_index)
valid_behaviors_file["user_idx"] = valid_behaviors_file["user_id"].apply(map_user_to_index)

# -------------------------------------------------------------------
# 7) Create & save utility artifacts (dicts + embedding matrix)
# -------------------------------------------------------------------
UTILS_DIR.mkdir(parents=True, exist_ok=True)

word_embeddings_path = UTILS_DIR / f"word_embeddings_{WORD_EMBEDDING_DIM}.npy"
word_dict_path = UTILS_DIR / "word_dict.pkl"
user_dict_path = UTILS_DIR / "user_dict.pkl"
vert_dict_path = UTILS_DIR / "vert_dict.pkl"
subvert_dict_path = UTILS_DIR / "subvert_dict.pkl"

print(f"Saving utility artifacts to {UTILS_DIR} ...")

# Download + build GloVe matrix aligned with word_dict indices
glove_dir = download_and_extract_glove(str(LOCAL_DATA_ROOT))
word_embeddings, covered_words = load_glove_matrix(
    glove_dir, word_dict, WORD_EMBEDDING_DIM
)
word_embeddings = word_embeddings.astype(np.float32, copy=False)

# Save artifacts
np.save(word_embeddings_path, word_embeddings)

with open(word_dict_path, "wb") as f:
    pickle.dump(word_dict, f)

with open(user_dict_path, "wb") as f:
    pickle.dump(user_dict, f)

with open(vert_dict_path, "wb") as f:
    pickle.dump(vert_dict, f)

with open(subvert_dict_path, "wb") as f:
    pickle.dump(subvert_dict, f)

# Coverage stats
coverage_ratio = (len(covered_words) / len(word_dict) * 100.0) if word_dict else 0.0

# Quick shape sanity check (common expectation: len(word_dict)+1 rows if index 0 reserved)
print(f"word_dict size: {len(word_dict)}")
print(f"word_embeddings shape: {word_embeddings.shape}")
print(
    f"Saved: {word_embeddings_path.name}, {word_dict_path.name}, {user_dict_path.name}, "
    f"{vert_dict_path.name}, {subvert_dict_path.name}. "
    f"GloVe coverage: {coverage_ratio:.1f}% of {len(word_dict)} words."
)

# Optional: report how many validation users are unknown under train-only user_dict
valid_unknown = int((valid_behaviors_file["user_idx"] == 0).sum())
valid_total = len(valid_behaviors_file)
print(f"Validation unknown users mapped to 0: {valid_unknown}/{valid_total} ({(valid_unknown/valid_total*100.0 if valid_total else 0.0):.1f}%)")