#!/usr/bin/env python3
"""
Blended recommender for the MIND dataset.

What this file does
-------------------
1. Loads MIND train/dev behaviors and news metadata
2. Trains a PySpark ALS collaborative filtering model on train impressions
3. Builds a popularity model from train CTRs (with Bayesian smoothing)
4. Builds a simple content model from vertical/subvertical history
5. Blends ALS + content + popularity using a smooth maturity function
6. Evaluates on dev impressions with:
   - nDCG@5
   - nDCG@10
   - Hit@5
   - Hit@10
   - MRR
   - AUC

Usage
-----
python blended_mind_recommender.py --data-root ../smallDataset

Expected layout
---------------
<data-root>/MINDsmall_train/news.tsv
<data-root>/MINDsmall_train/behaviors.tsv
<data-root>/MINDsmall_dev/news.tsv
<data-root>/MINDsmall_dev/behaviors.tsv

Notes
-----
- This is designed to be "drag and drop" and easy to modify.
- The content model is intentionally simple and interpretable.
- The blending is continuous, not hard-switched.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from csv import QUOTE_NONE

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


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


@dataclass
class ImpressionEval:
    labels: List[int]
    scores: List[float]


def load_tsv(path_str: str, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_table(
        path_str,
        header=None,
        names=columns,
        sep="\t",
        quoting=QUOTE_NONE,
        dtype=object,
        na_filter=False,
    )


def parse_impression_tokens(raw_impressions: str) -> List[Tuple[str, int]]:
    tokens: List[Tuple[str, int]] = []
    if not isinstance(raw_impressions, str) or not raw_impressions.strip():
        return tokens
    for token in raw_impressions.split():
        if "-" not in token:
            continue
        nid, label = token.rsplit("-", 1)
        if not nid:
            continue
        try:
            clicked = int(label)
        except ValueError:
            continue
        tokens.append((nid, 1 if clicked > 0 else 0))
    return tokens


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Tuple[str, str, float]] = []
    for row in df.itertuples(index=False):
        for nid, clicked in parse_impression_tokens(row.impressions):
            rows.append((row.user_id, nid, float(clicked)))
    return pd.DataFrame(rows, columns=["user_id", "nid", "label"])


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def minmax_scale(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def roc_auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        pos_in_group = sum(label for _, label in pairs[i:j])
        rank_sum_pos += avg_rank * pos_in_group
        rank += j - i
        i = j

    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def ndcg_at_k(labels_sorted_by_score: Sequence[int], k: int) -> float:
    dcg = 0.0
    for idx, label in enumerate(labels_sorted_by_score[:k], start=1):
        if label:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_positives = min(sum(labels_sorted_by_score), k)
    if ideal_positives == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_positives + 1))
    return dcg / idcg


def hit_at_k(labels_sorted_by_score: Sequence[int], k: int) -> float:
    return 1.0 if any(labels_sorted_by_score[:k]) else 0.0


def mrr(labels_sorted_by_score: Sequence[int]) -> float:
    for idx, label in enumerate(labels_sorted_by_score, start=1):
        if label:
            return 1.0 / idx
    return 0.0


def evaluate_impressions(eval_rows: Sequence[ImpressionEval]) -> Dict[str, float]:
    ndcg5: List[float] = []
    ndcg10: List[float] = []
    hit5: List[float] = []
    hit10: List[float] = []
    mrrs: List[float] = []
    aucs: List[float] = []

    for row in eval_rows:
        labels = row.labels
        scores = row.scores
        if not labels or len(labels) != len(scores):
            continue

        ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        ranked_labels = [label for _, label in ranked]

        ndcg5.append(ndcg_at_k(ranked_labels, 5))
        ndcg10.append(ndcg_at_k(ranked_labels, 10))
        hit5.append(hit_at_k(ranked_labels, 5))
        hit10.append(hit_at_k(ranked_labels, 10))
        mrrs.append(mrr(ranked_labels))

        auc = roc_auc_from_scores(labels, scores)
        if not math.isnan(auc):
            aucs.append(auc)

    return {
        "impressions_evaluated": float(len(eval_rows)),
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "ndcg@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
        "hit@5": float(np.mean(hit5)) if hit5 else 0.0,
        "hit@10": float(np.mean(hit10)) if hit10 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "auc": float(np.mean(aucs)) if aucs else 0.0,
    }


class BlendedMindRecommender:
    def __init__(self, data_root: Path, rank: int = 32, reg_param: float = 0.05, max_iter: int = 10):
        self.data_root = data_root
        self.rank = rank
        self.reg_param = reg_param
        self.max_iter = max_iter

        self.spark: SparkSession | None = None
        self.news_df: pd.DataFrame | None = None
        self.train_behaviors_df: pd.DataFrame | None = None
        self.valid_behaviors_df: pd.DataFrame | None = None

        self.news_feature_map: Dict[str, Tuple[str, str]] = {}
        self.popularity_score: Dict[str, float] = {}
        self.user_history_len: Dict[str, int] = {}
        self.user_vertical_prefs: Dict[str, Counter] = {}
        self.user_subvertical_prefs: Dict[str, Counter] = {}

        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.user_factors: Dict[int, np.ndarray] = {}
        self.item_factors: Dict[int, np.ndarray] = {}

    def load(self) -> None:
        train_dir = self.data_root / "MINDsmall_train"
        valid_dir = self.data_root / "MINDsmall_dev"

        self.news_df = load_tsv(str(train_dir / "news.tsv"), NEWS_COLUMNS)
        # Dev news rows can contain news not in train. Merge them in.
        valid_news_df = load_tsv(str(valid_dir / "news.tsv"), NEWS_COLUMNS)
        self.news_df = (
            pd.concat([self.news_df, valid_news_df], ignore_index=True)
            .drop_duplicates(subset=["nid"], keep="first")
            .fillna("")
        )

        self.train_behaviors_df = load_tsv(str(train_dir / "behaviors.tsv"), BEHAVIOR_COLUMNS)
        self.valid_behaviors_df = load_tsv(str(valid_dir / "behaviors.tsv"), BEHAVIOR_COLUMNS)

        self.news_feature_map = {
            row.nid: (
                (row.vertical or "").strip().lower(),
                (row.subvertical or "").strip().lower(),
            )
            for row in self.news_df.itertuples(index=False)
        }

    def _build_popularity(self) -> None:
        assert self.train_behaviors_df is not None
        click_counter = Counter()
        impression_counter = Counter()
        for impression_str in self.train_behaviors_df["impressions"]:
            for nid, clicked in parse_impression_tokens(impression_str):
                impression_counter[nid] += 1
                click_counter[nid] += clicked

        if not impression_counter:
            self.popularity_score = {}
            return

        global_ctr = (sum(click_counter.values()) + 1.0) / (sum(impression_counter.values()) + 2.0)
        prior_strength = 20.0

        scores: Dict[str, float] = {}
        for nid, imps in impression_counter.items():
            clicks = click_counter.get(nid, 0)
            bayes_ctr = (clicks + prior_strength * global_ctr) / (imps + prior_strength)
            scores[nid] = float(bayes_ctr)

        # Default score for unseen items.
        scores["__DEFAULT__"] = float(global_ctr)
        self.popularity_score = scores

    def _build_content_profiles(self) -> None:
        assert self.train_behaviors_df is not None
        vertical_prefs: Dict[str, Counter] = defaultdict(Counter)
        subvertical_prefs: Dict[str, Counter] = defaultdict(Counter)
        history_len: Dict[str, int] = defaultdict(int)

        for row in self.train_behaviors_df.itertuples(index=False):
            uid = row.user_id

            history_items = []
            if isinstance(row.history, str) and row.history.strip():
                history_items.extend(row.history.split())

            # Also let clicked train impressions strengthen the profile.
            for nid, clicked in parse_impression_tokens(row.impressions):
                if clicked:
                    history_items.append(nid)

            for nid in history_items:
                vertical, subvertical = self.news_feature_map.get(nid, ("", ""))
                if vertical:
                    vertical_prefs[uid][vertical] += 1
                if subvertical:
                    subvertical_prefs[uid][subvertical] += 1
                history_len[uid] += 1

        self.user_vertical_prefs = dict(vertical_prefs)
        self.user_subvertical_prefs = dict(subvertical_prefs)
        self.user_history_len = dict(history_len)

    def _start_spark(self) -> None:
        self.spark = SparkSession.builder.appName("mind-blended-recommender").getOrCreate()

    def _train_als(self) -> None:
        assert self.spark is not None
        assert self.train_behaviors_df is not None

        train_interactions_pd = build_interactions(self.train_behaviors_df)
        train_spark_df = self.spark.createDataFrame(train_interactions_pd)

        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip").fit(train_spark_df)
        item_indexer = StringIndexer(inputCol="nid", outputCol="item_idx", handleInvalid="skip").fit(train_spark_df)

        indexed_df = item_indexer.transform(user_indexer.transform(train_spark_df))
        ratings_df = indexed_df.select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            F.col("label").cast("float").alias("label"),
        )

        als = ALS(
            rank=self.rank,
            maxIter=self.max_iter,
            regParam=self.reg_param,
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="label",
            implicitPrefs=False,
            nonnegative=True,
            coldStartStrategy="drop",
        )
        model = als.fit(ratings_df)

        # Build raw id -> index maps from the StringIndexer labels.
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_indexer.labels)}
        self.item_to_idx = {nid: idx for idx, nid in enumerate(item_indexer.labels)}

        user_factor_rows = model.userFactors.collect()
        item_factor_rows = model.itemFactors.collect()
        self.user_factors = {int(row["id"]): np.array(row["features"], dtype=np.float32) for row in user_factor_rows}
        self.item_factors = {int(row["id"]): np.array(row["features"], dtype=np.float32) for row in item_factor_rows}

    def fit(self) -> None:
        self.load()
        self._build_popularity()
        self._build_content_profiles()
        self._start_spark()
        self._train_als()

    def maturity_weights(self, user_id: str) -> Tuple[float, float, float]:
        """
        Returns weights for (als, content, popularity).

        Design choice:
        - no history      -> popularity dominates
        - little history  -> content dominates
        - more history    -> ALS smoothly takes over
        """
        n = self.user_history_len.get(user_id, 0)

        # Smooth ALS ramp-up. Around 8-10 interactions it becomes dominant.
        w_als = min(0.85, math.log1p(n) / math.log(10.0)) if n > 0 else 0.0
        residual = max(0.0, 1.0 - w_als)

        if n <= 0:
            w_content = 0.0
            w_pop = 1.0
        elif n <= 2:
            w_content = 0.75 * residual
            w_pop = residual - w_content
        else:
            w_content = 0.60 * residual
            w_pop = residual - w_content

        total = w_als + w_content + w_pop
        if total <= 0:
            return 0.0, 0.0, 1.0
        return w_als / total, w_content / total, w_pop / total

    def als_score_for(self, user_id: str, nid: str) -> float:
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(nid)
        if user_idx is None or item_idx is None:
            return 0.0
        uf = self.user_factors.get(user_idx)
        vf = self.item_factors.get(item_idx)
        if uf is None or vf is None:
            return 0.0
        return float(np.dot(uf, vf))

    def popularity_score_for(self, nid: str) -> float:
        return float(self.popularity_score.get(nid, self.popularity_score.get("__DEFAULT__", 0.0)))

    def content_score_for(self, user_id: str, nid: str) -> float:
        vertical, subvertical = self.news_feature_map.get(nid, ("", ""))
        v_counter = self.user_vertical_prefs.get(user_id)
        sv_counter = self.user_subvertical_prefs.get(user_id)
        if not v_counter and not sv_counter:
            return 0.0

        vertical_score = 0.0
        subvertical_score = 0.0

        if v_counter and vertical:
            total_v = sum(v_counter.values())
            if total_v > 0:
                vertical_score = v_counter.get(vertical, 0) / total_v

        if sv_counter and subvertical:
            total_sv = sum(sv_counter.values())
            if total_sv > 0:
                subvertical_score = sv_counter.get(subvertical, 0) / total_sv

        return 0.4 * vertical_score + 0.6 * subvertical_score

    def blended_scores_for_impression(self, user_id: str, candidate_nids: Sequence[str]) -> List[float]:
        raw_als = [self.als_score_for(user_id, nid) for nid in candidate_nids]
        raw_content = [self.content_score_for(user_id, nid) for nid in candidate_nids]
        raw_pop = [self.popularity_score_for(nid) for nid in candidate_nids]

        als_scaled = minmax_scale(raw_als)
        content_scaled = minmax_scale(raw_content)
        pop_scaled = minmax_scale(raw_pop)

        w_als, w_content, w_pop = self.maturity_weights(user_id)

        blended = []
        for a, c, p in zip(als_scaled, content_scaled, pop_scaled):
            score = (w_als * a) + (w_content * c) + (w_pop * p)
            blended.append(float(score))
        return blended

    def evaluate(self) -> Dict[str, float]:
        assert self.valid_behaviors_df is not None
        eval_rows: List[ImpressionEval] = []

        for row in self.valid_behaviors_df.itertuples(index=False):
            parsed = parse_impression_tokens(row.impressions)
            if not parsed:
                continue
            candidate_nids = [nid for nid, _ in parsed]
            labels = [label for _, label in parsed]
            scores = self.blended_scores_for_impression(row.user_id, candidate_nids)
            eval_rows.append(ImpressionEval(labels=labels, scores=scores))

        return evaluate_impressions(eval_rows)

    def close(self) -> None:
        if self.spark is not None:
            self.spark.stop()
            self.spark = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blended MIND recommender with ALS + content + popularity.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("MIND_DATA_ROOT", "../smallDataset"),
        help="Root directory containing MINDsmall_train and MINDsmall_dev.",
    )
    parser.add_argument("--rank", type=int, default=32, help="ALS latent dimension.")
    parser.add_argument("--reg-param", type=float, default=0.05, help="ALS regularization.")
    parser.add_argument("--max-iter", type=int, default=10, help="ALS max iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommender = BlendedMindRecommender(
        data_root=Path(args.data_root).expanduser().resolve(),
        rank=args.rank,
        reg_param=args.reg_param,
        max_iter=args.max_iter,
    )

    try:
        recommender.fit()
        metrics = recommender.evaluate()
        print("\nBlended recommender results")
        print("=" * 40)
        for key, value in metrics.items():
            if key == "impressions_evaluated":
                print(f"{key}: {int(value)}")
            else:
                print(f"{key}: {value:.4f}")
    finally:
        recommender.close()


if __name__ == "__main__":
    main()
