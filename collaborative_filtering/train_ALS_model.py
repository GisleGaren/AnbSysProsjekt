import os
import math
import pandas as pd
from pathlib import Path
from csv import QUOTE_NONE

from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

LOCAL_DATA_ROOT = Path(os.environ.get("MIND_DATA_ROOT", "./smallDataset")).expanduser().resolve()

train_dir = LOCAL_DATA_ROOT / f"MINDsmall_train"
valid_dir = LOCAL_DATA_ROOT / f"MINDsmall_dev"

train_news_path      = str(train_dir / "news.tsv")
train_behaviors_path = str(train_dir / "behaviors.tsv")
valid_news_path      = str(valid_dir / "news.tsv")
valid_behaviors_path = str(valid_dir / "behaviors.tsv")

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
        sep='\t',
        quoting=QUOTE_NONE,
        dtype=object,
        na_filter=False,
    )

train_news_df = load_tsv(train_news_path, NEWS_COLUMNS)
valid_news_df = load_tsv(valid_news_path, NEWS_COLUMNS)
train_behaviors_df = load_tsv(train_behaviors_path, BEHAVIOR_COLUMNS)
valid_behaviors_df = load_tsv(valid_behaviors_path, BEHAVIOR_COLUMNS)

print(f"Loaded {len(train_news_df)} train news rows and {len(valid_news_df)} validation news rows from {LOCAL_DATA_ROOT}")
print(f"Loaded {len(train_behaviors_df)} train behaviors rows and {len(valid_behaviors_df)} validation behaviors rows")

def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Expands impression strings into (user, news, label) rows."""
    rows = []
    for row in df.itertuples(index=False):
        impressions = (row.impressions or '').strip()
        if not impressions:
            continue
        for token in impressions.split():
            if '-' not in token:
                continue
            news_id, label = token.rsplit('-', 1)
            if not news_id:
                continue
            try:
                label_value = float(label)
            except ValueError:
                continue
            rows.append((row.user_id, news_id, label_value))
    return pd.DataFrame(rows, columns=['user_id', 'nid', 'label'])


train_interactions_pd = build_interactions(train_behaviors_df)
valid_interactions_pd = build_interactions(valid_behaviors_df)
print(f"Prepared {len(train_interactions_pd):,} train interactions and {len(valid_interactions_pd):,} validation interactions")

spark = (
    SparkSession.builder
    .appName('mind-als-demo')
    .getOrCreate()
)

train_spark_df = spark.createDataFrame(train_interactions_pd)
valid_spark_df = spark.createDataFrame(valid_interactions_pd)

user_indexer = StringIndexer(
    inputCol='user_id',
    outputCol='user_idx',
    handleInvalid='skip'
).fit(train_spark_df)
item_indexer = StringIndexer(
    inputCol='nid',
    outputCol='item_idx',
    handleInvalid='skip'
).fit(train_spark_df)

train_indexed_df = item_indexer.transform(user_indexer.transform(train_spark_df))
valid_indexed_df = item_indexer.transform(user_indexer.transform(valid_spark_df))

train_ratings_df = train_indexed_df.select(
    F.col('user_idx').cast('int').alias('user_idx'),
    F.col('item_idx').cast('int').alias('item_idx'),
    F.col('label').cast('float').alias('label'),
).cache()

valid_ratings_df = valid_indexed_df.select(
    F.col('user_idx').cast('int').alias('user_idx'),
    F.col('item_idx').cast('int').alias('item_idx'),
    F.col('label').cast('float').alias('label'),
).cache()

print(f"Spark train interactions: {train_ratings_df.count():,}")
print(f"Spark validation interactions: {valid_ratings_df.count():,}")

als = ALS(
    rank=32,
    maxIter=10,
    regParam=0.05,
    userCol='user_idx',
    itemCol='item_idx',
    ratingCol='label',
    implicitPrefs=False,
    nonnegative=True,
    coldStartStrategy='drop',
)

als_model = als.fit(train_ratings_df)

# Save the trained model
model_save_path = Path("./models/als_model")
model_save_path.mkdir(parents=True, exist_ok=True)
als_model.write().overwrite().save(str(model_save_path))
print(f"Model saved to {model_save_path}")

