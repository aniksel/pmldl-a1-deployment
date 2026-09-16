# стадия 1: чистка и разбиение данных
# data/raw/housing.csv -> data/processed/train.csv, data/processed/test.csv

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    OUTLIER_COLUMNS,
    RANDOM_STATE,
    RAW_FILE,
    TARGET_CAP,
    TEST_FILE,
    TEST_SIZE,
    TRAIN_FILE,
)
from features import CATEGORICAL, NUMERIC, TARGET


def clean(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET]).copy()

    for col in NUMERIC:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        df[col] = df[col].fillna("UNKNOWN")

    df = df[df[TARGET] < TARGET_CAP]

    for col in OUTLIER_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    return df.reset_index(drop=True)


df = pd.read_csv(RAW_FILE)
print("Raw rows:", len(df), "| missing values:", int(df.isna().sum().sum()))

df = clean(df)
print("Rows after cleaning:", len(df))

train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)
print("Train:", len(train), "-> ", TRAIN_FILE)
print("Test:", len(test), "-> ", TEST_FILE)
