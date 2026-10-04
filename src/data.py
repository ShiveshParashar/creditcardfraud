"""Loading and time-based splitting for the ULB credit card fraud dataset.

The dataset is time-ordered (`Time` = seconds elapsed since the first
transaction). We split chronologically instead of randomly shuffling,
since a random split would leak future transaction patterns into
training and overstate performance.
"""
import pandas as pd

from src.config import RAW_CSV, TARGET_COL, TIME_COL, TRAIN_FRAC, VAL_FRAC


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' from "
            "https://www.kaggle.com/mlg-ulb/creditcardfraud and place it there."
        )
    df = pd.read_csv(path)
    return df.sort_values(TIME_COL).reset_index(drop=True)


def time_based_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Chronological train/val/test split (no shuffling)."""
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)
    return train, val, test


def check_duplicates(df: pd.DataFrame) -> int:
    """Return the count of fully duplicated rows (a common leakage source)."""
    return int(df.duplicated().sum())


def Xy(df: pd.DataFrame, feature_cols):
    return df[feature_cols], df[TARGET_COL]
