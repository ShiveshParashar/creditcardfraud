import numpy as np
import pandas as pd

from src.data import check_duplicates, time_based_split


def _toy_df(n=100):
    return pd.DataFrame({
        "Time": np.arange(n, dtype=float),
        "Amount": np.random.rand(n) * 100,
        "Class": np.random.randint(0, 2, n),
    })


def test_time_based_split_sizes():
    df = _toy_df(100)
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_time_based_split_is_chronological():
    df = _toy_df(100)
    train, val, test = time_based_split(df)
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_check_duplicates_counts_exact_dupes():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    assert check_duplicates(df) == 1
