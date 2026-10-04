"""Feature engineering.

The ULB dataset ships only `Time` (seconds since first transaction) and
`Amount` as raw, interpretable fields -- V1-V28 are already PCA components
and should not be engineered further. We derive a few cheap, legitimate
signals from Time/Amount.

If a richer dataset is used (e.g. IEEE-CIS, or anything with a card/customer
id and a real timestamp), `add_velocity_features` adds transaction-velocity
and deviation-from-average features. It's a no-op when those columns don't
exist, so it's always safe to call.
"""
import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60


def add_time_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % SECONDS_PER_DAY) // 3600
    df["day"] = (df["Time"] // SECONDS_PER_DAY).astype(int)
    df["amount_log"] = np.log1p(df["Amount"])
    return df


def add_velocity_features(df: pd.DataFrame, card_col="card_id", time_col="Time") -> pd.DataFrame:
    """Transaction velocity + deviation from the card's running average.

    No-op if `card_col` isn't present (true for the plain ULB dataset).
    """
    if card_col not in df.columns:
        return df

    df = df.sort_values(time_col).copy()
    grp = df.groupby(card_col)

    # Count of prior transactions by the same card in the last hour / day.
    df["txn_count_1h"] = grp[time_col].transform(
        lambda s: s.apply(lambda t: ((s < t) & (s >= t - 3600)).sum())
    )
    df["txn_count_24h"] = grp[time_col].transform(
        lambda s: s.apply(lambda t: ((s < t) & (s >= t - SECONDS_PER_DAY)).sum())
    )

    # Running mean amount for the card, excluding the current transaction,
    # and the current transaction's deviation from it.
    running_mean = grp["Amount"].transform(lambda s: s.shift().expanding().mean())
    df["amount_dev_from_avg"] = df["Amount"] - running_mean
    df["amount_ratio_to_avg"] = df["Amount"] / running_mean.replace(0, np.nan)

    df["time_since_last_txn"] = grp[time_col].diff()
    df["is_new_card"] = grp.cumcount().eq(0).astype(int)

    return df.fillna({"amount_dev_from_avg": 0, "amount_ratio_to_avg": 1,
                       "time_since_last_txn": -1})


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_amount_features(df)
    df = add_velocity_features(df)
    return df


def feature_columns(df: pd.DataFrame, target_col="Class", drop=("Time",)):
    return [c for c in df.columns if c != target_col and c not in drop]
