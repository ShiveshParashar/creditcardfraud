"""Evaluation metrics for a heavily imbalanced binary classifier.

Accuracy and ROC-AUC are both misleading at ~0.17% fraud prevalence (a
model predicting "never fraud" gets 99.8% accuracy). We report PR-AUC as
the headline metric, plus recall-at-fixed-FPR, precision@k, and a
cost-based decision threshold chosen on validation data only.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import REVIEW_COST


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(y_true, y_score, target_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def precision_at_k(y_true, y_score, k=100) -> float:
    y_true = np.asarray(y_true)
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].sum() / k)


def cost_of_threshold(y_true, y_score, amounts, threshold, review_cost=REVIEW_COST) -> float:
    """Total simulated cost at a given threshold.

    A missed fraud costs the transaction amount (the loss actually
    incurred). A false alarm costs a fixed manual-review fee.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    pred_fraud = y_score >= threshold

    missed = (~pred_fraud) & (y_true == 1)
    false_alarms = pred_fraud & (y_true == 0)

    return float(amounts[missed].sum() + review_cost * false_alarms.sum())


def find_best_threshold(y_true, y_score, amounts, review_cost=REVIEW_COST,
                         grid=None) -> tuple[float, float]:
    """Grid-search the threshold that minimizes total cost.

    IMPORTANT: call this on validation data only. The returned threshold
    is then applied once, unchanged, to the held-out test set.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    costs = [cost_of_threshold(y_true, y_score, amounts, t, review_cost) for t in grid]
    best_idx = int(np.argmin(costs))
    return float(grid[best_idx]), float(costs[best_idx])


def confusion_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y_true, y_score, amounts, threshold) -> dict:
    cm = confusion_at_threshold(y_true, y_score, threshold)
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "pr_auc": pr_auc(y_true, y_score),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_score, 0.01),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "total_cost": cost_of_threshold(y_true, y_score, amounts, threshold),
    }


def missed_frauds(df, y_true, y_score, threshold):
    """Rows that were fraud but scored below the threshold, for error analysis."""
    y_true = np.asarray(y_true)
    mask = (y_true == 1) & (y_score < threshold)
    out = df.loc[mask].copy()
    out["score"] = y_score[mask]
    return out.sort_values("score", ascending=False)
