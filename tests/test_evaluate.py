import numpy as np

from src.evaluate import (
    cost_of_threshold,
    find_best_threshold,
    pr_auc,
    precision_at_k,
    summarize,
)


def test_pr_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y_true, y_score) == 1.0


def test_precision_at_k():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.3, 0.2])
    assert precision_at_k(y_true, y_score, k=2) == 1.0
    assert precision_at_k(y_true, y_score, k=5) == 0.4


def test_cost_of_threshold_missed_fraud_costs_its_amount():
    y_true = np.array([1])
    y_score = np.array([0.1])
    amounts = np.array([250.0])
    # threshold above the score -> fraud missed -> cost == amount
    assert cost_of_threshold(y_true, y_score, amounts, threshold=0.5) == 250.0


def test_cost_of_threshold_false_alarm_costs_review_fee():
    y_true = np.array([0])
    y_score = np.array([0.9])
    amounts = np.array([250.0])
    assert cost_of_threshold(y_true, y_score, amounts, threshold=0.5, review_cost=5.0) == 5.0


def test_find_best_threshold_prefers_catching_large_fraud():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.6, 0.55, 0.1, 0.1])
    amounts = np.array([1000.0, 10.0, 10.0, 10.0])
    best_th, best_cost = find_best_threshold(
        y_true, y_score, amounts, review_cost=5.0, grid=np.array([0.5, 0.7])
    )
    # at th=0.5: the real fraud (0.6) is caught, one false alarm (0.55) -> cost = 1*5 = 5
    # at th=0.7: nothing is flagged, the $1000 fraud is missed -> cost = 1000
    assert best_th == 0.5
    assert best_cost == 5.0


def test_summarize_returns_expected_keys():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.3, 0.6])
    amounts = np.array([10, 20, 10, 30], dtype=float)
    out = summarize(y_true, y_score, amounts, threshold=0.5)
    assert set(out.keys()) >= {
        "pr_auc", "recall_at_1pct_fpr", "precision_at_100",
        "threshold", "precision", "recall", "f1",
        "confusion_matrix", "total_cost",
    }
