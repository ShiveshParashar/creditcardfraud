"""SHAP-based explainability for the XGBoost model.

Run as a script to produce a global summary plot (models/shap_summary.png).
Also exposes `top_reasons`, used by the FastAPI service to turn a single
transaction's SHAP values into a short human-readable explanation.
"""
import json

import matplotlib
import pandas as pd
import shap
from joblib import load

from src.config import MODELS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_explainer():
    model = load(MODELS_DIR / "xgb_model.joblib")
    return shap.TreeExplainer(model), model


def top_reasons(explainer: shap.TreeExplainer, row: pd.DataFrame, n=3) -> list[str]:
    """Turn one transaction's SHAP values into short natural-language reasons."""
    shap_values = explainer.shap_values(row)
    values = shap_values[0] if shap_values.ndim == 2 else shap_values
    contributions = sorted(
        zip(row.columns, values, row.iloc[0].values),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:n]

    reasons = []
    for feature, shap_val, feature_val in contributions:
        direction = "increases" if shap_val > 0 else "decreases"
        reasons.append(
            f"{feature}={feature_val:.2f} {direction} fraud risk "
            f"(impact {shap_val:+.3f})"
        )
    return reasons


def main():
    from src.data import load_raw, time_based_split
    from src.features import build_features, feature_columns

    explainer, _ = load_explainer()
    feat_cols = json.load(open(MODELS_DIR / "feature_columns.json"))

    raw = load_raw()
    df = build_features(raw)
    _, _, test = time_based_split(df)
    X_te = test[feat_cols]

    sample = X_te.sample(min(2000, len(X_te)), random_state=42)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    out_path = MODELS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved global SHAP summary plot to {out_path}")

    print("\nExample per-transaction explanation:")
    example = X_te.iloc[[0]]
    print(top_reasons(explainer, example))


if __name__ == "__main__":
    main()
