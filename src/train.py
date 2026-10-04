"""Train and compare fraud models.

Run with:  python -m src.train

Produces, under models/:
  - xgb_model.joblib          main supervised model
  - logreg_baseline.joblib    logistic regression baseline
  - isolation_forest.joblib   unsupervised baseline
  - autoencoder.joblib (or .keras)  trained on legitimate transactions only
  - scaler.joblib             StandardScaler fit on train, used by the
                               logistic regression + autoencoder + API
  - feature_columns.json      the exact feature list/order models expect
  - metrics.json              headline metrics + chosen threshold
"""
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import AMOUNT_COL, MODELS_DIR, RANDOM_STATE, TARGET_COL
from src.data import load_raw, time_based_split
from src.evaluate import find_best_threshold, pr_auc, summarize
from src.features import build_features, feature_columns


def rule_based_scores(df: pd.DataFrame, threshold_amount=200.0) -> np.ndarray:
    """Trivial baseline: flag anything above a fixed amount."""
    return (df[AMOUNT_COL] >= threshold_amount).astype(float).values


def compare_imbalance_strategies(X, y, random_state=RANDOM_STATE) -> dict:
    """Compare class-weighting, SMOTE, and undersampling via stratified CV.

    Resampling is applied inside each fold (via imblearn's Pipeline) so the
    synthetic/undersampled data never leaks into the held-out fold -- doing
    this outside the CV loop is a classic source of optimistic bias.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}

    class_weight_pipe = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    results["class_weight"] = cross_val_score(
        class_weight_pipe, X, y, cv=cv, scoring="average_precision"
    ).mean()

    smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    results["smote"] = cross_val_score(
        smote_pipe, X, y, cv=cv, scoring="average_precision"
    ).mean()

    undersample_pipe = ImbPipeline([
        ("under", RandomUnderSampler(random_state=random_state)),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    results["undersample"] = cross_val_score(
        undersample_pipe, X, y, cv=cv, scoring="average_precision"
    ).mean()

    return {k: float(v) for k, v in results.items()}


def train_logreg_baseline(X_tr, y_tr, random_state=RANDOM_STATE) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(X_tr, y_tr)
    return model


def train_xgb(X_tr, y_tr, X_val, y_val, random_state=RANDOM_STATE) -> XGBClassifier:
    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_isolation_forest(X_tr, y_tr, random_state=RANDOM_STATE) -> IsolationForest:
    """Unsupervised baseline, fit on legitimate transactions only."""
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1
    )
    model.fit(X_tr[y_tr == 0])
    return model


def isoforest_fraud_scores(model: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous = more likely fraud (IsolationForest's native
    score_samples is the opposite direction, so we flip the sign)."""
    return -model.score_samples(X)


def train_autoencoder(X_tr_scaled, y_tr, random_state=RANDOM_STATE):
    """Autoencoder trained on legitimate transactions only; reconstruction
    error on held-out data is used as an anomaly score. Tries TensorFlow/Keras
    first and falls back to an MLP-based bottleneck autoencoder (scikit-learn)
    if TensorFlow isn't available for this Python version.
    """
    legit = X_tr_scaled[y_tr == 0]
    n_features = legit.shape[1]
    encoding_dim = max(2, n_features // 4)

    try:
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(random_state)
        inputs = keras.Input(shape=(n_features,))
        encoded = keras.layers.Dense(max(encoding_dim * 2, 4), activation="relu")(inputs)
        encoded = keras.layers.Dense(encoding_dim, activation="relu")(encoded)
        decoded = keras.layers.Dense(max(encoding_dim * 2, 4), activation="relu")(encoded)
        decoded = keras.layers.Dense(n_features, activation="linear")(decoded)

        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(
            legit, legit,
            epochs=30, batch_size=256, shuffle=True,
            validation_split=0.1, verbose=0,
        )
        return ("keras", autoencoder)
    except ImportError:
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=(max(encoding_dim * 2, 4), encoding_dim, max(encoding_dim * 2, 4)),
            activation="relu", max_iter=300, random_state=random_state,
        )
        model.fit(legit, legit)
        return ("sklearn", model)


def autoencoder_scores(kind_model, X_scaled) -> np.ndarray:
    kind, model = kind_model
    recon = model.predict(X_scaled)
    return np.mean((X_scaled - recon) ** 2, axis=1)


def main():
    print("Loading data...")
    raw = load_raw()
    df = build_features(raw)
    train, val, test = time_based_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"(fraud rate train={train[TARGET_COL].mean():.4%})")

    feat_cols = feature_columns(df, target_col=TARGET_COL)
    X_tr, y_tr = train[feat_cols], train[TARGET_COL]
    X_val, y_val = val[feat_cols], val[TARGET_COL]
    X_te, y_te = test[feat_cols], test[TARGET_COL]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    print("\n--- Baselines ---")
    rule_val_scores = rule_based_scores(val)
    print(f"Rule-based (amount threshold) val PR-AUC: {pr_auc(y_val, rule_val_scores):.4f}")

    logreg = train_logreg_baseline(X_tr_s, y_tr)
    logreg_val_scores = logreg.predict_proba(X_val_s)[:, 1]
    print(f"Logistic regression val PR-AUC: {pr_auc(y_val, logreg_val_scores):.4f}")

    print("\n--- Imbalance-handling comparison (5-fold CV on train, PR-AUC) ---")
    imbalance_results = compare_imbalance_strategies(X_tr_s, y_tr.values)
    for k, v in imbalance_results.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Main model: XGBoost ---")
    xgb_model = train_xgb(X_tr, y_tr, X_val, y_val)
    xgb_val_scores = xgb_model.predict_proba(X_val)[:, 1]
    print(f"XGBoost val PR-AUC: {pr_auc(y_val, xgb_val_scores):.4f}")

    print("\n--- Isolation Forest (unsupervised) ---")
    iso = train_isolation_forest(X_tr_s, y_tr.values)
    iso_val_scores = isoforest_fraud_scores(iso, X_val_s)
    print(f"Isolation Forest val PR-AUC: {pr_auc(y_val, iso_val_scores):.4f}")

    print("\n--- Autoencoder (trained on legitimate transactions only) ---")
    ae = train_autoencoder(X_tr_s, y_tr.values)
    ae_val_scores = autoencoder_scores(ae, X_val_s)
    print(f"Autoencoder ({ae[0]}) val PR-AUC: {pr_auc(y_val, ae_val_scores):.4f}")

    print("\n--- Cost-based threshold selection (on validation only) ---")
    best_threshold, best_cost = find_best_threshold(
        y_val.values, xgb_val_scores, val[AMOUNT_COL].values
    )
    print(f"Chosen threshold: {best_threshold:.2f} (val cost: {best_cost:.2f})")

    print("\n--- Final test evaluation (threshold applied once, unchanged) ---")
    xgb_te_scores = xgb_model.predict_proba(X_te)[:, 1]
    test_summary = summarize(y_te.values, xgb_te_scores, test[AMOUNT_COL].values, best_threshold)
    print(json.dumps(test_summary, indent=2))

    print("\nSaving artifacts to", MODELS_DIR)
    dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    dump(logreg, MODELS_DIR / "logreg_baseline.joblib")
    dump(iso, MODELS_DIR / "isolation_forest.joblib")
    dump(scaler, MODELS_DIR / "scaler.joblib")
    if ae[0] == "keras":
        ae[1].save(MODELS_DIR / "autoencoder.keras")
    else:
        dump(ae[1], MODELS_DIR / "autoencoder.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    metrics = {
        "baselines": {
            "rule_based_pr_auc": pr_auc(y_val, rule_val_scores),
            "logreg_pr_auc": pr_auc(y_val, logreg_val_scores),
        },
        "imbalance_comparison_cv_pr_auc": imbalance_results,
        "isolation_forest_val_pr_auc": pr_auc(y_val, iso_val_scores),
        "autoencoder_val_pr_auc": pr_auc(y_val, ae_val_scores),
        "autoencoder_backend": ae[0],
        "xgb_val_pr_auc": pr_auc(y_val, xgb_val_scores),
        "chosen_threshold": best_threshold,
        "test_summary": test_summary,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
