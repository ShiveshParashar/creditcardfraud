"""FastAPI fraud-scoring service.

Run with:  uvicorn api.main:app --reload --port 8000

POST /score with a transaction's raw fields (Time, Amount, V1..V28) and get
back a fraud probability, a decision at the cost-optimal threshold chosen
during training, and the top SHAP reasons for that score.
"""
import json
from functools import lru_cache

import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODELS_DIR
from src.explain import top_reasons
from src.features import build_features

app = FastAPI(title="Fraud Detection API", version="1.0")


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float  # noqa: E702
    V6: float; V7: float; V8: float; V9: float; V10: float  # noqa: E702
    V11: float; V12: float; V13: float; V14: float; V15: float  # noqa: E702
    V16: float; V17: float; V18: float; V19: float; V20: float  # noqa: E702
    V21: float; V22: float; V23: float; V24: float; V25: float  # noqa: E702
    V26: float; V27: float; V28: float  # noqa: E702

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 50000, "Amount": 149.62,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            }
        }
    )


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    top_reasons: list[str]


@lru_cache(maxsize=1)
def get_artifacts():
    required = ["xgb_model.joblib", "feature_columns.json", "metrics.json"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing trained artifacts {missing} in {MODELS_DIR}. "
            "Run `python -m src.train` first."
        )
    model = load(MODELS_DIR / "xgb_model.joblib")
    feat_cols = json.load(open(MODELS_DIR / "feature_columns.json"))
    metrics = json.load(open(MODELS_DIR / "metrics.json"))
    explainer = shap.TreeExplainer(model)
    return model, feat_cols, metrics["chosen_threshold"], explainer


@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "status": "healthy"
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    try:
        model, feat_cols, threshold, explainer = get_artifacts()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw_row = pd.DataFrame([txn.model_dump()])
    features = build_features(raw_row)

    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Feature(s) {missing} required by the model are missing after feature engineering.",
        )
    row = features[feat_cols]

    probability = float(model.predict_proba(row)[:, 1][0])
    decision = "flag_for_review" if probability >= threshold else "approve"
    reasons = top_reasons(explainer, row, n=3)

    return ScoreResponse(
        fraud_probability=probability,
        decision=decision,
        threshold=threshold,
        top_reasons=reasons,
    )
