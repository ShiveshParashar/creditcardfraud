"""Streamlit dashboard: simulates a live transaction stream from the held-out
test split and shows real-time fraud alerts.

Run with:  streamlit run app/dashboard.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

# `streamlit run app/dashboard.py` puts this file's directory on sys.path,
# not the project root, so `src` wouldn't otherwise be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AMOUNT_COL, MODELS_DIR, TARGET_COL
from src.data import load_raw, time_based_split
from src.explain import top_reasons
from src.features import build_features

st.set_page_config(page_title="Fraud Monitoring", layout="wide")
st.title("Live Transaction Monitoring")


@st.cache_resource
def load_artifacts():
    import shap

    model = load(MODELS_DIR / "xgb_model.joblib")
    feat_cols = json.load(open(MODELS_DIR / "feature_columns.json"))
    metrics = json.load(open(MODELS_DIR / "metrics.json"))
    explainer = shap.TreeExplainer(model)
    return model, feat_cols, metrics["chosen_threshold"], explainer


@st.cache_data
@st.cache_data
def load_stream_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "test_stream.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test stream dataset not found at {data_path}"
        )

    raw = pd.read_csv(data_path)
    df = build_features(raw)

    return df.reset_index(drop=True)

with st.sidebar:
    st.header("Controls")
    speed = st.slider("Transactions per tick", 1, 50, 5)
    delay = st.slider("Delay between ticks (s)", 0.0, 2.0, 0.3)
    threshold = st.slider("Decision threshold", 0.0, 1.0, float(threshold), 0.01)
    run = st.toggle("Start stream", value=False)

col1, col2, col3 = st.columns(3)
seen_metric = col1.empty()
alerts_metric = col2.empty()
precision_metric = col3.empty()

alerts_placeholder = st.empty()
chart_placeholder = st.empty()

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
    st.session_state.alerts = []
    st.session_state.scores_over_time = []

if run:
    end = min(st.session_state.cursor + speed, len(test_df))
    batch = test_df.iloc[st.session_state.cursor:end]

    if len(batch):
        X_batch = batch[feat_cols]
        scores = model.predict_proba(X_batch)[:, 1]
        st.session_state.scores_over_time.extend(scores.tolist())

        flagged = scores >= threshold
        for i, (idx, row) in enumerate(batch.iterrows()):
            if flagged[i]:
                reasons = top_reasons(explainer, X_batch.iloc[[i]], n=2)
                st.session_state.alerts.insert(0, {
                    "time": row["Time"],
                    "amount": row[AMOUNT_COL],
                    "score": float(scores[i]),
                    "actual_fraud": bool(row[TARGET_COL]),
                    "reasons": "; ".join(reasons),
                })
        st.session_state.alerts = st.session_state.alerts[:25]
        st.session_state.cursor = end

seen_metric.metric("Transactions processed", st.session_state.cursor)
alerts_metric.metric("Alerts raised", len(st.session_state.alerts))
if st.session_state.alerts:
    true_positives = sum(a["actual_fraud"] for a in st.session_state.alerts)
    precision_metric.metric(
        "Alert precision (so far)",
        f"{true_positives / len(st.session_state.alerts):.0%}",
    )

if st.session_state.alerts:
    alerts_placeholder.dataframe(pd.DataFrame(st.session_state.alerts), use_container_width=True)
else:
    alerts_placeholder.info("No alerts yet.")

if st.session_state.scores_over_time:
    chart_placeholder.line_chart(st.session_state.scores_over_time[-200:])

if run and st.session_state.cursor < len(test_df):
    time.sleep(delay)
    st.rerun()
elif run:
    st.success("Reached the end of the simulated stream.")
