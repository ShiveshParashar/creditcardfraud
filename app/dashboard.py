"""Streamlit dashboard for live credit-card fraud monitoring."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from src.config import AMOUNT_COL, MODELS_DIR, TARGET_COL
from src.explain import top_reasons
from src.features import build_features


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Monitoring",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection")
st.subheader("Live Transaction Monitoring")


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():

    required_files = [
        "xgb_model.joblib",
        "feature_columns.json",
        "metrics.json",
    ]

    missing_files = [
        file
        for file in required_files
        if not (MODELS_DIR / file).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {missing_files}. "
            "Make sure the trained model artifacts are committed to GitHub."
        )

    model = load(
        MODELS_DIR / "xgb_model.joblib"
    )

    with open(
        MODELS_DIR / "feature_columns.json",
        "r"
    ) as f:
        feature_columns = json.load(f)

    with open(
        MODELS_DIR / "metrics.json",
        "r"
    ) as f:
        metrics = json.load(f)

    model_threshold = float(
        metrics["chosen_threshold"]
    )

    # SHAP explainer
    import shap

    explainer = shap.TreeExplainer(model)

    return (
        model,
        feature_columns,
        model_threshold,
        explainer,
    )


# ---------------------------------------------------------
# Load test stream
# ---------------------------------------------------------

@st.cache_data
def load_stream_data():

    data_path = (
        PROJECT_ROOT
        / "data"
        / "test_stream.csv"
    )

    if not data_path.exists():

        raise FileNotFoundError(
            f"Test stream dataset not found at {data_path}. "
            "Create data/test_stream.csv and push it to GitHub."
        )

    raw = pd.read_csv(data_path)

    # Apply the same feature engineering
    # used during model training.
    df = build_features(raw)

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# Load everything
# ---------------------------------------------------------

try:

    (
        model,
        feat_cols,
        model_threshold,
        explainer,
    ) = load_artifacts()

except FileNotFoundError as e:

    st.error(str(e))
    st.stop()


try:

    test_df = load_stream_data()

except FileNotFoundError as e:

    st.error(str(e))
    st.stop()


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Controls")

    speed = st.slider(
        "Transactions per tick",
        min_value=1,
        max_value=50,
        value=5,
    )

    delay = st.slider(
        "Delay between ticks (seconds)",
        min_value=0.0,
        max_value=2.0,
        value=0.3,
        step=0.1,
    )

    # IMPORTANT:
    # Use model_threshold as the default value.
    # Do NOT use threshold here because threshold
    # does not exist before this slider is created.
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(model_threshold),
        step=0.01,
    )

    st.divider()

    st.write(
        f"**Training threshold:** `{model_threshold:.4f}`"
    )

    st.write(
        f"**Current threshold:** `{threshold:.4f}`"
    )

    run = st.toggle(
        "▶️ Start stream",
        value=False,
    )


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "cursor" not in st.session_state:

    st.session_state.cursor = 0

if "alerts" not in st.session_state:

    st.session_state.alerts = []

if "scores_over_time" not in st.session_state:

    st.session_state.scores_over_time = []


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

seen_metric = col1.empty()
alerts_metric = col2.empty()
precision_metric = col3.empty()
fraud_metric = col4.empty()


# ---------------------------------------------------------
# Alert and chart placeholders
# ---------------------------------------------------------

alerts_placeholder = st.empty()

chart_placeholder = st.empty()


# ---------------------------------------------------------
# Process transaction stream
# ---------------------------------------------------------

if run:

    end = min(
        st.session_state.cursor + speed,
        len(test_df),
    )

    batch = test_df.iloc[
        st.session_state.cursor:end
    ]

    if len(batch) > 0:

        # Features required by model
        X_batch = batch[feat_cols]

        # Fraud probabilities
        scores = model.predict_proba(
            X_batch
        )[:, 1]

        # Store scores for chart
        st.session_state.scores_over_time.extend(
            scores.tolist()
        )

        # Determine which transactions are suspicious
        flagged = scores >= threshold

        # Process alerts
        for i, (idx, row) in enumerate(
            batch.iterrows()
        ):

            if flagged[i]:

                reasons = top_reasons(
                    explainer,
                    X_batch.iloc[[i]],
                    n=2,
                )

                st.session_state.alerts.insert(
                    0,
                    {
                        "Time": row["Time"],
                        "Amount": row[AMOUNT_COL],
                        "Fraud Score": float(
                            scores[i]
                        ),
                        "Decision": "FLAGGED",
                        "Actual Fraud": bool(
                            row[TARGET_COL]
                        ),
                        "Reasons": "; ".join(
                            reasons
                        ),
                    },
                )

        # Keep only latest 25 alerts
        st.session_state.alerts = (
            st.session_state.alerts[:25]
        )

        # Move cursor
        st.session_state.cursor = end


# ---------------------------------------------------------
# Display metrics
# ---------------------------------------------------------

seen_metric.metric(
    "Transactions Processed",
    st.session_state.cursor,
)

alerts_metric.metric(
    "Alerts Raised",
    len(st.session_state.alerts),
)


if st.session_state.alerts:

    true_positives = sum(
        alert["Actual Fraud"]
        for alert in st.session_state.alerts
    )

    precision = (
        true_positives
        / len(st.session_state.alerts)
    )

    precision_metric.metric(
        "Alert Precision",
        f"{precision:.0%}",
    )

else:

    precision_metric.metric(
        "Alert Precision",
        "N/A",
    )


# ---------------------------------------------------------
# Actual fraud count in processed transactions
# ---------------------------------------------------------

if st.session_state.cursor > 0:

    processed = test_df.iloc[
        :st.session_state.cursor
    ]

    actual_fraud_count = int(
        processed[TARGET_COL].sum()
    )

    fraud_metric.metric(
        "Actual Fraud Detected",
        actual_fraud_count,
    )

else:

    fraud_metric.metric(
        "Actual Fraud Detected",
        0,
    )


# ---------------------------------------------------------
# Alert table
# ---------------------------------------------------------

st.subheader("🚨 Fraud Alerts")

if st.session_state.alerts:

    alert_df = pd.DataFrame(
        st.session_state.alerts
    )

    alerts_placeholder.dataframe(
        alert_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    alerts_placeholder.info(
        "No suspicious transactions detected yet."
    )


# ---------------------------------------------------------
# Fraud score chart
# ---------------------------------------------------------

st.subheader("📈 Fraud Probability Over Time")

if st.session_state.scores_over_time:

    chart_placeholder.line_chart(
        st.session_state.scores_over_time[-200:]
    )

else:

    chart_placeholder.info(
        "Start the transaction stream to see fraud scores."
    )


# ---------------------------------------------------------
# Stream control
# ---------------------------------------------------------

if (
    run
    and st.session_state.cursor < len(test_df)
):

    time.sleep(delay)

    st.rerun()


elif (
    run
    and st.session_state.cursor >= len(test_df)
):

    st.success(
        "✅ Reached the end of the simulated transaction stream."
    )