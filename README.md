# Credit Card Fraud Detection
Deployment Link-https://creditcardfraud-1-x2nh.onrender.com
End-to-end fraud detection on the [ULB Kaggle credit card fraud
dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (284,807
transactions, 492 frauds, 0.172% positive rate): time-based evaluation,
multiple imbalance-handling strategies, a cost-sensitive decision
threshold, SHAP explanations, and a deployed scoring API + live dashboard.

## Why this isn't just "XGBoost got 99% accuracy"

At 0.17% fraud prevalence, accuracy and ROC-AUC are both misleading — a
model that never predicts fraud still scores ~99.8% accuracy. This project
instead:

- Uses a **time-based split** (train on earlier transactions, test on
  later ones), not a random shuffle, since a random split leaks future
  patterns into training.
- Reports **PR-AUC** as the headline metric, plus recall at a fixed
  false-positive rate and precision@k.
- Picks a **cost-based decision threshold** on validation data only
  (missed fraud costs the transaction amount; a false alarm costs a fixed
  review fee), then applies it once, unchanged, to the test set.
- Compares **class weighting vs. SMOTE vs. undersampling**, resampling
  inside each CV fold to avoid leakage.
- Produces **SHAP-based per-transaction explanations** ("flagged because
  V14 is far outside its normal range"), not just a bare probability.
- Ships as a working **FastAPI scoring service + Streamlit live-monitoring
  dashboard**, not just a notebook.

## Project structure

```
creditcardfraud/
├── data/
│   ├── raw/            # place creditcard.csv here (gitignored)
│   └── processed/
├── src/
│   ├── config.py        # paths, constants, cost model
│   ├── data.py           # loading, time-based split
│   ├── features.py       # time/amount features (+ velocity features if card ids exist)
│   ├── train.py           # baselines, XGBoost, Isolation Forest, autoencoder
│   ├── evaluate.py        # PR-AUC, cost-based threshold, confusion matrix
│   └── explain.py         # SHAP summary + per-transaction reasons
├── api/
│   └── main.py            # FastAPI /score endpoint
├── app/
│   └── dashboard.py       # Streamlit live transaction stream + alerts
├── tests/
├── models/                 # trained artifacts (gitignored)
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**macOS only:** XGBoost needs the OpenMP runtime:

```bash
brew install libomp
```

Download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it at
`data/raw/creditcard.csv`.

## Running the pipeline

```bash
# 1. Train all models (baselines, XGBoost, Isolation Forest, autoencoder)
#    and pick the cost-optimal threshold on validation data.
python -m src.train

# 2. Generate a global SHAP summary plot + example per-transaction reasons.
python -m src.explain

# 3. Run tests.
pytest
```

`python -m src.train` writes to `models/`:
- `xgb_model.joblib`, `logreg_baseline.joblib`, `isolation_forest.joblib`,
  `autoencoder.joblib`/`.keras`, `scaler.joblib`
- `feature_columns.json` — exact feature list/order the models expect
- `metrics.json` — PR-AUC for every model, the imbalance-strategy
  comparison, the chosen threshold, and the final test-set summary

## Serving

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"Time": 50000, "Amount": 149.62, "V1": 0, "V2": 0, "V3": 0, "V4": 0,
       "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0,
       "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0,
       "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0,
       "V26": 0, "V27": 0, "V28": 0}'
```

returns:

```json
{
  "fraud_probability": 0.041,
  "decision": "approve",
  "threshold": 0.33,
  "top_reasons": ["V14=... decreases fraud risk (impact -0.50)", "..."]
}
```

### Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

(Train first and make sure `models/` is populated — it's copied into the
image.)

### Dashboard

```bash
streamlit run app/dashboard.py
```

Simulates a live transaction stream from the held-out test split, scores
each transaction with the trained model, and shows running alerts, an
alert-precision metric, and a score timeline. The decision threshold is
adjustable live from the sidebar.

## Evaluation methodology

1. **Split**: chronological 70/15/15 train/val/test. No shuffling.
2. **Baselines**: a fixed-amount rule, and logistic regression with
   balanced class weights.
3. **Imbalance comparison**: 5-fold CV on the training set only, comparing
   class weighting, SMOTE, and random undersampling — each applied inside
   the fold via an `imblearn` pipeline so no resampled data leaks across
   the fold boundary.
4. **Main model**: XGBoost with `scale_pos_weight` set from the training
   class ratio, early-stopped on PR-AUC against the validation set.
5. **Unsupervised baselines**: Isolation Forest and an autoencoder, both
   fit on legitimate transactions only; anomaly score = isolation score /
   reconstruction error.
6. **Threshold selection**: grid search over a cost function on the
   *validation* set (`missed fraud → lose the transaction amount`,
   `false alarm → fixed review cost`, see `src/config.REVIEW_COST`). The
   resulting threshold is applied once, unchanged, to the test set.
7. **Explainability**: SHAP `TreeExplainer` on the XGBoost model, both a
   global summary plot and per-transaction top-3 contributing features.

## Known pitfalls this avoids

- Reporting accuracy or ROC-AUC alone on an imbalanced dataset.
- Applying SMOTE before splitting (leaks synthetic neighbors across the
  train/test boundary).
- Random train/test splits on time-ordered transaction data.
- Tuning the decision threshold on the test set.
- Shipping a model with no baseline comparison.

## Extending to a richer dataset

`src/features.add_velocity_features` is a no-op on the plain ULB columns
but activates automatically on any dataset with a `card_id` column (e.g.
IEEE-CIS or a PaySim-style simulation), adding transaction velocity
(1h/24h counts), deviation from the card's running average amount, time
since the last transaction, and a new-card flag.
