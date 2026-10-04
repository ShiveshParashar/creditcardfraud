"""Shared paths and constants for the fraud detection pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_CSV = DATA_RAW / "creditcard.csv"

TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"

# Fraction of rows (sorted by Time) assigned to each split.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test

RANDOM_STATE = 42

# Cost model used for threshold selection (see src/evaluate.py).
# A false alarm costs a fixed review fee; a missed fraud costs the
# transaction amount itself.
REVIEW_COST = 5.0

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
