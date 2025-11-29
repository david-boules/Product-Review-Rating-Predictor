"""
Retrain the TF-IDF + len_words + LinearSVC model using:
- the original cleaned_data.csv
- any extra labelled samples from data/feedback.csv

This script overwrites the model artifacts used by the API:
  model_artifacts/linear_svc_model.joblib
  model_artifacts/tfidf_vectorizer.joblib
  model_artifacts/scaler_len_words.joblib
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

CLEANED_PATH = DATA_DIR / "cleaned_data.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

MODEL_PATH = ARTIFACTS_DIR / "linear_svc_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler_len_words.joblib"


def load_original_data() -> pd.DataFrame:
    """Load the original cleaned dataset."""
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Could not find {CLEANED_PATH}")
    df = pd.read_csv(CLEANED_PATH)

    # Expecting at least: score, content_clean, len_words (from Phase 2)
    if "score" not in df.columns or "content_clean" not in df.columns:
        raise ValueError("cleaned_data.csv must contain 'score' and 'content_clean' columns")

    # Recompute len_words to be safe (in case it’s missing/outdated)
    df["content_clean"] = df["content_clean"].astype(str)
    df["len_words"] = df["content_clean"].str.split().str.len()

    return df


def load_feedback_data() -> pd.DataFrame:
    """
    Load feedback.csv if it exists and convert it to the same schema
    as cleaned_data.csv: columns 'score', 'content_clean', 'len_words'.
    """
    if not FEEDBACK_PATH.exists():
        print("No feedback.csv found – retraining on original data only.")
        return pd.DataFrame(columns=["score", "content_clean", "len_words"])

    df_fb = pd.read_csv(FEEDBACK_PATH)

    # Expect columns created by the /feedback endpoint:
    # timestamp, review_text, true_score, model_score
    if "review_text" not in df_fb.columns or "true_score" not in df_fb.columns:
        print("feedback.csv found but missing expected columns – ignoring feedback.")
        return pd.DataFrame(columns=["score", "content_clean", "len_words"])

    df_fb = df_fb.copy()
    df_fb["content_clean"] = df_fb["review_text"].astype(str)
    df_fb["score"] = df_fb["true_score"].astype(int)
    df_fb["len_words"] = df_fb["content_clean"].str.split().str.len()

    return df_fb[["score", "content_clean", "len_words"]]


def build_features_and_labels(df_all: pd.DataFrame):
    """
    Build:
      - TF-IDF matrix on 'content_clean'
      - scaled len_words feature
      - combined sparse feature matrix
      - target vector y (scores 1–5)
    """
    texts = df_all["content_clean"].astype(str).values
    y = df_all["score"].astype(int).values

    # Same TF-IDF configuration as Phase 2
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=5,
        max_df=0.9,
        max_features=50000,
    )

    X_tfidf = tfidf.fit_transform(texts)

    # len_words scaling
    scaler = StandardScaler()
    len_words = df_all["len_words"].values.reshape(-1, 1)
    len_words_scaled = scaler.fit_transform(len_words)
    X_len_sparse = sparse.csr_matrix(len_words_scaled)

    # Combine features
    X = sparse.hstack([X_tfidf, X_len_sparse], format="csr")

    return X, y, tfidf, scaler


def train_linear_svc(X, y):
    """
    Train LinearSVC with same hyperparameters as in Phase 4.
    Uses class_weight='balanced' like before.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    lin_svc = LinearSVC(
        C=0.1,
        class_weight=class_weights,
        loss="squared_hinge",
        max_iter=5000,
        random_state=42,
    )

    lin_svc.fit(X, y)
    return lin_svc


def main():
    print("🔄 Retraining model with original data + feedback (if any)...")

    # 1. Load data
    df_orig = load_original_data()
    df_fb = load_feedback_data()

    print(f"Original samples: {len(df_orig)}")
    print(f"Feedback samples: {len(df_fb)}")

    if len(df_fb) > 0:
        df_all = pd.concat([df_orig, df_fb], ignore_index=True)
    else:
        df_all = df_orig

    print(f"Total samples used for training: {len(df_all)}")

    # 2. Build features
    X, y, tfidf, scaler = build_features_and_labels(df_all)
    print("Feature matrix shape:", X.shape)

    # 3. Train classifier
    lin_svc = train_linear_svc(X, y)

    # 4. Save artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(lin_svc, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("✅ Retraining complete.")
    print(f"Saved model to:       {MODEL_PATH}")
    print(f"Saved vectorizer to:  {VECTORIZER_PATH}")
    print(f"Saved scaler to:      {SCALER_PATH}")


if __name__ == "__main__":
    main()
