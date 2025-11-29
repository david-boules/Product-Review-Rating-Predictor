"""
Train the initial TF-IDF + len_words + LinearSVC model
using only the original cleaned_data.csv (no feedback).

This script is essentially the Phase 4 training code, refactored
into a standalone script that produces the three core artifacts:

  model_artifacts/linear_svc_model.joblib
  model_artifacts/tfidf_vectorizer.joblib
  model_artifacts/scaler_len_words.joblib
"""

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

CLEANED_PATH = DATA_DIR / "cleaned_data.csv"

MODEL_PATH = ARTIFACTS_DIR / "linear_svc_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler_len_words.joblib"


def load_data() -> pd.DataFrame:
    """
    Load the original cleaned dataset.

    Expected columns:
      - 'score'         : integer rating 1–5
      - 'content_clean' : cleaned review text

    Recompute 'len_words' to be safe.
    """
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Could not find {CLEANED_PATH}")

    df = pd.read_csv(CLEANED_PATH)

    if "score" not in df.columns or "content_clean" not in df.columns:
        raise ValueError("cleaned_data.csv must contain 'score' and 'content_clean' columns")

    df["content_clean"] = df["content_clean"].astype(str)
    df["len_words"] = df["content_clean"].str.split().str.len()

    return df


def build_features_and_labels(df: pd.DataFrame):
    """
    Build feature matrix X and label vector y from the dataframe.

    - TF-IDF on 'content_clean'
    - Standardized len_words
    - Combined via sparse hstack
    """
    texts = df["content_clean"].astype(str).values
    y = df["score"].astype(int).values

    # TF-IDF configuration as used in earlier phases
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        max_features=50000,
    )
    X_tfidf = tfidf.fit_transform(texts)

    # len_words feature scaling
    len_words = df["len_words"].values.reshape(-1, 1)
    scaler = StandardScaler()
    len_words_scaled = scaler.fit_transform(len_words)
    X_len_sparse = sparse.csr_matrix(len_words_scaled)

    # Combine TF-IDF and len_words into one feature matrix
    X = sparse.hstack([X_tfidf, X_len_sparse], format="csr")

    return X, y, tfidf, scaler


def train_linear_svc(X, y):
    """
    Train the LinearSVC classifier with the same settings as Phase 4.
    Uses explicit class weights to handle rating imbalance.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    clf = LinearSVC(
        C=0.1,
        class_weight=class_weights,
        loss="squared_hinge",
        max_iter=5000,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def main():
    print("🚀 Training initial model from cleaned_data.csv (no feedback)...")

    # 1. Load and prepare data
    df = load_data()
    print(f"Samples: {len(df)}")

    # 2. Build features and labels
    X, y, tfidf, scaler = build_features_and_labels(df)
    print("Feature matrix shape:", X.shape)

    # 3. Train classifier
    clf = train_linear_svc(X, y)

    # 4. Save artifacts
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("✅ Initial training complete.")
    print(f"Saved model to:       {MODEL_PATH}")
    print(f"Saved vectorizer to:  {VECTORIZER_PATH}")
    print(f"Saved scaler to:      {SCALER_PATH}")


if __name__ == "__main__":
    main()
