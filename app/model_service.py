"""
This module wraps the trained ML model and its preprocessing steps
into a single function: predict_review_score(text) -> int rating (1–5).

It is imported by the FastAPI app so that the API can call the model.
"""

import joblib                   #for loading saved sklearn objects (.joblib files)
import numpy as np
from scipy import sparse
from pathlib import Path

# Paths relative to project root
ARTIFACTS_DIR = Path("model_artifacts")
MODEL_PATH = ARTIFACTS_DIR / "linear_svc_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler_len_words.joblib"

# Load once at import time
model = joblib.load(MODEL_PATH)             # trained LinearSVC
tfidf = joblib.load(VECTORIZER_PATH)        # fitted TFIDF-Vectorizer
scaler = joblib.load(SCALER_PATH)           # fitted StandardScaler for len_words

def predict_review_score(text: str) -> int:
    """
    End-to-end prediction using:
    - TF-IDF on raw text
    - len_words feature (scaled)
    - LinearSVC classifier
    """
    # TF-IDF
    X_tfidf_input = tfidf.transform([text])

    # len_words feature
    len_words = np.array([len(text.split())]).reshape(-1, 1)
    len_words_scaled = scaler.transform(len_words)

    # Combine
    X_len_sparse = sparse.csr_matrix(len_words_scaled)
    X_input = sparse.hstack([X_tfidf_input, X_len_sparse], format="csr")

    # Predict
    prediction = model.predict(X_input)[0]
    return int(prediction)
