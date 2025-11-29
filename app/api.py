"""
FastAPI application that exposes the ML model as a REST API.

Endpoints:
- GET  /health    : simple status check.
- POST /predict   : take review_text -> return predicted_score.
- POST /feedback  : log user feedback for future retraining.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import csv
from datetime import datetime

# Import model prediction function
from app.model_service import predict_review_score

# The 'app' variable is what Uvicorn looks for in `uvicorn app.api:app`.
app = FastAPI(title="ChatGPT Review Rating API")

# Feedback storage paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

#Pydantic models (request/response schemas)
class PredictRequest(BaseModel):
    review_text: str = Field(..., min_length=3)

class PredictResponse(BaseModel):
    predicted_score: int

class FeedbackRequest(BaseModel):
    review_text: str
    true_score: int = Field(..., ge=1, le=5)
    model_score: int = Field(..., ge=1, le=5)

# API Endpoints
@app.get("/health")
def health():
    """
    Simple health check endpoint.
    Returns HTTP 200 with JSON: {"status": "ok"} if the server is running (for debugging purposes)
    """
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict endpoint.
    Receives JSON like:
      {"review_text": "I love this app!"}

    Uses predict_review_score() from model_service and returns:
      {"predicted_score": 5}
    """
    score = predict_review_score(req.review_text)
    return PredictResponse(predicted_score=score)

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Feedback endpoint.
    Receives JSON like:
      {
        "review_text": "This app is not bad",
        "true_score": 4,
        "model_score": 3
      }

    Appends a row to data/feedback.csv for future retraining.
    If the file does not exist, we create it and write a header row first.
    """
    # Create file with header if not existing
    is_new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "review_text", "true_score", "model_score"])
        writer.writerow([
            datetime.utcnow().isoformat(), # write with current UTC time
            req.review_text,
            req.true_score,
            req.model_score,
        ])
    return {"status": "recorded"}
