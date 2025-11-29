# Product Review Rating Predictor

This project predicts a **1–5 star rating** from a raw product review text (using the ChatGPT Google Play Store Reviews dataset, link to dataset: https://www.kaggle.com/datasets/bhavikjikadara/chatgpt-user-feedback).


It uses:
- **TF–IDF text features**
- **`len_words` engineered feature**
- **Linear SVM classifier**
- **FastAPI backend** (model service)
- **Streamlit frontend** (utility application)
- **Online learning** via a feedback + retraining loop

This work completes **Phase V** of the “CSCE 3602 - Fundamentals of Machine Learning” course project.

---

## Project Structure

```
.
├── app/
│   ├── api.py                     # FastAPI app: /predict and /feedback endpoints
│   └── model_service.py           # Loads artifacts and runs predictions
│
├── client/
│   └── streamlit_app.py           # User-facing UI (Streamlit frontend)
│
├── training/
│   ├── train_initial.py           # Initial training from cleaned_data.csv
│   └── retrain_with_feedback.py   # Retraining using original + feedback data
│
├── data/
│   ├── cleaned_data.csv           # Original cleaned dataset (from Phases 2–3)
│   └── feedback.csv               # Logged user feedback (created at runtime)
│
├── model_artifacts/
│   ├── linear_svc_model.joblib     # Trained Linear SVC model
│   ├── tfidf_vectorizer.joblib     # Fitted TF–IDF vectorizer
│   └── scaler_len_words.joblib     # Fitted StandardScaler for len_words
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

It is recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal dependency list:

- fastapi
- uvicorn
- streamlit
- requests
- numpy
- pandas
- scikit-learn
- scipy
- joblib

---

## Running the Backend (Model Service)

From the project root, run FastAPI:

```bash
uvicorn app.api:app --reload
```

This starts the model service at:

- **API root:** http://127.0.0.1:8000  
- **Health check:** http://127.0.0.1:8000/health  
- **Interactive API docs (Swagger UI):** http://127.0.0.1:8000/docs  

### Exposed Endpoints

| Method | Endpoint    | Description                                                |
|--------|-------------|------------------------------------------------------------|
| GET    | `/health`   | Simple status check                                       |
| POST   | `/predict`  | `{review_text: "..."}` → `{predicted_score: int}`          |
| POST   | `/feedback` | Logs corrected labels for online learning (feedback.csv)  |

---

## Running the Frontend (Utility Application)

Run the Streamlit app **in a second terminal** from the project root:

```bash
streamlit run client/streamlit_app.py
```

This opens a local UI (usually at):

```text
http://localhost:8501
```

### Usage

1. Enter a product review text.
2. Click **Predict rating** to get a predicted score (1–5).
3. Adjust the **True rating** slider if needed.
4. Click **Submit feedback** to log the corrected label.

All feedback is saved to `data/feedback.csv`.

---

## Online Learning (Retraining with Feedback)

The model supports *online learning* by periodically retraining using user feedback.

To retrain on the union of the original dataset and all collected feedback samples:

```bash
python training/retrain_with_feedback.py
```

This script:

1. Loads `data/cleaned_data.csv` (original dataset)
2. Loads `data/feedback.csv` (logged user corrections, if any)
3. Combines both datasets
4. Recomputes:
   - TF–IDF vectorizer  
   - `len_words` scaler  
   - LinearSVC classifier  
5. Overwrites all model artifacts under `model_artifacts/`

After retraining, restart the backend to use the updated model:

```bash
uvicorn app.api:app --reload
```

---

## Initial Model Training (Optional)

If you want to retrain the *initial* version of the model from scratch using only the original dataset:

```bash
python training/train_initial.py
```

This uses `cleaned_data.csv` and recreates the three artifacts:

- `linear_svc_model.joblib`
- `tfidf_vectorizer.joblib`
- `scaler_len_words.joblib`

---

## Model Pipeline Summary

**Input**  
A single string containing a product review.

**Processing**

1. TF–IDF vectorizer transforms text into a high-dimensional sparse feature vector.
2. `len_words` = number of words in the review.
3. `len_words` is standardised using `StandardScaler`.
4. Text features and `len_words` are concatenated using a sparse horizontal stack.
5. A `LinearSVC` classifier predicts an integer rating in `{1, 2, 3, 4, 5}`.

**Output**  
JSON containing the predicted score, e.g. `{ "predicted_score": 4 }`.

---

## Notes

- The system follows a **client–server architecture**: the Streamlit frontend communicates with the FastAPI backend via HTTP and does not depend directly on the model implementation.
- All model artefacts are stored under `model_artifacts/` and can be regenerated from the training scripts.
- Retraining can be scheduled (e.g., daily) to simulate a continuous online-learning loop.

---

## Authors

- Beshoy Botros
- David Boules
