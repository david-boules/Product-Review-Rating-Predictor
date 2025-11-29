"""
Streamlit client application.

- Lets the user type a review.
- Calls the FastAPI /predict endpoint to get a rating.
- Shows the predicted rating.
- Lets the user provide the true rating as feedback.
- Sends feedback to the /feedback endpoint for online learning.
"""

import streamlit as st
import requests         # used to send HTTP requests from Streamlit to FastAPI

# URL of the API server (FastAPI). If running locally, it's localhost:8000.
API_URL = "http://127.0.0.1:8000"

st.title("ChatGPT Review Rating Predictor")

st.write(
    "Enter a ChatGPT app review and get a predicted rating (1–5 stars).\n"
    "Optionally, provide the true rating as feedback to improve the model over time."
)

# # Text input area for the review
review = st.text_area("Review text:", height=200)

if st.button("Predict rating"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            # POST request to FastAPI
            resp = requests.post(f"{API_URL}/predict", json={"review_text": review})
            # Raise an exception if HTTP status is not 2xx
            resp.raise_for_status()
            
            data = resp.json()  # parse JSON response into Python dict
            
            st.session_state["last_pred"] = data["predicted_score"]
            st.success(f"Predicted rating: **{data['predicted_score']}**")
        except Exception as e:
            st.error(f"Error calling API: {e}")

# Feedback section (for online learning)
if "last_pred" in st.session_state:
    st.subheader("Was this prediction correct?")
    true_score = st.slider("True rating (1–5)", 1, 5, st.session_state["last_pred"])
    if st.button("Submit feedback"):
        try:
            payload = {
                "review_text": review,
                "true_score": true_score,
                "model_score": st.session_state["last_pred"]
            }
            # POST feedback to API
            resp = requests.post(f"{API_URL}/feedback", json=payload)
            resp.raise_for_status()
            
            st.success("Thank you! Your feedback will be used to improve the model.")
        except Exception as e:
            st.error(f"Error sending feedback: {e}")
