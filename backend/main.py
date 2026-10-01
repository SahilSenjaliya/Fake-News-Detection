"""
FastAPI Backend for Fake News Detection.

Serves the trained LinearSVC classification model and TF-IDF vectorizer
via high-performance RESTful API endpoints.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path to import modules from model/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.preprocessing import clean_text, combine_and_clean

# Global references for loaded artifacts
model = None
vectorizer = None


def load_artifacts():
    """
    Loads the serialized LinearSVC model and TF-IDF vectorizer from the model directory.
    """
    global model, vectorizer
    model_path = os.path.join(BASE_DIR, "model", "best_model.joblib")
    vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"TF-IDF vectorizer file not found at: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"[STARTUP] Successfully loaded model from: {model_path}")
    print(f"[STARTUP] Successfully loaded vectorizer from: {vectorizer_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown events.
    Loads the ML model and vectorizer into memory before accepting requests.
    """
    try:
        load_artifacts()
    except Exception as e:
        print(f"[ERROR] Failed to load ML artifacts: {e}", file=sys.stderr)
    yield
    print("[SHUTDOWN] Releasing FastAPI server resources.")


# Initialize FastAPI application
app = FastAPI(
    title="Fake News Detection API",
    description="Machine Learning REST API for classifying news articles as Real or Fake",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# CORS Middleware: Allows requests from React/frontend origins
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development; can restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------
class NewsRequest(BaseModel):
    text: str = Field(..., description="The main text or content of the news article")
    title: Optional[str] = Field(None, description="Optional title or headline of the news article")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "The White House announced a bipartisan agreement on infrastructure spending today."
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Classification outcome: 'Real' or 'Fake'")
    confidence: float = Field(..., description="Calibrated confidence score between 0.50 and 1.00")
    decision_score: Optional[float] = Field(None, description="Raw hyperplane distance from LinearSVC")


# ---------------------------------------------------------
# Custom Error Handlers
# ---------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles malformed or missing JSON requests with a clear, readable error response.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid JSON format or missing required fields",
            "details": exc.errors()
        }
    )


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", summary="Health Check")
async def health_check():
    """
    Health check endpoint to verify that the API server is operational.
    """
    is_ready = model is not None and vectorizer is not None
    return {
        "status": "ok",
        "model_loaded": is_ready
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict News Authenticity")
async def predict_news(request: NewsRequest):
    """
    Classifies a news article as either Real or Fake using the trained LinearSVC model.
    Steps:
      1. Validates non-empty input.
      2. Cleans text with the same preprocessing pipeline as training.
      3. Transforms text to TF-IDF feature space.
      4. Calculates LinearSVC decision function and calibrated confidence score.
      5. Returns prediction ('Fake' or 'Real') and confidence.
    """
    # 1. Verify model and vectorizer are loaded
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model or vectorizer is not currently loaded."
        )

    # 2. Check for empty or whitespace-only text
    raw_text = request.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input 'text' cannot be empty or contain only whitespace."
        )

    # 3. Clean the text using the exact preprocessing function
    if request.title and request.title.strip():
        cleaned = combine_and_clean(request.title, raw_text)
    else:
        cleaned = clean_text(raw_text)

    # Handle case where text consists solely of removed special characters
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text contains no valid alphanumeric words after cleaning."
        )

    # 4. Transform using the loaded TF-IDF vectorizer
    try:
        text_vector = vectorizer.transform([cleaned])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vectorization failed: {str(e)}"
        )

    # 5. Predict with LinearSVC
    try:
        # decision_function gives distance to the separating hyperplane:
        # Positive (>0) corresponds to Class 1 (Real)
        # Negative (<0) corresponds to Class 0 (Fake)
        decision_score = float(model.decision_function(text_vector)[0])
        prediction_int = int(model.predict(text_vector)[0])

        # Map binary label to string representation
        prediction_label = "Real" if prediction_int == 1 else "Fake"

        # Calculate calibrated confidence using the standard logistic sigmoid:
        # P(Real) = 1 / (1 + exp(-decision_score))
        # P(Fake) = 1 / (1 + exp(decision_score))
        # Confidence reflects the model's certainty in its chosen prediction:
        calibrated_confidence = float(1.0 / (1.0 + np.exp(-abs(decision_score))))

        return PredictionResponse(
            prediction=prediction_label,
            confidence=round(calibrated_confidence, 4),
            decision_score=round(decision_score, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )


# ---------------------------------------------------------
# Mount Frontend Static Files (serves UI at http://127.0.0.1:8000/)
# ---------------------------------------------------------
from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Render provides PORT in the environment; defaults to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    # Listen on 0.0.0.0 to accept external requests on cloud environments
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
