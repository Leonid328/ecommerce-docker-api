import os
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/recommendation_model.pkl")
model = None

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

app = FastAPI(title="Ecommerce prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    monetary: float = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    purchase_probability: float
    will_purchase: int
    features_used: List[str]
    model_path: Optional[str] = None


@app.get("/health")
def health():
    ready = model is not None
    return {
        "status": "healthy" if ready else "degraded",
        "service": "ecommerce-prediction-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": ready,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    frame = pd.DataFrame([payload.model_dump()])
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(frame)[0][1])
        else:
            prob = float(model.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(
        purchase_probability=round(prob, 4),
        will_purchase=int(prob >= 0.5),
        features_used=list(frame.columns),
        model_path=MODEL_PATH,
    )
