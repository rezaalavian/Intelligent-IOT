from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn

app = FastAPI(title="Intelligent-IOT Forecast API")


class PredictRequest(BaseModel):
    station_id: str
    timestamp: str
    features: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # Placeholder: load model from models/saved_models and run prediction.
    return {"station_id": req.station_id, "timestamp": req.timestamp, "prediction": None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
