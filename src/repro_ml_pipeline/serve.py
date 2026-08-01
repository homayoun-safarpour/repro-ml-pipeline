"""FastAPI serving for a versioned MLflow registry model."""

from __future__ import annotations

import json
import logging
import os
from threading import Lock
from typing import Annotated, Any

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from repro_ml_pipeline.registry import load_model, model_version_metadata

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
LOGGER = logging.getLogger("repro_ml_pipeline.serve")

FEATURE_COUNT = 30


class PredictionRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)]


class PredictionResponse(BaseModel):
    prediction: int
    model_uri: str
    run_signature: str | None = None


class ModelManager:
    def __init__(self, model_uri: str) -> None:
        self.model_uri = model_uri
        self._model: Any | None = None
        self._metadata: dict[str, str] | None = None
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._model = load_model(self.model_uri)
            metadata = model_version_metadata(self.model_uri)
            run_id = metadata.get("run_id")
            if run_id:
                run = mlflow.MlflowClient().get_run(run_id)
                metadata["run_signature"] = run.data.tags.get("run_signature", "")
                metadata["code_revision"] = run.data.tags.get("code_revision", "")
                metadata["dataset_sha256"] = run.data.tags.get("dataset_sha256", "")
            self._metadata = metadata
            LOGGER.info(json.dumps({"event": "model_loaded", **metadata}, sort_keys=True))

    @property
    def metadata(self) -> dict[str, str]:
        self.load()
        return dict(self._metadata or {})

    def predict(self, features: list[float]) -> int:
        self.load()
        assert self._model is not None
        prediction = self._model.predict(np.asarray([features], dtype=float))
        return int(prediction[0])


tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
manager = ModelManager(os.getenv("MODEL_URI", "models:/repro-ml-classifier@champion"))
app = FastAPI(title="Repro ML Pipeline", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness() -> dict[str, str]:
    try:
        manager.load()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
    return {"status": "ready", "model_uri": manager.model_uri}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    try:
        return manager.metadata
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = manager.predict(request.features)
        meta = manager.metadata
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
    LOGGER.info(
        json.dumps(
            {"event": "prediction", "model_uri": manager.model_uri, "prediction": prediction},
            sort_keys=True,
        )
    )
    return PredictionResponse(
        prediction=prediction,
        model_uri=manager.model_uri,
        run_signature=meta.get("run_signature"),
    )
