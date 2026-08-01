from __future__ import annotations

from fastapi.testclient import TestClient

from repro_ml_pipeline import serve


class FakeManager:
    model_uri = "models:/fixture@champion"
    metadata = {
        "model_uri": model_uri,
        "model_version": "7",
        "run_signature": "abc123",
    }

    @staticmethod
    def load() -> None:
        return None

    @staticmethod
    def predict(features: list[float]) -> int:
        return int(sum(features) > 0)


def test_api_health_readiness_metadata_and_registry_prediction(monkeypatch):
    monkeypatch.setattr(serve, "manager", FakeManager())
    client = TestClient(serve.app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_uri"] == FakeManager.model_uri
    assert client.get("/metadata").json()["model_version"] == "7"
    response = client.post("/predict", json={"features": [1.0] * 30})
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "model_uri": FakeManager.model_uri,
        "run_signature": "abc123",
    }


def test_api_rejects_wrong_feature_count(monkeypatch):
    monkeypatch.setattr(serve, "manager", FakeManager())
    response = TestClient(serve.app).post("/predict", json={"features": [1.0]})
    assert response.status_code == 422
