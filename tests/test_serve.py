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


# Finding 8: a failed metadata lookup must not leave the model half loaded.
def test_ready_stays_unavailable_while_metadata_lookup_fails(monkeypatch):
    calls = {"metadata": 0}

    def failing_metadata(model_uri: str) -> dict[str, str]:
        calls["metadata"] += 1
        raise RuntimeError("no such alias")

    monkeypatch.setattr(serve, "load_model", lambda uri: object())
    monkeypatch.setattr(serve, "model_version_metadata", failing_metadata)
    monkeypatch.setattr(serve, "manager", serve.ModelManager("models:/fixture/3"))
    client = TestClient(serve.app)

    assert client.get("/ready").status_code == 503
    assert client.get("/ready").status_code == 503
    assert client.get("/metadata").status_code == 503
    assert calls["metadata"] == 3


def test_version_uri_metadata_uses_get_model_version(monkeypatch):
    from types import SimpleNamespace

    from repro_ml_pipeline import registry

    class FakeClient:
        @staticmethod
        def get_model_version(name: str, version: str):
            assert (name, version) == ("fixture", "3")
            return SimpleNamespace(version="3", run_id="run-3")

        @staticmethod
        def get_model_version_by_alias(name: str, alias: str):
            raise AssertionError("a version URI has no alias")

    monkeypatch.setattr(registry, "MlflowClient", FakeClient)
    meta = registry.model_version_metadata("models:/fixture/3")
    assert meta["model_name"] == "fixture"
    assert meta["model_version"] == "3"
    assert meta["run_id"] == "run-3"
