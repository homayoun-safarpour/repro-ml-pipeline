"""MLflow Model Registry operations."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient


@dataclass(frozen=True)
class RegisteredModel:
    name: str
    version: str
    alias: str

    @property
    def uri(self) -> str:
        return f"models:/{self.name}@{self.alias}"


def register_model_uri(
    model_uri: str,
    model_name: str,
    alias: str = "champion",
) -> RegisteredModel:
    result = mlflow.register_model(model_uri, model_name)
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, result.version)
    return RegisteredModel(model_name, str(result.version), alias)


def load_model(model_uri: str):
    """Load inference from a versioned MLflow model URI."""
    return mlflow.pyfunc.load_model(model_uri)


def model_version_metadata(model_uri: str) -> dict[str, str]:
    if not model_uri.startswith("models:/"):
        return {"model_uri": model_uri}
    suffix = model_uri.removeprefix("models:/")
    name, _, alias = suffix.partition("@")
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    return {
        "model_uri": model_uri,
        "model_name": name,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "alias": alias,
    }
