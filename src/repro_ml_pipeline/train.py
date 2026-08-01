"""Train, evaluate, log, and register a versioned sklearn classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from repro_ml_pipeline.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_MANIFEST_PATH,
    DatasetBundle,
    load_versioned_dataset,
)
from repro_ml_pipeline.registry import register_model_uri
from repro_ml_pipeline.signature import compute_signature, signature_payload, write_signature

# Local demos/CI may still hit file-store paths; opt in for newer MLflow releases.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

DEFAULT_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 500,
    "solver": "lbfgs",
}


def load_demo_xy(random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning the deterministic training subset."""
    data = load_versioned_dataset()
    X_train, _, y_train, _ = train_test_split(
        data.features,
        data.target,
        test_size=0.25,
        random_state=random_state,
        stratify=data.target,
    )
    return X_train, y_train


def build_pipeline(params: dict[str, Any] | None = None, random_state: int = 42) -> Pipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    clf = LogisticRegression(
        C=float(p["C"]),
        max_iter=int(p["max_iter"]),
        solver=str(p["solver"]),
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _normalize_tracking_uri(tracking_uri: str, artifact_dir: Path) -> str:
    """Prefer sqlite for local runs so CI stays compatible with current MLflow."""
    if tracking_uri.startswith("file:"):
        db = artifact_dir / "mlflow.db"
        return f"sqlite:///{db.as_posix()}"
    return tracking_uri


def train_and_log(  # noqa: PLR0913, PLR0917
    tracking_uri: str,
    experiment: str,
    artifact_dir: str | Path,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    data_path: str | Path = DEFAULT_DATA_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    model_name: str = "repro-ml-classifier",
    alias: str = "champion",
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetBundle = load_versioned_dataset(data_path, manifest_path)
    X, y = dataset.features, dataset.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    merged = {**DEFAULT_PARAMS, **(params or {})}
    pipe = build_pipeline(merged, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_validate(
        pipe,
        X_tr,
        y_tr,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        n_jobs=1,
    )
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
        "cv_f1_std": float(np.std(cv_scores["test_f1"])),
    }

    signature_kwargs = {
        "dataset_sha256": dataset.content_sha256,
        "dataset_version": str(dataset.manifest["version"]),
    }
    digest = compute_signature(X, y, merged, random_state, **signature_kwargs)
    meta = signature_payload(X, y, merged, random_state, **signature_kwargs)
    sig_path = artifact_dir / "run_signature.json"
    write_signature(sig_path, digest, meta)

    model_path = artifact_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    resolved = _normalize_tracking_uri(tracking_uri, artifact_dir)
    mlflow.set_tracking_uri(resolved)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**merged, "random_state": random_state})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("run_signature", digest)
        mlflow.set_tag("dataset_sha256", dataset.content_sha256)
        mlflow.set_tag("code_revision", meta["code"]["revision"])
        mlflow.log_artifact(str(sig_path))
        model_info = mlflow.sklearn.log_model(pipe, name="model")
        run_id = run.info.run_id

    registered = register_model_uri(model_info.model_uri, model_name, alias)
    summary = {
        "run_id": run_id,
        "metrics": metrics,
        "signature": digest,
        "model_path": model_path.as_posix(),
        "signature_path": sig_path.as_posix(),
        "tracking_uri": resolved,
        "model_uri": registered.uri,
        "model_name": registered.name,
        "model_version": registered.version,
        "model_alias": registered.alias,
        "dataset_sha256": dataset.content_sha256,
        "code_revision": meta["code"]["revision"],
    }
    (artifact_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
