from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro_ml_pipeline.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_MANIFEST_PATH,
    load_versioned_dataset,
)
from repro_ml_pipeline.signature import (
    DEFAULT_LOCKFILE,
    check_signature,
    compute_signature,
    signature_payload,
)
from repro_ml_pipeline.train import DEFAULT_PARAMS, load_demo_xy, train_and_log

ROOT = Path(__file__).resolve().parents[1]


def test_signature_stable_for_fixed_seed():
    X, y = load_demo_xy(random_state=42)
    a = compute_signature(X, y, DEFAULT_PARAMS, 42)
    b = compute_signature(X, y, DEFAULT_PARAMS, 42)
    assert a == b
    assert len(a) == 64


def test_full_signature_covers_data_environment_code_params_and_seed():
    dataset = load_versioned_dataset()
    payload = signature_payload(
        dataset.features,
        dataset.target,
        DEFAULT_PARAMS,
        42,
        dataset_sha256=dataset.content_sha256,
        dataset_version=str(dataset.manifest["version"]),
    )
    assert set(payload) == {"schema_version", "data", "environment", "code", "params", "seed"}
    assert payload["data"]["content_sha256"] == dataset.content_sha256
    assert payload["environment"]["lockfile"] == "uv.lock"
    assert payload["code"]["revision"].startswith("sha256:")
    assert payload["seed"] == 42
    assert DEFAULT_LOCKFILE.read_bytes() == (ROOT / "uv.lock").read_bytes()


def test_signature_changes_when_params_change():
    X, y = load_demo_xy(random_state=42)
    a = compute_signature(X, y, DEFAULT_PARAMS, 42)
    b = compute_signature(X, y, {**DEFAULT_PARAMS, "C": 0.1}, 42)
    assert a != b


def test_check_signature_mismatch_exit_semantics():
    status, code = check_signature("abc", "def")
    assert status == "SIGNATURE_MISMATCH"
    assert code == 2
    status, code = check_signature("abc", "abc")
    assert status == "PASS"
    assert code == 0


def test_data_quality_rejects_content_drift(tmp_path: Path):
    changed = tmp_path / "changed.csv"
    changed.write_bytes(DEFAULT_DATA_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        load_versioned_dataset(changed, DEFAULT_MANIFEST_PATH)


def test_train_and_log_writes_artifacts(tmp_path: Path):
    tracking = tmp_path / "mlruns"
    artifacts = tmp_path / "artifacts"
    summary = train_and_log(
        tracking_uri=f"file:{tracking}",
        experiment="test-exp",
        artifact_dir=artifacts,
        random_state=42,
    )
    assert (artifacts / "model.joblib").exists()
    assert (artifacts / "run_signature.json").exists()
    assert (artifacts / "train_summary.json").exists()
    assert summary["metrics"]["accuracy"] > 0.9
    assert summary["metrics"]["cv_accuracy_mean"] > 0.9
    assert len(summary["signature"]) == 64
    assert summary["model_uri"] == "models:/repro-ml-classifier@champion"
    assert summary["model_version"] == "1"


def test_metric_regression_floor(tmp_path: Path):
    summary = train_and_log(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment="metric-floor",
        artifact_dir=tmp_path / "artifacts",
        random_state=42,
        model_name="metric-floor-model",
    )
    assert summary["metrics"]["accuracy"] >= 0.95
    assert summary["metrics"]["cv_f1_mean"] >= 0.95
    signature = json.loads((tmp_path / "artifacts" / "run_signature.json").read_text())
    assert signature["meta"]["data"]["version"] == "1"
