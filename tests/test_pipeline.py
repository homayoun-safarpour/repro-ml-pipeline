from __future__ import annotations

from pathlib import Path

from repro_ml_pipeline.signature import check_signature, compute_signature
from repro_ml_pipeline.train import DEFAULT_PARAMS, load_demo_xy, train_and_log

ROOT = Path(__file__).resolve().parents[1]


def test_signature_stable_for_fixed_seed():
    X, y = load_demo_xy(random_state=42)
    a = compute_signature(X, y, DEFAULT_PARAMS, 42)
    b = compute_signature(X, y, DEFAULT_PARAMS, 42)
    assert a == b
    assert len(a) == 64


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
    assert len(summary["signature"]) == 64
