from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_train_and_verify(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    mlruns = tmp_path / "mlruns"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro_ml_pipeline.cli",
            "train",
            "--tracking-uri",
            f"file:{mlruns}",
            "--artifact-dir",
            str(artifacts),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert train.returncode == 0, train.stderr
    pin = artifacts / "run_signature.json"
    assert pin.exists()
    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro_ml_pipeline.cli",
            "verify-signature",
            "--pin",
            str(pin),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "PASS" in verify.stdout

    summary = json.loads((artifacts / "train_summary.json").read_text(encoding="utf-8"))
    predict = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro_ml_pipeline.cli",
            "predict",
            "--tracking-uri",
            summary["tracking_uri"],
            "--model-uri",
            summary["model_uri"],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert predict.returncode == 0, predict.stdout + predict.stderr
    result = json.loads(predict.stdout)
    assert result["prediction"] in {0, 1}
    assert result["model_uri"] == summary["model_uri"]

    # Tamper pin
    data = json.loads(pin.read_text(encoding="utf-8"))
    data["signature"] = "0" * 64
    bad = tmp_path / "bad_pin.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    fail = subprocess.run(
        [
            sys.executable,
            "-m",
            "repro_ml_pipeline.cli",
            "verify-signature",
            "--pin",
            str(bad),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail.returncode == 2
    assert "SIGNATURE_MISMATCH" in fail.stdout
