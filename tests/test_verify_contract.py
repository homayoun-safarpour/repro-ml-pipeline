"""The verify-signature contract: verdict words, exit codes and the committed pin."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from repro_ml_pipeline.cli import main
from repro_ml_pipeline.data import DEFAULT_DATA_PATH, DEFAULT_MANIFEST_PATH, load_versioned_dataset
from repro_ml_pipeline.signature import compute_signature, signature_payload, write_signature
from repro_ml_pipeline.train import DEFAULT_PARAMS

ROOT = Path(__file__).resolve().parents[1]
SRC_PKG = ROOT / "src" / "repro_ml_pipeline"
COMMITTED_PIN = ROOT / "examples" / "artifacts" / "run_signature.json"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _write_pin(path: Path, seed: int = 42, params: dict | None = None) -> Path:
    data = load_versioned_dataset()
    params = dict(params or DEFAULT_PARAMS)
    kwargs = {
        "dataset_sha256": data.content_sha256,
        "dataset_version": str(data.manifest["version"]),
    }
    digest = compute_signature(data.features, data.target, params, seed, **kwargs)
    meta = signature_payload(data.features, data.target, params, seed, **kwargs)
    write_signature(path, digest, meta)
    return path


def _changed_csv(tmp_path: Path) -> Path:
    lines = DEFAULT_DATA_PATH.read_text(encoding="utf-8").split("\n")
    cells = lines[1].split(",")
    cells[0] = str(float(cells[0]) + 0.01)
    lines[1] = ",".join(cells)
    path = tmp_path / "changed.csv"
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return path


# Finding 1: CI must check a committed pin, not one it wrote in the same job.
def test_ci_verifies_against_the_committed_pin_before_training():
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    committed = "repro-ml verify-signature --pin examples/artifacts/run_signature.json"
    assert committed in text
    assert text.index(committed) < text.index("repro-ml train")
    for line in text.splitlines():
        if "verify-signature" in line:
            assert "examples/ci-" not in line, line


# Finding 2: the bundled pin must verify, so it cannot go stale unnoticed.
def test_committed_pin_verifies_with_exit_0(capsys):
    assert main(["verify-signature", "--pin", str(COMMITTED_PIN)]) == 0
    assert capsys.readouterr().out == "verdict: PASS\n"


@pytest.mark.parametrize(
    ("rel_path", "field"),
    [
        (None, None),
        ("cli.py", "code.revision"),
        ("signature.py", "code.revision"),
        ("resources/uv.lock", "environment.lock_sha256"),
    ],
)
def test_edit_to_package_source_or_lock_makes_committed_pin_fail(tmp_path, rel_path, field):
    pkg = tmp_path / "repro_ml_pipeline"
    shutil.copytree(SRC_PKG, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    if rel_path is not None:
        with (pkg / rel_path).open("ab") as handle:
            handle.write(b"\n# one edit\n")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    run = subprocess.run(
        [sys.executable, "-m", "repro_ml_pipeline.cli", "verify-signature",
         "--pin", str(COMMITTED_PIN)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if rel_path is None:
        assert run.returncode == 0, run.stdout + run.stderr
        assert run.stdout == "verdict: PASS\n"
    else:
        assert run.returncode == 2, run.stdout + run.stderr
        assert run.stdout.startswith("verdict: SIGNATURE_MISMATCH\n")
        assert f"  changed: {field}\n" in run.stdout


# Finding 3: a data change is a clean SIGNATURE_MISMATCH with exit 2.
def test_data_change_is_signature_mismatch_exit_2_not_a_traceback(tmp_path, capsys):
    code = main(["verify-signature", "--pin", str(COMMITTED_PIN),
                 "--data", str(_changed_csv(tmp_path))])
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("verdict: SIGNATURE_MISMATCH\n")
    assert "  data: dataset hash mismatch" in out


def test_data_change_with_updated_manifest_is_signature_mismatch(tmp_path, capsys):
    csv_path = _changed_csv(tmp_path)
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "changed.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code = main(["verify-signature", "--pin", str(COMMITTED_PIN),
                 "--data", str(csv_path), "--manifest", str(manifest_path)])
    out = capsys.readouterr().out
    assert code == 2
    assert "  changed: data.content_sha256\n" in out


def test_missing_data_file_is_bad_input_exit_1(tmp_path, capsys):
    code = main(["verify-signature", "--pin", str(COMMITTED_PIN),
                 "--data", str(tmp_path / "absent.csv")])
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("verdict: BAD_INPUT\n")


def test_param_change_in_pin_is_signature_mismatch(tmp_path, capsys):
    pin = json.loads(COMMITTED_PIN.read_text(encoding="utf-8"))
    pin["meta"]["params"]["C"] = 0.5
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(pin), encoding="utf-8")
    # The check uses the pin's own params, so no field differs, but the digest does.
    assert main(["verify-signature", "--pin", str(path)]) == 2
    assert capsys.readouterr().out.startswith("verdict: SIGNATURE_MISMATCH\n")


def test_seed_override_that_differs_from_pin_is_signature_mismatch(capsys):
    assert main(["verify-signature", "--pin", str(COMMITTED_PIN), "--random-state", "7"]) == 2
    assert "  changed: seed\n" in capsys.readouterr().out


# Finding 4: the pin's meta must agree with the recomputed run, field by field.
def test_pin_meta_seed_edit_is_signature_mismatch(tmp_path, capsys):
    pin_path = _write_pin(tmp_path / "seed7.json", seed=7)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["meta"]["seed"] = 99
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    code = main(["verify-signature", "--pin", str(pin_path), "--random-state", "7"])
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("verdict: SIGNATURE_MISMATCH\n")
    assert "  changed: seed\n" in out


# Finding 5: the seed defaults to the pin's seed.
def test_verify_uses_the_pin_seed_by_default(tmp_path, capsys):
    pin_path = _write_pin(tmp_path / "seed7.json", seed=7)
    assert main(["verify-signature", "--pin", str(pin_path)]) == 0
    assert capsys.readouterr().out == "verdict: PASS\n"


# Finding 6: a bad or missing pin is BAD_PIN with exit 1, never a traceback.
@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        "[]",
        "{}",
        json.dumps({"signature": "abc", "meta": {"params": {}, "seed": 42}}),
        json.dumps({"signature": "0" * 64, "meta": {"seed": 42}}),
        json.dumps({"signature": "0" * 64, "meta": {"params": {}, "seed": "42"}}),
        json.dumps({"signature": "0" * 64, "meta": {"params": {}, "seed": True}}),
    ],
    ids=["missing", "not-json", "list", "empty", "short-signature", "no-params",
         "string-seed", "bool-seed"],
)
def test_bad_or_missing_pin_is_bad_pin_exit_1(tmp_path, capsys, content):
    path = tmp_path / "pin.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    code = main(["verify-signature", "--pin", str(path)])
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("verdict: BAD_PIN\n")
    assert len(out.splitlines()) == 2


# Finding 7: a usage error must not share exit 2 with SIGNATURE_MISMATCH.
def test_usage_error_exits_1_not_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["verify-signature"])
    assert exc.value.code == 1
    assert "--pin" in capsys.readouterr().err


def test_reliability_card_states_the_real_exit_codes():
    card = (ROOT / "docs" / "RELIABILITY_CARD.md").read_text(encoding="utf-8")
    assert "metric-floor breach" not in card
    for word in ("`SIGNATURE_MISMATCH`", "`BAD_PIN`", "`BAD_INPUT`"):
        assert word in card
