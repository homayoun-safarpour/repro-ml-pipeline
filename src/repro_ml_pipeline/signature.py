"""Full data, environment, code, parameter, and seed run signatures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from repro_ml_pipeline.data import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_LOCKFILE = RESOURCE_DIR / "uv.lock"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent


def source_revision(source_dir: str | Path = DEFAULT_SOURCE_DIR) -> str:
    """Hash tracked Python source content into a stable code revision."""
    source_dir = Path(source_dir)
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*.py")):
        digest.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def signature_payload(  # noqa: PLR0913
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
    *,
    dataset_sha256: str | None = None,
    dataset_version: str = "unknown",
    lockfile: str | Path = DEFAULT_LOCKFILE,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
) -> dict[str, Any]:
    lockfile = Path(lockfile)
    return {
        "schema_version": 2,
        "data": {
            "version": dataset_version,
            "content_sha256": dataset_sha256
            or hashlib.sha256(
                np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()
            ).hexdigest(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "environment": {
            "lockfile": lockfile.name,
            "lock_sha256": sha256_file(lockfile),
            "python_requires": ">=3.10,<3.13",
        },
        "code": {"revision": source_revision(source_dir)},
        "params": params,
        "seed": int(random_state),
    }


def compute_signature(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
    **kwargs: Any,
) -> str:
    payload = signature_payload(X, y, params, random_state, **kwargs)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_signature(path: str | Path, digest: str, meta: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps({"signature": digest, "meta": meta}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_signature(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_signature(expected: str, actual: str) -> tuple[str, int]:
    if expected != actual:
        return "SIGNATURE_MISMATCH", 2
    return "PASS", 0
