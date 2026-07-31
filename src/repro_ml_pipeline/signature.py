"""Dataset + params signature for reproducibility gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def signature_payload(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "x_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        "params": params,
        "random_state": int(random_state),
    }


def compute_signature(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    random_state: int,
) -> str:
    payload = signature_payload(X, y, params, random_state)
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
