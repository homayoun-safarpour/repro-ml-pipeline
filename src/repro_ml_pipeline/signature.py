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


class PinError(ValueError):
    """The pin file is missing, unreadable or not a valid pin."""


_HEX = frozenset("0123456789abcdef")
_DIGEST_LENGTH = 64


def load_pin(path: str | Path) -> dict[str, Any]:
    """Load a pin and check its shape; raise PinError with a one-line reason."""
    try:
        pin = load_signature(path)
    except OSError as exc:
        raise PinError(f"cannot read pin {path}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PinError(f"pin {path} is not valid JSON: {exc}") from exc
    if not isinstance(pin, dict):
        raise PinError(f"pin {path} must be a JSON object")
    signature = pin.get("signature")
    is_digest = (
        isinstance(signature, str)
        and len(signature) == _DIGEST_LENGTH
        and set(signature) <= _HEX
    )
    if not is_digest:
        raise PinError(f"pin {path} needs a 64-character lowercase hex 'signature'")
    meta = pin.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("params"), dict):
        raise PinError(f"pin {path} needs a 'meta' object with a 'params' object")
    seed = meta.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PinError(f"pin {path} needs an integer 'meta.seed'")
    return pin


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def changed_fields(pinned: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Dotted names of fields that differ between the pin's meta and the recomputed payload."""
    old, new = _flatten(pinned), _flatten(current)
    return sorted(
        key
        for key in old.keys() | new.keys()
        if key not in old or key not in new or old[key] != new[key]
    )


def check_signature(expected: str, actual: str) -> tuple[str, int]:
    if expected != actual:
        return "SIGNATURE_MISMATCH", 2
    return "PASS", 0
