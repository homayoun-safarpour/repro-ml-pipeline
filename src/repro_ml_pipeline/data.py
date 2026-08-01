"""Versioned dataset loading and quality checks."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_DATA_PATH = RESOURCE_DIR / "breast_cancer_v1.csv"
DEFAULT_MANIFEST_PATH = RESOURCE_DIR / "breast_cancer_v1.manifest.json"


@dataclass(frozen=True)
class DatasetBundle:
    features: np.ndarray
    target: np.ndarray
    feature_names: tuple[str, ...]
    manifest: dict[str, object]
    content_sha256: str


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_versioned_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> DatasetBundle:
    data_path = Path(data_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest = sha256_file(data_path)
    if digest != manifest["sha256"]:
        raise ValueError(
            f"dataset hash mismatch: expected {manifest['sha256']}, actual {digest}"
        )

    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader]

    features = np.asarray([[float(value) for value in row[:-1]] for row in rows])
    target = np.asarray([int(row[-1]) for row in rows])
    feature_names = tuple(header[:-1])
    validate_dataset(features, target, manifest)
    return DatasetBundle(features, target, feature_names, manifest, digest)


def validate_dataset(
    features: np.ndarray,
    target: np.ndarray,
    manifest: dict[str, object],
) -> None:
    expected_shape = (int(manifest["rows"]), int(manifest["features"]))
    if features.shape != expected_shape:
        raise ValueError(
            f"dataset shape mismatch: expected {expected_shape}, actual {features.shape}"
        )
    if target.shape != (expected_shape[0],):
        raise ValueError(f"target shape mismatch: {target.shape}")
    if not np.isfinite(features).all():
        raise ValueError("dataset contains non-finite feature values")
    classes = set(np.unique(target).tolist())
    if classes != {0, 1}:
        raise ValueError(f"expected binary target classes {{0, 1}}, actual {classes}")
