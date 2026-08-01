"""Export the pinned sklearn breast-cancer fixture as a versioned CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "breast_cancer_v1.csv"
MANIFEST_PATH = DATA_DIR / "breast_cancer_v1.manifest.json"
RESOURCE_DIR = ROOT / "src" / "repro_ml_pipeline" / "resources"


def main() -> None:
    dataset = load_breast_cancer()
    DATA_DIR.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*dataset.feature_names.tolist(), "target"])
        for features, target in zip(dataset.data, dataset.target, strict=True):
            writer.writerow([*[format(float(value), ".17g") for value in features], int(target)])

    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "sklearn-breast-cancer-wisconsin-diagnostic",
        "version": "1",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "license": "CC BY 4.0",
        "file": CSV_PATH.name,
        "sha256": digest,
        "rows": int(dataset.data.shape[0]),
        "features": int(dataset.data.shape[1]),
        "target": "target",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CSV_PATH, RESOURCE_DIR / CSV_PATH.name)
    shutil.copyfile(MANIFEST_PATH, RESOURCE_DIR / MANIFEST_PATH.name)
    shutil.copyfile(ROOT / "uv.lock", RESOURCE_DIR / "uv.lock")
    print(f"{CSV_PATH.relative_to(ROOT)} sha256={digest}")


if __name__ == "__main__":
    main()
