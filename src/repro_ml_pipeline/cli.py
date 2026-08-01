"""CLI: train, verify-signature, and registry-backed predict."""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlflow

from repro_ml_pipeline.data import DEFAULT_DATA_PATH, DEFAULT_MANIFEST_PATH, load_versioned_dataset
from repro_ml_pipeline.registry import load_model
from repro_ml_pipeline.signature import check_signature, compute_signature, load_signature
from repro_ml_pipeline.train import train_and_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repro-ml",
        description="Train sklearn models with MLflow and verify run signatures.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train demo classifier and log to MLflow")
    tr.add_argument("--tracking-uri", default="sqlite:///examples/artifacts/mlflow.db")
    tr.add_argument("--experiment", default="repro-ml-demo")
    tr.add_argument("--artifact-dir", default="examples/artifacts")
    tr.add_argument("--random-state", type=int, default=42)
    tr.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    tr.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    tr.add_argument("--model-name", default="repro-ml-classifier")
    tr.add_argument("--alias", default="champion")
    tr.add_argument("--C", type=float, default=1.0)

    vf = sub.add_parser("verify-signature", help="fail if signature does not match pin")
    vf.add_argument("--pin", required=True, help="path to pinned signature JSON")
    vf.add_argument("--random-state", type=int, default=42)
    vf.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    vf.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))

    pr = sub.add_parser("predict", help="score a row through a versioned MLflow model URI")
    pr.add_argument(
        "--model-uri",
        default=os.getenv("MODEL_URI", "models:/repro-ml-classifier@champion"),
    )
    pr.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))

    args = parser.parse_args(argv)

    if args.cmd == "train":
        summary = train_and_log(
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
            artifact_dir=args.artifact_dir,
            random_state=args.random_state,
            data_path=args.data,
            manifest_path=args.manifest,
            params={"C": args.C},
            model_name=args.model_name,
            alias=args.alias,
        )
        print(
            f"run_id={summary['run_id']} "
            f"accuracy={summary['metrics']['accuracy']:.4f} "
            f"f1={summary['metrics']['f1']:.4f} "
            f"signature={summary['signature'][:16]}…"
        )
        return 0

    if args.cmd == "verify-signature":
        pin = load_signature(args.pin)
        dataset = load_versioned_dataset(args.data, args.manifest)
        params = pin["meta"]["params"]
        actual = compute_signature(
            dataset.features,
            dataset.target,
            params,
            args.random_state,
            dataset_sha256=dataset.content_sha256,
            dataset_version=str(dataset.manifest["version"]),
        )
        status, code = check_signature(pin["signature"], actual)
        print(f"verdict: {status}")
        if code != 0:
            print(f"  expected={pin['signature'][:16]}… actual={actual[:16]}…")
        return code

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    model = load_model(args.model_uri)
    dataset = load_versioned_dataset()
    pred = model.predict(dataset.features[:1])
    print(json.dumps({"prediction": int(pred[0]), "model_uri": args.model_uri}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
