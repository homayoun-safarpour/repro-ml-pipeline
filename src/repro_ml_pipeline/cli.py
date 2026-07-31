"""CLI: train / verify-signature / predict."""

from __future__ import annotations

import argparse
import json
import sys

import joblib

from repro_ml_pipeline.signature import check_signature, compute_signature, load_signature
from repro_ml_pipeline.train import DEFAULT_PARAMS, load_demo_xy, train_and_log


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

    vf = sub.add_parser("verify-signature", help="fail if signature does not match pin")
    vf.add_argument("--pin", required=True, help="path to pinned signature JSON")
    vf.add_argument("--random-state", type=int, default=42)

    pr = sub.add_parser("predict", help="score first row of demo holdout features")
    pr.add_argument("--model", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        summary = train_and_log(
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
            artifact_dir=args.artifact_dir,
            random_state=args.random_state,
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
        X, y = load_demo_xy(random_state=args.random_state)
        actual = compute_signature(X, y, DEFAULT_PARAMS, args.random_state)
        status, code = check_signature(pin["signature"], actual)
        print(f"verdict: {status}")
        if code != 0:
            print(f"  expected={pin['signature'][:16]}… actual={actual[:16]}…")
        return code

    model = joblib.load(args.model)
    X, _ = load_demo_xy(random_state=42)
    pred = model.predict(X[:1])
    print(json.dumps({"prediction": int(pred[0])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
