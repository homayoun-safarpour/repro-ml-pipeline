"""CLI: train, verify-signature, and registry-backed predict."""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlflow

from repro_ml_pipeline.data import DEFAULT_DATA_PATH, DEFAULT_MANIFEST_PATH, load_versioned_dataset
from repro_ml_pipeline.registry import load_model
from repro_ml_pipeline.signature import (
    PinError,
    changed_fields,
    check_signature,
    compute_signature,
    load_pin,
    signature_payload,
)
from repro_ml_pipeline.train import train_and_log

EXIT_NO_VERDICT = 1
EXIT_MISMATCH = 2


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1, so exit 2 always means SIGNATURE_MISMATCH."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_NO_VERDICT, f"{self.prog}: error: {message}\n")


def _no_verdict(verdict: str, reason: object) -> int:
    print(f"verdict: {verdict}")
    print(f"  {reason}")
    return EXIT_NO_VERDICT


def verify_signature(
    pin_path: str,
    data_path: str,
    manifest_path: str,
    random_state: int | None = None,
) -> int:
    """Print one verdict and return 0 (PASS), 2 (SIGNATURE_MISMATCH) or 1 (no verdict)."""
    try:
        pin = load_pin(pin_path)
    except PinError as exc:
        return _no_verdict("BAD_PIN", exc)
    expected, meta = pin["signature"], pin["meta"]
    seed = meta["seed"] if random_state is None else random_state

    try:
        dataset = load_versioned_dataset(data_path, manifest_path)
        dataset_version = str(dataset.manifest["version"])
    except OSError as exc:
        return _no_verdict("BAD_INPUT", f"cannot read data or manifest: {exc}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return _no_verdict("BAD_INPUT", f"malformed manifest: {exc!r}")
    except ValueError as exc:
        print("verdict: SIGNATURE_MISMATCH")
        print(f"  data: {exc}")
        return EXIT_MISMATCH

    kwargs = {
        "dataset_sha256": dataset.content_sha256,
        "dataset_version": dataset_version,
    }
    params = meta["params"]
    actual = compute_signature(dataset.features, dataset.target, params, seed, **kwargs)
    payload = signature_payload(dataset.features, dataset.target, params, seed, **kwargs)
    changed = changed_fields(meta, payload)
    status, code = check_signature(expected, actual)
    if code == 0 and changed:
        status, code = "SIGNATURE_MISMATCH", EXIT_MISMATCH
    print(f"verdict: {status}")
    if code != 0:
        print(f"  expected={expected[:16]}... actual={actual[:16]}...")
        if changed:
            print(f"  changed: {', '.join(changed)}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(
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
    vf.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="seed to check (default: the seed stored in the pin)",
    )
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
            f"signature={summary['signature'][:16]}..."
        )
        return 0

    if args.cmd == "verify-signature":
        return verify_signature(args.pin, args.data, args.manifest, args.random_state)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    model = load_model(args.model_uri)
    dataset = load_versioned_dataset()
    pred = model.predict(dataset.features[:1])
    print(json.dumps({"prediction": int(pred[0]), "model_uri": args.model_uri}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
