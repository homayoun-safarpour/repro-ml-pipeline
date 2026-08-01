"""Run the committed reproducibility and deliberate-drift benchmark."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from repro_ml_pipeline.data import load_versioned_dataset
from repro_ml_pipeline.signature import check_signature, compute_signature
from repro_ml_pipeline.train import DEFAULT_PARAMS, train_and_log

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DRIFT_EXIT_CODE = 2


def main() -> None:
    results: list[dict[str, object]] = []
    work = Path(tempfile.mkdtemp(prefix="repro-ml-benchmark-"))
    try:
        for index in (1, 2):
            started = time.perf_counter()
            run_dir = work / f"run-{index}"
            summary = train_and_log(
                tracking_uri=f"sqlite:///{(run_dir / 'mlflow.db').as_posix()}",
                experiment="reproducibility-benchmark",
                artifact_dir=run_dir / "artifacts",
                random_state=42,
                model_name=f"repro-benchmark-{index}",
            )
            results.append(
                {
                    "run": index,
                    "signature": summary["signature"],
                    "metrics": summary["metrics"],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    dataset = load_versioned_dataset()
    baseline = str(results[0]["signature"])
    drifted = compute_signature(
        dataset.features,
        dataset.target,
        {**DEFAULT_PARAMS, "C": 0.5},
        42,
        dataset_sha256=dataset.content_sha256,
        dataset_version=str(dataset.manifest["version"]),
    )
    drift_verdict, drift_exit = check_signature(baseline, drifted)
    equivalent = (
        results[0]["signature"] == results[1]["signature"]
        and results[0]["metrics"] == results[1]["metrics"]
    )
    docker_evidence = EXAMPLES / "docker_smoke.txt"
    report = {
        "benchmark": "reproducibility-v1",
        "dataset_sha256": dataset.content_sha256,
        "seed": 42,
        "repeated_runs_equivalent": equivalent,
        "runs": results,
        "deliberate_param_drift": {
            "changed": {"C": [1.0, 0.5]},
            "baseline_signature": baseline,
            "drifted_signature": drifted,
            "verdict": drift_verdict,
            "exit_code": drift_exit,
        },
        "container_smoke": (
            docker_evidence.read_text(encoding="utf-8").strip()
            if docker_evidence.exists()
            else "Run `docker compose up --build --wait`; CI captures logs as an artifact."
        ),
    }
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES / "benchmark_reproducibility.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = results[0]["metrics"]
    markdown = f"""# Reproducibility benchmark

Command: `uv run python scripts/run_benchmark.py`

- Dataset SHA-256: `{dataset.content_sha256}`
- Seed: `42`
- Repeated signatures and metrics equal: `{str(equivalent).lower()}`
- Holdout accuracy: `{metrics['accuracy']:.6f}`
- Holdout F1: `{metrics['f1']:.6f}`
- Five-fold CV accuracy: `{metrics['cv_accuracy_mean']:.6f} ± {metrics['cv_accuracy_std']:.6f}`
- Five-fold CV F1: `{metrics['cv_f1_mean']:.6f} ± {metrics['cv_f1_std']:.6f}`
- Deliberate `C=1.0 -> 0.5` drift: `{drift_verdict}` (exit `{drift_exit}`)

## Container smoke

```
{report['container_smoke']}
```

## Scope

This benchmark proves deterministic equivalence on the committed dataset, lockfile,
source revision, parameters, and seed. It does not prove performance on another dataset
or availability in an externally hosted environment.
"""
    (EXAMPLES / "benchmark_reproducibility.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not equivalent or drift_exit != DRIFT_EXIT_CODE:
        raise SystemExit(DRIFT_EXIT_CODE)


if __name__ == "__main__":
    main()
