# repro-ml-pipeline

**Pin a run signature after training and fail CI when data, environment, code, parameters, or seed no longer match that contract.**

[![CI](https://github.com/homayoun-safarpour/repro-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/homayoun-safarpour/repro-ml-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository is a local-first train, register, serve, and verify path for a sklearn
classifier. Ordinary development and CI need no paid API, cloud account, or production
credential.

## The failure it catches

| Change | Evidence in the signature | Gate |
| --- | --- | --- |
| Dataset bytes or version | Manifest and content SHA-256 | exit `2` |
| Resolved environment | `uv.lock` SHA-256 | exit `2` |
| Training source | Stable source-tree revision | exit `2` |
| Hyperparameters | Canonical parameter object | exit `2` |
| Randomness | Explicit seed | exit `2` |

`tests/test_pipeline.py::test_full_signature_covers_data_environment_code_params_and_seed`
names the central claim. `test_data_quality_rejects_content_drift` and
`test_metric_regression_floor` cover the data and quality boundaries.

## Install in under 30 minutes

Claim boundaries: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/homayoun-safarpour/repro-ml-pipeline
cd repro-ml-pipeline
uv sync --frozen --extra dev
uv run repro-ml train --artifact-dir examples/artifacts
uv run repro-ml verify-signature --pin examples/artifacts/run_signature.json
```

The CSV and its manifest are committed under `data/`. Once dependencies are cached, the
same workflow remains available offline through the local SQLite MLflow backend.

## Real benchmark

`uv run python scripts/run_benchmark.py` produced the committed
[`examples/benchmark_reproducibility.md`](examples/benchmark_reproducibility.md):

```text
Repeated signatures and metrics equal: true
Holdout accuracy: 0.986014
Five-fold CV accuracy: 0.978878 +/- 0.008790
Deliberate C=1.0 -> 0.5 drift: SIGNATURE_MISMATCH (exit 2)
```

The report includes both run records and full hashes. It proves repeatability for the
committed Wisconsin Diagnostic data and locked pipeline, not generalization to another
population.

## Train, register, predict

Training logs metrics and the signature to MLflow, creates an immutable registry version,
and points the `champion` alias at that version:

```bash
uv run repro-ml train --artifact-dir examples/artifacts
uv run repro-ml predict \
  --tracking-uri sqlite:///examples/artifacts/mlflow.db \
  --model-uri models:/repro-ml-classifier@champion
```

`tests/test_cli.py::test_cli_train_and_verify` proves both registry-backed prediction and
signature exit semantics.

## Serve the registry model

The API exposes liveness, registry-backed readiness, run metadata, and a typed 30-feature
prediction request:

```bash
MLFLOW_TRACKING_URI=sqlite:///examples/artifacts/mlflow.db \
MODEL_URI=models:/repro-ml-classifier@champion \
uv run uvicorn repro_ml_pipeline.serve:app --port 8000
```

- `GET /health` checks the process.
- `GET /ready` loads the versioned model URI.
- `GET /metadata` returns registry version, run ID, data hash, code revision, and signature.
- `POST /predict` validates all 30 numeric features before inference.

`tests/test_serve.py::test_api_health_readiness_metadata_and_registry_prediction` backs
these claims.

## Docker Compose

```bash
docker compose up --build --wait
curl http://localhost:8000/metadata
docker compose down -v
```

Compose starts the MLflow HTTP tracking server, runs a one-shot trainer that registers
`champion`, then starts the API. The `docker-smoke` CI job builds the images and calls
readiness, metadata, and prediction endpoints.

## CI and release boundary

- CI installs `uv.lock`, runs ruff and pytest on Python 3.10, 3.11, and 3.12, then performs
  an explicit train and signature verification on each version.
- A separate Docker job executes the tracking, registry, trainer, and API path.
- Pushing a `v*` tag reruns quality gates, builds wheel/sdist files, publishes a versioned
  image to GitHub Container Registry, and creates a GitHub Release.
- No workflow deploys to an external runtime. Cloud hosting still requires an approved
  account, credentials, persistence, networking, monitoring, and rollback policy.

## How we did it

1. Kept MLflow's documented sklearn logging and registry interfaces.
2. Replaced an implicit sklearn data loader with a versioned CSV, source record, and hash.
3. Expanded the gate from data and parameters to the complete run identity.
4. Split process health from model readiness and made inference resolve a registry URI.
5. Kept local SQLite and committed data so the core path does not depend on hosted systems.

See [`docs/INTERVIEW.md`](docs/INTERVIEW.md) for the two-minute walkthrough and trade-offs.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md). Changes to data, signatures, training, or
dependencies must rerun the benchmark. Small extension ideas are listed there and in the
issue template.

## Field alignment

Deterministic run-identity gates for classical ML paths. Claim boundaries: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

## License

MIT. Author: Homayoun Safarpour · [LinkedIn](https://www.linkedin.com/in/homayoun-safarpour/)
