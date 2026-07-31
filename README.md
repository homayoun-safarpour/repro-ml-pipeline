# repro-ml-pipeline

**Notebooks train a model that nobody can re-run with the same bits. This pipeline logs sklearn fits to MLflow, pins a data+params signature hash, and fails CI when the signature drifts.**

[![CI](https://github.com/homayoun-safarpour/repro-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/homayoun-safarpour/repro-ml-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## The problem

A "production ML pipeline" claim usually means a Docker image and a hope. Without a content hash over the training matrix and hyperparameters, a silent data pull or param tweak still produces a green accuracy number. Interviewers ask whether you can prove the run is the same run.

## Threat model (when this fails in production)

| Failure | What it looks like | What this repo does |
| --- | --- | --- |
| Silent data change | Accuracy similar; different rows | `SIGNATURE_MISMATCH` on `verify-signature` |
| Param drift | Someone changes `C` in CI YAML only | Signature includes params JSON |
| Metrics without lineage | MLflow metrics, no pin file | Signature artifact + tag `run_signature` |
| Irreproducible seed | "Works on my laptop" | Fixed `random_state` in signature |
| Image-only ship | Docker without a gate | CI runs train + verify on every push |

## Install

```bash
git clone https://github.com/homayoun-safarpour/repro-ml-pipeline
cd repro-ml-pipeline
pip install -e ".[dev]"
```

Python 3.10+. Requires scikit-learn and MLflow.

## Quickstart

```bash
repro-ml train --artifact-dir examples/artifacts
repro-ml verify-signature --pin examples/artifacts/run_signature.json
```

Real output from this repository (committed under `examples/`):

```
$ cat examples/train_summary.json
(see file: run_id, accuracy, f1, signature)

$ repro-ml verify-signature --pin examples/artifacts/run_signature.json
verdict: PASS
```

## How we did it

1. **Chose upstream patterns.** MLflow's sklearn examples ([mlflow/mlflow](https://github.com/mlflow/mlflow), Apache-2.0) show tracking + model logging. Full MLflow monorepo forks are not a 30-minute portfolio instrument.
2. **Restyled into one instrument.** MIT package `repro-ml-pipeline`: breast-cancer demo, LogisticRegression pipeline, local file store, Docker, GitHub Actions.
3. **Sharp improvement.** Data+params `run_signature` SHA-256 verified in CI (`verify-signature` exit `0`/`2`). Named tests: `test_signature_changes_when_params_change`, `test_cli_train_and_verify`.
4. **Reproduce committed artifacts:**

```bash
repro-ml train --artifact-dir examples/artifacts
cp examples/artifacts/train_summary.json examples/train_summary.json
```

## Compose with the rest of the stack

| Repo | Role next to this |
| --- | --- |
| [judge-reliability-kit](https://github.com/homayoun-safarpour/judge-reliability-kit) | Evaluation math when humans/LLMs label model outputs |
| [judge-drift-sentinel](https://github.com/homayoun-safarpour/judge-drift-sentinel) | Drift gate for judges; this repo is the training signature gate |
| [rag-eval-service](https://github.com/homayoun-safarpour/rag-eval-service) | Retrieval eval service when the product is RAG, not tabular ML |
| [agent-loop-engine](https://github.com/homayoun-safarpour/agent-loop-engine) | Can treat `verify-signature` exit `2` like any other quality gate |

## Docker

```bash
docker build -t repro-ml-pipeline .
docker run --rm repro-ml-pipeline train --artifact-dir /tmp/out \
  --tracking-uri sqlite:////tmp/out/mlflow.db
```

## Topics

`mlops` · `mlflow` · `scikit-learn` · `reproducibility` · `docker` · `ci-cd` · `python`

## License

MIT. Author: Homayoun Safarpour · [LinkedIn](https://www.linkedin.com/in/homayoun-safarpour/)
