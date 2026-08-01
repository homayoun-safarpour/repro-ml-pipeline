# Interview notes

## Two-minute walkthrough

1. Open `data/breast_cancer_v1.manifest.json` and show the external source, version, and
   content SHA-256.
2. Open a generated `run_signature.json`. Point to its data, environment, code,
   parameters, and seed fields.
3. Run `docker compose up --build --wait`.
4. Call `GET /metadata` to show the registry model version, run ID, and signature.
5. Open `examples/benchmark_reproducibility.md` and compare the equal repeated runs with
   the deliberate `SIGNATURE_MISMATCH`.

## Three questions

### Why is a random seed not enough?

A seed controls supported pseudo-random operations. It says nothing about changed rows,
library versions, source logic, or hyperparameters. This pipeline hashes those inputs
into one gate while retaining each component in readable metadata.

### Why use an MLflow alias URI?

`models:/repro-ml-classifier@champion` separates serving configuration from a filesystem
path. Registration records an immutable version; the alias states which version the API
should load. `/metadata` resolves the alias back to its version and run.

### What does the benchmark establish?

Two independent runs with the same committed inputs produce equal signatures and
metrics. A changed parameter produces exit code 2. The result is scoped to this dataset,
CPU sklearn path, and locked environment specification. It is not evidence of external
availability or generalization to a new population.

## Design trade-offs

- The CSV is committed because a fresh clone must remain reproducible without network
  access. The manifest retains the upstream source and license.
- CI uses SQLite-backed MLflow for deterministic isolation. Compose exposes the same
  registry behavior through an HTTP tracking server.
- The API loads lazily. `/health` proves the process is alive; `/ready` proves the
  versioned model can be resolved and loaded.
- GitHub Releases and GHCR are artifact publication paths. Deployment to a cloud runtime
  needs an external account, credentials, networking, persistence, and monitoring.

## Failure to discuss

The run signature detects input changes, not whether a changed input is harmful. Real
production monitoring still needs feature and prediction drift thresholds tied to a
domain review process.
# Interview gate — repro-ml-pipeline

## Three questions

1. **What goes into the run signature?**  
   Sample/feature counts, SHA-256 of the training `X` and `y` bytes, hyperparameter dict, and `random_state`. Change any of those and the digest moves.

2. **Why not rely on MLflow run IDs alone?**  
   A run ID names a logged experiment. It does not prove the next CI job trained on the same matrix. The pin file is the contract; MLflow is the ledger.

3. **How does this relate to judge-drift-sentinel?**  
   Sentinel answers "did the judge move?". This answers "did the training inputs/params move?". Both fail closed with exit `2`.

## Two-minute demo

```bash
git clone https://github.com/homayoun-safarpour/repro-ml-pipeline
cd repro-ml-pipeline
pip install -e ".[dev]"
repro-ml verify-signature --pin examples/artifacts/run_signature.json
pytest -q
```

Expect: `verdict: PASS`, tests green including `test_signature_changes_when_params_change`.

## One limitation

The demo uses scikit-learn's breast-cancer dataset loaded in-process. A production warehouse pull needs an adapter that feeds the same `compute_signature` API; the gate contract stays identical.
