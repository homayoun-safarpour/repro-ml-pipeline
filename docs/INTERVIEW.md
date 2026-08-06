# Interview talking points : repro-ml-pipeline

Five CLI-backed points for a technical screen (no resume recap).

- **`repro-ml train --tracking-uri sqlite:///examples/artifacts/mlflow.db`** : trains the demo classifier, logs metrics to MLflow, registers `models:/repro-ml-classifier@champion`, and writes a pin-able run signature JSON.
- **`repro-ml verify-signature --pin examples/artifacts/run_signature.json`** : recomputes the digest from committed data, manifest, params, and seed; exit `0` on PASS, exit `2` on mismatch.
- **`repro-ml predict --model-uri models:/repro-ml-classifier@champion`** : scores through the registry alias so serving proves the resolved version, not a stale local pickle path.
- **`docker compose up --build --wait` and `GET /metadata`** : the API exposes run ID, model version, and signature fields aligned with the CLI pin file.
- **`pytest -q -k signature`** : includes `test_signature_changes_when_params_change`, which proves a hyperparameter edit fails the gate without silent metric drift.

## Three questions

1. **What goes into the run signature?**  
   Sample/feature counts, SHA-256 of the training `X` and `y` bytes, hyperparameter dict, and `random_state`. Change any of those and the digest moves.

2. **Why not rely on MLflow run IDs alone?**  
   A run ID names a logged experiment. It does not prove the next CI job trained on the same matrix. The pin file is the contract; MLflow is the ledger.

3. **How does this relate to judge-drift-sentinel?**  
   Sentinel answers "did the judge move?". This answers "did the training inputs/params move?". Both fail closed with exit `2`.

## One limitation

The demo uses scikit-learn's breast-cancer dataset loaded in-process. A production warehouse pull needs an adapter that feeds the same `compute_signature` API; the gate contract stays identical.
