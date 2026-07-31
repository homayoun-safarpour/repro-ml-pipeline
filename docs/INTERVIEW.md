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
