# Contributing

## Local setup

```bash
git clone https://github.com/homayoun-safarpour/repro-ml-pipeline
cd repro-ml-pipeline
uv sync --frozen --extra dev
uv run ruff check src tests scripts
uv run pytest -q
```

The committed dataset and lockfile keep this path offline after the packages are cached.
Do not replace `data/breast_cancer_v1.csv` without regenerating its manifest and benchmark.

## Change contract

1. Add a named test for each behavior claim.
2. Run explicit train and signature verification:

   ```bash
   uv run repro-ml train --artifact-dir .local-artifacts
   uv run repro-ml verify-signature --pin .local-artifacts/run_signature.json
   ```

3. Run `uv run python scripts/run_benchmark.py` when changing data, training, signatures,
   or dependency versions.
4. Do not commit credentials, local MLflow databases, or generated model binaries.

## Good first extensions

- Add a second versioned CSV fixture and preserve the same manifest contract.
- Add a batch prediction request with an explicit maximum row count.
- Add a model-card artifact populated from MLflow run metadata.

Open an issue before changing signature schema semantics or registry alias behavior.
