# Reproducibility benchmark

Command: `uv run python scripts/run_benchmark.py`

- Dataset SHA-256: `cd1c796fd1753b7ca60a00b1bd4a5adc090c87f69be2ca2e742cedf77bb260a2`
- Seed: `42`
- Repeated signatures and metrics equal: `true`
- Holdout accuracy: `0.986014`
- Holdout F1: `0.988889`
- Five-fold CV accuracy: `0.978878 ± 0.008790`
- Five-fold CV F1: `0.983314 ± 0.006960`
- Deliberate `C=1.0 -> 0.5` drift: `SIGNATURE_MISMATCH` (exit `2`)

## Container smoke

```
GitHub Actions run: https://github.com/homayoun-safarpour/repro-ml-pipeline/actions/runs/30936022580
docker-smoke: PASS (1m10s)
tracking GET /health: 200
trainer registered repro-ml-classifier version 1 and alias champion
api GET /ready: 200
api GET /metadata: 200
api POST /predict: 200
model_uri=models:/repro-ml-classifier@champion
run_signature=bc9e9ed0dfe7a7e3ac4b2c95b759afedf1d0d623bd3718e0e5e0249eee89de36
```

## Scope

This benchmark proves deterministic equivalence on the committed dataset, lockfile,
source revision, parameters, and seed. It does not prove performance on another dataset
or availability in an externally hosted environment.
