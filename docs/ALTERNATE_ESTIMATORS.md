# Alternate estimators: config example

The train path stays estimator-agnostic on purpose. The signature gate covers
data, environment, code, parameters, and seed; it does not care which
scikit-learn estimator produces the model. Swapping the estimator is a code
change, so the code hash in the signature changes with it.

## Current estimator

`src/repro_ml_pipeline/train.py` builds a `LogisticRegression` inside
`build_pipeline()`:

```python
DEFAULT_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 500,
    "solver": "lbfgs",
}

def build_pipeline(params: dict[str, Any] | None = None, random_state: int = 42) -> Pipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    clf = LogisticRegression(
        C=float(p["C"]),
        max_iter=int(p["max_iter"]),
        solver=str(p["solver"]),
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
```

## Example: RandomForestClassifier

An alternate estimator follows the same shape: a parameter dictionary with a
fixed seed, an estimator built from it, and the same scaler wrapper.

```python
# Config example for an alternate estimator, for contributor reference.
# Not wired into the CLI; it documents the estimator-agnostic contract.
ALT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 4,
}

def build_alt_pipeline(params: dict[str, Any] | None = None, random_state: int = 42) -> Pipeline:
    p = {**ALT_PARAMS, **(params or {})}
    clf = RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
```

Why this is a code change and not a data flag:

- The estimator is constructed in source, so the signature's code hash
  reflects the swap automatically.
- The parameter object is canonical (`params` merged over defaults), so the
  parameters hash changes exactly when the estimator's parameters change.
- The seed is explicit, so the randomness hash stays meaningful.
- MLflow logs whatever estimator is in the pipeline; the registry path does
  not special-case the model class.

What a contributor must not do:

- Do not load an estimator from a string in the config file. A string does
  not identify a versioned code path; the signature would claim coverage it
  does not have.
- Do not loosen the lockfile to try estimator variants ad hoc. Add the
  dependency normally (it is already present for this example: scikit-learn
  is a base dependency) and regenerate `uv.lock` deliberately.

## Verification on the pinned demo data

The existing estimator path still passes on the committed Wisconsin
Diagnostic data after this documentation change:

```text
$ uv run repro-ml train --artifact-dir examples/artifacts
run_id=8ef716b6507847ef969171ad39ce0d20 accuracy=0.9860 f1=0.9889 signature=bc9e9ed0…

$ uv run repro-ml verify-signature --pin examples/artifacts/run_signature.json
verdict: PASS

$ uv run pytest -q
10 passed

$ uv run ruff check src tests scripts
All checks passed!
```

The demo data, manifest, and `uv.lock` are unchanged. No dependency was
added or removed.
