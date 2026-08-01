# Production ML Pipeline week

## BENCHMARK GATE

- [x] CI matrix declared for Python 3.10, 3.11, and 3.12
- [x] Named claim tests cover signature, data drift, metric floor, registry prediction, and API
- [x] Versioned public dataset and SHA-256 manifest
- [x] Full signature covers data, environment lock, source revision, parameters, and seed
- [x] MLflow tracking, registry alias, and registry-backed inference
- [x] FastAPI health, readiness, metadata, and typed prediction
- [x] Docker Compose tracking, train/register, and serve path
- [x] Tag-gated GitHub release and GHCR artifact workflow
- [x] Repeated-run and deliberate-drift benchmark committed under `examples/`
- [ ] GitHub Actions green on the pushed commit
- [ ] Public repository and sole-contributor checks confirmed after push

No field benchmark is claimed. The committed benchmark tests this repository's
reproducibility contract against the versioned Wisconsin Diagnostic dataset.

## Threat model

The gate detects changed dataset bytes, dependency lock, source files, parameters, or
seed. It does not detect upstream infrastructure failure, malicious registry access, or
real-world concept drift after deployment.

## Daily ticks

- [x] W1 version and validate external data
- [x] W2 expand the run signature and lock dependencies
- [x] W3 add cross-validation, metric floors, and registry promotion
- [x] W4 add registry-backed CLI and FastAPI serving
- [x] W5 add Compose and Docker smoke CI
- [x] W6 add release artifacts and benchmark evidence
- [x] W7 close documentation, guard, public visibility, and CI

## Interview gate

1. Why hash the lockfile and source tree instead of only model parameters?
2. Why does serving load `models:/name@alias` instead of a local pickle?
3. What does repeated-run equality prove, and what does it leave unproven?

Two-minute path: run `docker compose up --build --wait`, inspect `/metadata`, submit one
typed prediction, then show the deliberate signature failure in the benchmark report.

Honest limitation: the workflow publishes release artifacts to GitHub, but no external
cloud service is configured or claimed.

## NEXT TICK

Confirm the pushed CI run is green and record its URL.
