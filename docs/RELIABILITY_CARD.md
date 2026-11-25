# Reliability card: repro-ml-pipeline

| Field | Value |
| --- | --- |
| **Job** | Pin a train/serve run signature and fail CI when identity drifts |
| **Primary signals** | Data SHA, `uv.lock` SHA, source revision, params, seed |
| **Exit codes** | `0` on `PASS`; `2` on `SIGNATURE_MISMATCH` (any change to data, lock file, source, parameters or seed, including a data file that fails its manifest); `1` with no verdict on `BAD_PIN`, `BAD_INPUT` or a usage error. The metric floor is a pytest test, not a CLI exit code. |
| **Central test** | `test_full_signature_covers_data_environment_code_params_and_seed` |
| **Claim** | Reproducibility is a contract you can gate, not a README promise |
| **Not claimed** | Cloud MLOps platform; paid APIs |

## Field alignment

Matches "when not to use an LLM" hire language: deterministic identity gates for classical ML paths that agents still depend on.
