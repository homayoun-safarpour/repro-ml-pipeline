# Reliability card — repro-ml-pipeline

| Field | Value |
| --- | --- |
| **Job** | Pin a train/serve run signature and fail CI when identity drifts |
| **Primary signals** | Data SHA, `uv.lock` SHA, source revision, params, seed |
| **Exit codes** | `2` on signature / content / metric-floor breach |
| **Central test** | `test_full_signature_covers_data_environment_code_params_and_seed` |
| **Claim** | Reproducibility is a contract you can gate, not a README promise |
| **Not claimed** | Cloud MLOps platform; paid APIs |

## Field alignment

Matches "when not to use an LLM" hire language: deterministic identity gates for classical ML paths that agents still depend on.
