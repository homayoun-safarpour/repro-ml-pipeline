# LinkedIn draft (public-safe) - repro-ml first-screen restyle 2026-09-30

Field pain first. Homayoun posts himself.
Paste verified 2026-09-30 on examples/bad_pin.json: SIGNATURE_MISMATCH, exit 2.

---

Paste block (copy from the next line to the URL):

A matching accuracy number is not a matching run.

repro-ml-pipeline pins data, environment, code, parameters, and seed, then fails CI when any of them move.

The stranger run is a zeroed pin:

git clone https://github.com/homayoun-safarpour/repro-ml-pipeline
cd repro-ml-pipeline
uv sync --frozen
uv run repro-ml verify-signature --pin examples/bad_pin.json

verdict: SIGNATURE_MISMATCH
  expected=0000000000000000... actual=05091462d75dbc95...

Exit 2. The recorded train pin is examples/artifacts/run_signature.json.

The limit: one locked sklearn demo, not a general MLOps platform.

Repo:
https://github.com/homayoun-safarpour/repro-ml-pipeline
