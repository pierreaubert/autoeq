# autoeq-qa

Independent numerical cross-validation for AutoEQ and RoomEQ. The checked-in
[`validation-manifest.toml`](validation-manifest.toml) owns 124 cases: 118 Rust
integration-test cases and six Python runner cases. Every case records its
oracle, family, owner, tolerance, and expected test entrypoint.

Rust cases compare implementation results with closed-form Wolfram oracles in
`wolfram/*.wls`. They use a live Wolfram Engine when `WOLFRAMSCRIPT` is set and
otherwise compare against checked-in files under `wolfram/goldens/`. Python
cases use the same manifest and emit one machine-readable `QA_RESULT` line so
the runner can reject missing, duplicate, or incorrectly named results.

Run the full Rust and Python case set with:

```bash
just qa-wolfram-validation
```

Install the pinned Python numerical requirements from
[`scripts/autoeq-qa-requirements.txt`](../../scripts/autoeq-qa-requirements.txt)
before running the Python cases. Manifest validation also verifies that every
Rust case maps to a Cargo test target and that every oracle/golden is present.
The `negative_controls_tailpy.py` helper is a separate supplemental check, not
one of the 124 manifest cases.
