# --------------------------------------------------------- -*- just -*-
# QA WOLFRAM (autoeq-qa: Wolfram Engine cross-validation)
# ----------------------------------------------------------------------
# Structure mirrors math-qa in the neighboring math-audio checkout:
# one .wls oracle -> versioned golden JSON -> comparison test, listed
# in crates/autoeq-qa/validation-manifest.toml with a QA_RESULT record.
#
# The engine needs its kernel path, e.g.
# export WolframKernel="/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel"

# Smoke tier: fast closed-form comparisons (uses goldens unless
# WOLFRAMSCRIPT points at a live engine).
[group('qa-wolfram')]
qa-wolfram-smoke:
	{{cargo}} test -p autoeq-qa --locked --release --test qa_contract --test wolfram_peq_response --test wolfram_fir_response --test wolfram_log_interp --test wolfram_erb_quadrature

# Full AutoEQ numerical QA. The runner checks manifest/target membership,
# executes all 118 Rust cases plus the library tests, and records per-case evidence.
[group('qa-wolfram')]
qa-wolfram:
	python3 scripts/run_crate_test_matrix.py --qa-runner rust --release

# Contract gate only: manifest <-> oracle/golden/test completeness.
[group('qa-wolfram')]
qa-wolfram-contract:
	python3 scripts/run_crate_test_matrix.py --check-qa-manifest
	{{cargo}} test -p autoeq-qa --locked --release --test qa_contract

# The manifest runner executes exactly its six Python cases and verifies
# identity, result count, tolerance kind, and expected/executed case totals.
[group('qa-wolfram')]
qa-wolfram-python:
	python3 scripts/run_crate_test_matrix.py --qa-runner python
	# This helper is a negative control, not a seventh manifest case.
	python3 crates/autoeq-qa/tests/negative_controls_tailpy.py

[group('qa-wolfram')]
qa-wolfram-validation:
	just qa-wolfram
	just qa-wolfram-python

# Regenerate engine-blessed goldens (needs an activated Wolfram Engine:
# WOLFRAMSCRIPT=wolframscript just qa-wolfram-goldens).
[group('qa-wolfram')]
qa-wolfram-goldens:
	#!/usr/bin/env bash
	set -euo pipefail
	python3 scripts/run_crate_test_matrix.py --check-qa-manifest --allow-missing-goldens
	mkdir -p crates/autoeq-qa/wolfram/goldens
	engine="${WOLFRAMSCRIPT:-wolframscript}"
	kernel_default="/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel"
	kernel_alt="/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel"
	if [ -z "${WolframKernel:-}" ] && [ -f "$kernel_default" ]; then export WolframKernel="$kernel_default"; fi
	if [ -z "${WolframKernel:-}" ] && [ -f "$kernel_alt" ]; then export WolframKernel="$kernel_alt"; fi
	for script in crates/autoeq-qa/wolfram/*.wls; do
		base="$(basename "$script" .wls)"
		echo "--- $base"
		"$engine" -file "$script" | tail -n 1 > "crates/autoeq-qa/wolfram/goldens/$base.json"
	done
	{{cargo}} test -p autoeq-qa --locked --release --test qa_contract
