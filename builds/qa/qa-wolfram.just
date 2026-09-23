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
	mbx test -p autoeq-qa --release --test qa_contract --test wolfram_peq_response --test wolfram_fir_response --test wolfram_log_interp --test wolfram_erb_quadrature

# Full autoeq-qa suite (contract + all comparisons + lib unit tests).
[group('qa-wolfram')]
qa-wolfram:
	mbx test -p autoeq-qa --release

# Contract gate only: manifest <-> oracle/golden/test completeness.
[group('qa-wolfram')]
qa-wolfram-contract:
	mbx test -p autoeq-qa --release --test qa_contract

# Python-runner comparisons (PY/H families): each tests/*.py loads its
# golden, asserts, and prints QA_RESULT; any failure exits nonzero.
[group('qa-wolfram')]
qa-wolfram-python:
	#!/usr/bin/env bash
	set -euo pipefail
	pass=0; fail=0
	for t in crates/autoeq-qa/tests/*.py; do
		[ -e "$t" ] || continue
		echo "--- $t"
		if python3 "$t"; then pass=$((pass+1)); else fail=$((fail+1)); fi
	done
	echo "python comparisons: $pass passed, $fail failed"
	[ "$fail" -eq 0 ]

# Regenerate engine-blessed goldens (needs an activated Wolfram Engine:
# WOLFRAMSCRIPT=wolframscript just qa-wolfram-goldens).
[group('qa-wolfram')]
qa-wolfram-goldens:
	#!/usr/bin/env bash
	set -euo pipefail
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
	mbx test -p autoeq-qa --release --test qa_contract
