# ----------------------------------------------------------------------
# QA ROOMEQ
# ----------------------------------------------------------------------

alias qa-roomeq := qa-roomeq-all

# Registry-driven tiers. The scenario inventory, claims, and expectations live
# in qa/registry/roomeq.json; recipes only select cost tier.
[group('qa-roomeq')]
qa-roomeq-pr:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --tier pr
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --pr
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier pr

[group('qa-roomeq')]
qa-roomeq-nightly:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --tier nightly
	cargo run --features qa --bin roomeq-qa-quality --release
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --full-matrix
	cargo run --features qa --bin roomeq-qa-features --no-default-features --release
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier nightly
	cargo run --bin roomeq-fuzzer --release --features="qa,plotly" -- -n 200 --seed 42 --skip-kautz-modal

[group('qa-roomeq')]
qa-roomeq-weekly:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --tier weekly
	cargo run --features qa --bin roomeq-qa-quality --release
	cargo test --release -p autoeq --features qa --test roomeq_generated_data_test -- --ignored --test-threads=1
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --full-matrix
	cargo run --features qa --bin roomeq-qa-features --no-default-features --release
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier nightly
	cargo run --bin roomeq-fuzzer --release --features="qa,plotly" -- -n 1000 --seed 42 --skip-kautz-modal

[group('qa-roomeq')]
qa-roomeq-all: qa-roomeq-weekly

# Memory-capped convergence run. The `--jobs` default in
# `roomeq-qa-quality` is `num_cpus/2` so each outer test case still gets
# parallel CMA-ES evaluators without OOM'ing the machine when 70+ cases are
# scheduled. Override with `just qa-roomeq-convergence jobs=N` or run the
# binary directly.
[group('qa-roomeq')]
qa-roomeq-convergence jobs="":
	#!/usr/bin/env bash
	set -euo pipefail
	if [ -n "{{jobs}}" ]; then
	  cargo run --features qa --bin roomeq-qa-quality --release -- --jobs {{jobs}}
	else
	  cargo run --features qa --bin roomeq-qa-quality --release
	fi

[group('qa-roomeq')]
qa-roomeq-small-stereo-20: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	      mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_0; \
	      cargo run --features cli --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_0/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_0/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_$method.json; \
		  ./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_$method.json \
		                   --base-config ./data_tests/roomeq/generate/$algo/small_stereo_2_0/config.json; \
	  done \
	done


[group('qa-roomeq')]
qa-roomeq-small-stereo-21: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_1; \
	    cargo run --features cli --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_1/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_1/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_$method.json \
		                   --base-config ./data_tests/roomeq/generate/$algo/small_stereo_2_1/config.json; \
	  done \
	done


[group('qa-roomeq')]
qa-roomeq-small-stereo-22: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_2_group; \
	    cargo run --features cli --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_2_group/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_2_group/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_2_group/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_2_group/dsp_$method.json \
		                   --base-config ./data_tests/roomeq/generate/$algo/small_stereo_2_2_group/config.json; \
	  done \
	done

[group('qa-roomeq')]
qa-roomeq-small-stereo-51: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/medium_surround_5_1; \
	    cargo run --features cli --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/medium_surround_5_1/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/medium_surround_5_1/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/medium_surround_5_1/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/medium_surround_5_1/dsp_$method.json \
		                   --base-config ./data_tests/roomeq/generate/$algo/medium_surround_5_1/config.json; \
	  done \
	done

# ----------------------------------------------------------------------
# QA ROOMEQ — Multi-Measurement
# Tests every generated FEM scenario across all multi-measurement strategies,
# then exercises focused home-cinema and height-channel feature configurations.
# Requires generated data (run `just generate-roomeq-data` first).
# ----------------------------------------------------------------------

[group('qa-roomeq-multi')]
qa-roomeq-multi-measurement: ensure-venv
	#!/usr/bin/env bash
	set -euo pipefail
	scenario_root="./data_tests/roomeq/generate/fem"
	override_root="./data_tests/roomeq/generate/optimiser-config/multi_measurement"
	output_root="./data_generated/roomeq/generated/fem"
	shopt -s nullglob
	configs=("$scenario_root"/*/config.json)
	if (( ${#configs[@]} == 0 )); then
	  echo "No FEM RoomEQ scenarios found under $scenario_root" >&2
	  exit 1
	fi
	for config in "${configs[@]}"; do
	  scenario="$(basename "$(dirname "$config")")"
	  scenario_output="$output_root/$scenario"
	  mkdir -p "$scenario_output"
	  for strategy in minimax weighted_sum variance_penalized; do
	    output="$scenario_output/dsp_iir_multi_$strategy.json"
	    echo "=== Multi-measurement $strategy (fem) $scenario ==="
	    cargo run --features cli --bin roomeq --release -- \
	      --config "$config" \
	      --override-config "$override_root/$strategy.json" \
	      --output "$output"
	    ./venv/bin/python3 ./scripts/display-roomeq.py "$output" --base-config "$config"
	  done
	done
	echo "=== Correctness-gated home-cinema feature matrix ==="
	cargo run --features qa --bin roomeq-qa-coverage --release -- --home-cinema

[group('qa-roomeq-multi')]
qa-roomeq-home-cinema-configs:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --home-cinema

# Exhaustive offline cinema QA. The Sonium corpus covers 5.1.2, 7.1.2,
# 7.4.4, 7.1.6, 9.1.6, and 9.8.6 (five seats, up to eight physical subs).
# This is intentionally a scheduled/local gate rather than a PR gate; allow up
# to 12 hours on a typical development workstation.
[group('qa-roomeq')]
qa-roomeq-cinema-full jobs="4" maxeval="5000":
	cargo run --features qa --bin roomeq-qa-coverage --release -- --home-cinema --jobs {{jobs}} --maxeval {{maxeval}}
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --full-matrix
	cargo run --features qa --bin roomeq-qa-features --no-default-features --release
	# Acoustic baseline regression is a separate cross-product gate:
	# `just qa-roomeq-acoustic-nightly`.

# New comprehensive QA using roomeq-qa-full binary
[group('qa-roomeq')]
qa-roomeq-coverage: prod-autoeq
	cargo run --features qa --bin roomeq-qa-coverage --release

# Hard line-coverage gate for the production RoomEQ libraries.
# CLI adapters, QA runners, synthetic fixtures, and the root compatibility
# facade are checked separately and are not part of the production denominator.
# Cross-crate integration profiles are accumulated because WP11 deliberately
# moved compatibility/workflow boundary tests out of the root library.
[group('qa-roomeq')]
qa-roomeq-coverage-gate:
	cargo llvm-cov --package roomeq-model --package roomeq-analysis --package roomeq-quality --package roomeq-engine --package roomeq-export --package roomeq-workflow --lib --no-report --release
	cargo llvm-cov --no-clean --package autoeq --features cli --test roomeq_integration_test --summary-only --release
	cargo llvm-cov --no-clean --package autoeq --features cli --test roomeq_pipeline_test --summary-only --release
	cargo llvm-cov --no-clean --package autoeq --features cli --test system_config_test --summary-only --release
	cargo llvm-cov --no-clean --package autoeq --features cli --test roomeq_generated_data_test --summary-only --release
	cargo llvm-cov report --package roomeq-model --package roomeq-analysis --package roomeq-quality --package roomeq-engine --package roomeq-export --package roomeq-workflow --summary-only --release --fail-under-lines 90

[group('qa-roomeq')]
qa-roomeq-quick: prod-autoeq
	cargo run --features qa --bin roomeq-qa-coverage --release -- --quick --maxeval 200

[group('qa-roomeq')]
qa-roomeq-list:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --list

[group('qa-roomeq')]
qa-roomeq-matrix:
	cargo run --features qa --bin roomeq-qa-coverage --release -- --matrix

[group('qa-roomeq')]
qa-roomeq-synthetic:
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release

[group('qa-roomeq')]
qa-roomeq-multiseat-guards:
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

[group('qa-roomeq')]
qa-roomeq-home-cinema:
	cargo test --release -p roomeq-model --lib -- --nocapture
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-all-channel-multiseat:
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-bass-management:
	cargo test --release -p roomeq-model --lib -- --nocapture
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-dsp-consistency:
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-phase-critical:
	cargo test --release -p roomeq-analysis --lib -- --nocapture
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

[group('qa-roomeq')]
qa-roomeq-perceptual:
	# nextest's explicit no-test failure prevents stale filters from producing a false-green QA run.
	cargo nextest run --release -p autoeq-optim --lib --no-tests fail -E 'test(/loss::epa::/)'
	cargo nextest run --release -p roomeq-workflow --lib --no-tests fail -E 'test(/generate_validation_bundle_report_creates_json|correction_report_(rejected_guardrails|failed_constraints_and_null_advisory)/)'
	cargo nextest run --release -p roomeq-workflow --lib --no-tests fail -E 'test(/home_cinema_(no_sub|with_sub)_multiseat_rejection_reports|run_channel_via_generic_path_multiseat_rejected_recovers|all_channel_multiseat_acceptance_rejects_subs/)'
	cargo nextest run --release -p roomeq-qa --lib --no-tests fail -E 'test(/scorecard_/)'

[group('qa-roomeq')]
qa-roomeq-gd:
	cargo test --release -p roomeq-engine --lib -- --nocapture
	cargo test --release -p roomeq-workflow --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-features:
	cargo run --features qa --bin roomeq-qa-features --no-default-features --release

# Supporting-source room compensation QA: integration test + targeted lib tests.
[group('qa-roomeq')]
qa-roomeq-supporting-source:
	cargo test --release -p autoeq --test roomeq_supporting_source -- --nocapture
	cargo test --release -p roomeq-workflow supporting_source --lib -- --nocapture

# Compact CI-friendly QA run: bounded five-seed fuzzer + small coverage subset.
# The coverage quick subset keeps each scenario's tuned optimizer (bare
# local-optimizer runs cannot improve the FEM corpus and are rejected by the
# final safety gate).
[group('qa-roomeq')]
qa-roomeq-ci:
	cargo run --bin roomeq-fuzzer --release --features="qa,plotly" -- -n 50 --seed 42 --skip-kautz-modal
	cargo run --features qa --bin roomeq-qa-coverage --release --features="cli" -- --quick
	cargo run --features qa --bin roomeq-qa-synthetic --features="cli" --release -- --multiseat-guards-only
	# Leaf-crate unit coverage is enforced by the preceding CI matrix job.
	just qa-roomeq-perceptual

# Audibility-first PR gate: deterministic, offline, and bounded. The synthetic
# easy matrix exercises representative layouts while the nightly recipe owns
# the exhaustive combinatorics.
[group('qa-roomeq')]
qa-audibility-pr:
	cargo test --release -p autoeq-core --doc
	cargo test --release -p autoeq-measurements quality
	cargo test --release -p roomeq-quality acoustic_qa_pr_ --lib
	cargo test --release -p roomeq-workflow correction_acceptance --lib
	cargo test --release -p roomeq-workflow final_safety_gate --lib
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --pr
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier pr

# Repository-backed real/FEM corpus with calibrated blocking quality gates.
[group('qa-roomeq')]
qa-roomeq-acoustic-pr:
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier pr

[group('qa-roomeq')]
qa-roomeq-acoustic-nightly:
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier nightly

# Human + machine report, trend history, and explicit PR resource budgets.
[group('qa-roomeq')]
qa-roomeq-acoustic-report baseline="":
	#!/usr/bin/env bash
	set -euo pipefail
	baseline_args=()
	if [ -n "{{baseline}}" ]; then
	  baseline_args+=(--baseline "{{baseline}}")
	fi
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier pr "${baseline_args[@]}" --output target/qa/roomeq-acoustic.json --markdown-output target/qa/roomeq-acoustic.md --history target/qa/roomeq-acoustic-history.ndjson --max-runtime-ms 900000 --max-peak-rss-kib 4194304

# Deterministically replace paired snapshots after an intentional quality change.
[group('qa-roomeq')]
qa-roomeq-acoustic-recalibrate:
	cargo run --features qa --bin roomeq-qa-acoustic --release -- --tier nightly --recalibrate-baseline

# Per-subsystem coverage floors, stricter around the acoustic acceptance layer.
[group('qa-roomeq')]
qa-roomeq-subsystem-coverage:
	cargo llvm-cov clean --workspace
	cargo llvm-cov test --release --package roomeq-quality --lib --no-report
	cargo llvm-cov test --release --package roomeq-engine --lib --no-report
	cargo llvm-cov test --release --package roomeq-workflow --lib --no-report
	mkdir -p target/qa
	cargo llvm-cov report --release --package roomeq-quality --package roomeq-engine --package roomeq-workflow --json --summary-only --skip-functions --output-path target/qa/roomeq-coverage.json
	python3 scripts/check_roomeq_subsystem_coverage.py target/qa/roomeq-coverage.json

# Fast, deterministic mutation shard for pull requests and local smoke checks.
[group('qa-roomeq')]
qa-roomeq-mutation-smoke:
	mkdir -p target/qa
	cargo mutants --profile release --package autoeq --file 'src/roomeq/acoustic_qa/**/*.rs' --baseline skip --timeout 60 --shard 1/50 --sharding round-robin -j 4 -o target/qa/roomeq-mutants-smoke -- roomeq::acoustic_qa

# Exhaustive mutation testing for quality metrics, acceptance, and corpus contracts.
[group('qa-roomeq')]
qa-roomeq-mutation:
	cargo mutants --profile release --package autoeq --file 'src/roomeq/acoustic_qa/**/*.rs'

# Unit contracts for privacy-safe REW MDAT conversion.
[group('qa-roomeq')]
qa-roomeq-mdat:
	python3 scripts/test_mdat2csv.py

# Exhaustive layout x sub-topology x processing-mode audibility matrix.
[group('qa-roomeq')]
qa-audibility-nightly:
	just qa-roomeq-convergence
	cargo run --features qa --bin roomeq-qa-synthetic --no-default-features --release -- --full-matrix
	just qa-roomeq-acoustic-nightly
	just qa-roomeq-coverage-gate
