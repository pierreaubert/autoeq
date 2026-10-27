# --------------------------------------------------------- -*- just -*-
# How to install Just?
# cargo install just
# ----------------------------------------------------------------------
# Use `mbx` when installed, else plain `cargo` (mirrors ../math-audio).
cargo := `if command -v mbx >/dev/null 2>&1; then echo mbx; else echo cargo; fi`

import 'builds/cross-autoeq.just'
import 'builds/qa/qa-autoeq.just'
import 'builds/qa/qa-roomeq.just'
import 'builds/qa/qa-export.just'
import 'builds/qa/qa-wolfram.just'
# ----------------------------------------------------------------------

_default:
	just --list

# ----------------------------------------------------------------------
# BUILD
# ----------------------------------------------------------------------

# Build all release binaries.
[group('build')]
prod: prod-autoeq prod-roomeq

[group('build')]
prod-autoeq:
	{{cargo}} build -p autoeq-cli --release --bin autoeq
	{{cargo}} build -p autoeq-cli --release --bin benchmark-autoeq-speaker
	{{cargo}} build -p autoeq-cli --release --bin autoeq-download-speakers

# Speaker-data visualization demos are owned by AutoEQ.
[group('build')]
demo-d3rs-spinorama:
	{{cargo}} build --manifest-path crates/autoeq-gpui-examples/Cargo.toml --release --bin d3rs-spinorama

[group('build')]
demo-px-spinorama:
	{{cargo}} build --manifest-path crates/autoeq-gpui-examples/Cargo.toml --release --bin px-spinorama

[group('build')]
roomeq:
	{{cargo}} build -p roomeq-cli --release --bin roomeq

# Build the canonical native producer used by standalone Python report commands.
[group('build')]
report-producer:
	{{cargo}} build --release -p autoeq-report-wasm --bin report-producer

# Rebuild the report-shell WASM bundles into crates/autoeq-report-wasm/dist/.
# 2D canvas bundle builds on stable; the GPUI viewer needs nightly
# (gpui-web's wasm_thread/parking_lot-nightly deps). Requires wasm-bindgen
# matching the locked wasm-bindgen crate (0.2.129) on PATH (or set WANDBIN).
# Reports embed dist/ at generation time.
# The 2D exports live in the cdylib-only autoeq-report-wasm-shell crate;
# autoeq-report-wasm itself is a plain rlib (see its Cargo.toml).
# Resolve Cargo metadata so CARGO_TARGET_DIR and configured target directories work.
[group('build')]
report-dist:
	#!/usr/bin/env bash
	set -euo pipefail
	target_dir="$({{cargo}} metadata --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
	{{cargo}} build --release --target wasm32-unknown-unknown -p autoeq-report-wasm-shell --target-dir "$target_dir"
	mkdir -p crates/autoeq-report-wasm/dist
	"${WANDBIN:-wasm-bindgen}" --target web --out-name report2d --out-dir crates/autoeq-report-wasm/pkg2d "$target_dir/wasm32-unknown-unknown/release/autoeq_report_wasm_shell.wasm"
	cp crates/autoeq-report-wasm/pkg2d/report2d.js crates/autoeq-report-wasm/dist/report2d.js
	cp crates/autoeq-report-wasm/pkg2d/report2d_bg.wasm crates/autoeq-report-wasm/dist/report2d.wasm

[group('build')]
report-dist-gpui:
	#!/usr/bin/env bash
	set -euo pipefail
	target_dir="$(cargo +nightly metadata --no-deps --format-version 1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
	cargo +nightly build --release --target wasm32-unknown-unknown -p autoeq-report-gpui --target-dir "$target_dir"
	mkdir -p crates/autoeq-report-wasm/dist
	"${WANDBIN:-wasm-bindgen}" --target web --out-name reportgpui --out-dir crates/autoeq-report-gpui/pkg "$target_dir/wasm32-unknown-unknown/release/autoeq_report_gpui.wasm"
	cp crates/autoeq-report-gpui/pkg/reportgpui.js crates/autoeq-report-wasm/dist/reportgpui.js
	cp crates/autoeq-report-gpui/pkg/reportgpui_bg.wasm crates/autoeq-report-wasm/dist/reportgpui.wasm

[group('build')]
prod-roomeq: roomeq
	{{cargo}} build -p roomeq-qa --release --bin roomeq-qa-quality
	{{cargo}} build -p roomeq-qa --release --bin roomeq-qa-coverage
	{{cargo}} build -p roomeq-qa --release --bin roomeq-qa-features
	{{cargo}} build -p roomeq-qa --release --bin roomeq-qa-synthetic
	{{cargo}} build -p roomeq-cli --release --bin convert-recording

[group('build')]
dev:
	{{cargo}} build -p autoeq-cli -p roomeq-cli -p roomeq-qa --bins --all-features

# ----------------------------------------------------------------------
# TEST — lanes (refactor-20261025 Phase 1)
# ----------------------------------------------------------------------
# Cargo package/target selection bounds compilation; nextest profiles
# (`.config/nextest.toml`) only control execution. `test`/`test-crate`
# compile with the `test` Cargo profile, `ntest`/`test-cli` with the
# single shared `qa` artifact set. `ntest` remains the complete native
# inventory — do not turn it into a fast subset; use `test-pr` for
# affected-package feedback.

# Ordinary library development set; keep in sync with `default-members`
# in Cargo.toml and `build_lanes` in scripts/crate_partition_policy.json
# (the partition checker enforces the latter against Cargo metadata).
lib_packages := "-p autoeq-core -p autoeq-artifacts -p autoeq-fir -p autoeq-plot -p autoeq-measurements -p autoeq-optim -p autoeq-optim-types -p autoeq-objective -p autoeq-workflow -p roomeq-model -p roomeq-analysis -p roomeq-dsp -p roomeq-quality -p roomeq-engine -p roomeq-export -p roomeq-synthetic -p roomeq-workflow"

[group('test')]
check:
	{{cargo}} check --workspace --all-targets --all-features

# Optional manual CLI build. Owning integration tests already make Cargo
# prepare these executables; test lanes do not need an unconditional prebuild.
[group('test')]
build-test-bins profile="qa":
	{{cargo}} build -p autoeq-cli -p roomeq-cli --locked --profile {{profile}} --bin autoeq --bin roomeq

# Fast library lane: the default-member libraries and their own
# integration tests, no binary prebuild, fail at first failure.
[group('test')]
test:
	{{cargo}} nextest run --locked --cargo-profile test --profile fast {{lib_packages}} --lib --tests --no-tests fail

# Focused feedback for one package (library plus its owning
# integration targets). Extra args pass through to nextest.
[group('test')]
test-crate package *args:
	{{cargo}} nextest run --locked --cargo-profile test --profile fast -p {{package}} --lib --tests --no-tests fail {{args}}

# Changed packages (vs base) plus reverse dependencies; shared-config
# changes widen to the whole library set. See scripts/changed_packages.py.
[group('test')]
test-pr base="main":
	#!/usr/bin/env bash
	set -euo pipefail
	packages="$(python3 scripts/changed_packages.py --base {{base}})"
	if [ -z "$packages" ]; then
		echo "test-pr: no package-owned changes vs {{base}}"
		exit 0
	fi
	echo "test-pr: $packages"
	{{cargo}} nextest run --locked --cargo-profile test --profile fast $packages --lib --tests --no-tests fail

# Real executable tests: the CLI-owned integration targets that spawn the
# CLI binaries, built and run with the shared `qa` artifact set.
[group('test')]
test-cli:
	python3 scripts/run_with_report_producer.py --profile qa --cargo "{{cargo}}" -- {{cargo}} nextest run --locked --cargo-profile qa --profile qa -p autoeq-cli -p roomeq-cli --test autoeq_golden_tests --test autoeq_integration_test --test broadband_matching_test --test multi_measurement_test --test roomeq_integration_test --test roomeq_capture_verification --test product_renderer_capabilities --no-tests fail

# Each optimizer internally forks rayon evaluators over all
# cores, so the effective thread count is num_cpus × num_cpus. On small-
# RAM boxes this OOMs. Cap via `RUST_TEST_THREADS` (default = 2 so BEM
# tests still interleave but memory stays bounded). Override with
# `just test-autoeq threads=N`.
[group('test')]
test-autoeq threads="2":
	RUST_TEST_THREADS={{threads}} python3 scripts/run_with_report_producer.py --profile qa --cargo "{{cargo}}" -- {{cargo}} test --workspace --lib --tests --all-features --profile qa

# Complete native Rust test/benchmark inventory; all cases are preserved.
# Proven-empty executable/example harnesses are excluded (see
# docs/baselines/refactor-20261025/empty-harness-scope-proof.json). Compiled with the
# `qa` Cargo profile so helper executables and test artifacts share one
# configuration instead of a release prebuild plus a second test build.
[group('test')]
ntest:
	python3 scripts/run_with_report_producer.py --profile qa --cargo "{{cargo}}" -- {{cargo}} nextest run --locked --cargo-profile qa --profile full --workspace --lib --tests --benches --all-features

# Examples are compiled explicitly instead of linked as empty test harnesses.
[group('test')]
check-examples:
	{{cargo}} check --locked --workspace --examples --all-features

# Criterion's shipped bench profile is checked independently of its QA-mode
# correctness cases, which remain part of `ntest`.
[group('test')]
check-benches:
	{{cargo}} bench --locked --workspace --all-features --no-run

[group('test')]
test-docs:
	{{cargo}} test --locked --workspace --all-features --doc

# WP0 crate-partition gates. Keep the fast checker tests and graph/ownership
# report independently runnable; the umbrella also regenerates both schemas.
[group('test')]
check-crate-partition: test-crate-partition-checker check-crate-partition-fitness check-roomeq-schema-baselines

[group('test')]
test-crate-partition-checker:
	python3 -m unittest scripts/test_check_crate_partition.py

[group('test')]
check-crate-partition-fitness:
	python3 scripts/check_crate_partition.py

[group('test')]
check-roomeq-schema-baselines:
	python3 scripts/check_roomeq_schema_baselines.py

[group('test')]
test-roomeq-gui:
	PYTHONPATH=ui/roomeq-gui python3 -m unittest discover ui/roomeq-gui/tests

[group('build')]
roomeq-gui:
	PYTHONPATH=ui/roomeq-gui python3 -m roomeq_gui

[group('test')]
test-recording-gui:
	PYTHONPATH=ui/recording-gui python3 -m unittest discover ui/recording-gui/tests

[group('build')]
recording-gui *args:
	PYTHONPATH=ui/recording-gui python3 -m recording_gui {{args}}

# ----------------------------------------------------------------------
# LINT / FORMAT
# ----------------------------------------------------------------------

[group('lint')]
lint:
	{{cargo}} clippy --all -- -D warnings

[group('lint')]
clippy:
	{{cargo}} clippy --workspace --tests --bins --all-features -- -D warnings

alias format := fmt

[group('lint')]
fmt:
	{{cargo}} fmt --all

# ----------------------------------------------------------------------
# DIST — release-cut profile (fat LTO + codegen-units = 1)
# ----------------------------------------------------------------------
# Artifacts land in `target/dist/` (NOT `target/release/`). Compile time is
# noticeably longer than `prod-*`; only run these for actual release cuts.

# Top-level umbrella — builds all shipping binaries, including the plot bins.
[group('dist')]
dist: dist-autoeq dist-roomeq dist-plot-bins

[group('dist')]
dist-autoeq:
	{{cargo}} build -p autoeq-cli --profile dist --bin autoeq
	{{cargo}} build -p autoeq-cli --profile dist --bin benchmark-autoeq-speaker
	{{cargo}} build -p autoeq-cli --profile dist --bin autoeq-download-speakers

[group('dist')]
dist-roomeq:
	{{cargo}} build -p roomeq-cli --profile dist --bin roomeq

# AutoEQ QA fuzzer is gated by the workspace `qa` feature.
[group('dist')]
dist-plot-bins:
	{{cargo}} build -p roomeq-qa --profile dist --bin roomeq-fuzzer

# ----------------------------------------------------------------------
# CLEAN
# ----------------------------------------------------------------------

clean:
	{{cargo}} clean
	find . -name '*~' -exec rm {} \; -print
	rm -f *.wav *.log TAGS ETAGS
	rm -fr fuzzer_output mutants.out
	rm -fr venv .tokensave .venv
	rm -fr data_generated

# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------

[group('download')]
download-speakers:
	{{cargo}} run -p autoeq-cli --bin autoeq-download-speakers --release

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

[group('bench')]
bench-autoeq: bench-autoeq-speaker

[group('bench')]
bench-autoeq-speaker:
	# either jobs=1 or --no-parallel ; or a mix if you have a lot of
	# CPU cores
	{{cargo}} run -p autoeq-cli --release --bin benchmark-autoeq-speaker -- --qa --jobs 1

# ----------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------

[group('examples')]
examples-autoeq:
	{{cargo}} run -p autoeq-objective --release --example headphone_loss_validation

# ----------------------------------------------------------------------
# PUBLISH
# ----------------------------------------------------------------------

[group('publish')]
publish-autoeq:
	{{cargo}} release publish --workspace

# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

[group('demo')]
demo-headphone-loss:
	{{cargo}} run -p autoeq-objective --release --example headphone_loss_demo -- \
	--spl "./data_tests/headphones/asr/bowerwilkins_p7/Bowers & Wilkins P7.csv" \
	--target "./data_tests/targets/harman-over-ear-2018.csv"

# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------

# Run every QA domain and report all of them: a red domain must not hide
# the evidence of the others. The aggregate exit code still fails when
# any domain fails.
qa:
	#!/usr/bin/env bash
	set -u
	status=0
	just qa-autoeq-all || status=1
	just qa-roomeq-all || status=1
	just qa-export-all || status=1
	just qa-wolfram-validation || status=1
	exit $status

# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------

[group('install')]
rust-install-tools:
	~/.cargo/bin/rustup default stable
	~/.cargo/bin/cargo install cargo-wizard
	~/.cargo/bin/cargo install cross
	~/.cargo/bin/cargo install cargo-binstall
	~/.cargo/bin/cargo install cargo-release
	~/.cargo/bin/cargo binstall cargo-nextest --secure
	~/.cargo/bin/cargo install cargo-insta
	~/.cargo/bin/cargo install tokensave
	~/.cargo/bin/tokensave init
	~/.cargo/bin/cargo install samply
	~/.cargo/bin/cargo install mbx
	~/.cargo/bin/cargo install --git https://github.com/rtk-ai/rtk


