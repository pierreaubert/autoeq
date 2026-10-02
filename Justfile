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
	{{cargo}} build --release --features cli --bin autoeq
	{{cargo}} build --release --features cli --bin benchmark-autoeq-speaker
	{{cargo}} build --release --features cli --bin autoeq-download-speakers

[group('build')]
roomeq:
	{{cargo}} build --release --features cli --bin roomeq

# Rebuild the report-shell WASM bundles into crates/autoeq-report-wasm/dist/.
# 2D canvas bundle builds on stable; the GPUI viewer needs nightly
# (gpui-web's wasm_thread/parking_lot-nightly deps). Requires wasm-bindgen
# 0.2.128 on PATH (or set WANDBIN). Reports embed dist/ at generation time.
# The 2D exports live in the cdylib-only autoeq-report-wasm-shell crate;
# autoeq-report-wasm itself is a plain rlib (see its Cargo.toml).
[group('build')]
report-dist:
	{{cargo}} build --release --target wasm32-unknown-unknown -p autoeq-report-wasm-shell
	${WANDBIN:-wasm-bindgen} --target web --out-name report2d --out-dir crates/autoeq-report-wasm/pkg2d target/wasm32-unknown-unknown/release/autoeq_report_wasm_shell.wasm
	cp crates/autoeq-report-wasm/pkg2d/report2d.js crates/autoeq-report-wasm/dist/report2d.js
	cp crates/autoeq-report-wasm/pkg2d/report2d_bg.wasm crates/autoeq-report-wasm/dist/report2d.wasm

[group('build')]
report-dist-gpui:
	cargo +nightly build --release --target wasm32-unknown-unknown -p autoeq-report-gpui
	${WANDBIN:-wasm-bindgen} --target web --out-name reportgpui --out-dir crates/autoeq-report-gpui/pkg target/wasm32-unknown-unknown/release/autoeq_report_gpui.wasm
	cp crates/autoeq-report-gpui/pkg/reportgpui.js crates/autoeq-report-wasm/dist/reportgpui.js
	cp crates/autoeq-report-gpui/pkg/reportgpui_bg.wasm crates/autoeq-report-wasm/dist/reportgpui.wasm

[group('build')]
prod-roomeq: roomeq
	{{cargo}} build --release --features qa --bin roomeq-qa-quality
	{{cargo}} build --release --features qa --bin roomeq-qa-coverage
	{{cargo}} build --release --features qa --bin roomeq-qa-features
	{{cargo}} build --release --features qa --bin roomeq-qa-synthetic
	{{cargo}} build --release --features cli --bin convert-recording

[group('build')]
dev:
	{{cargo}} build --bins --all-features

# ----------------------------------------------------------------------
# TEST
# we use --release (faster overall since the tests do some computations)
# ----------------------------------------------------------------------

[group('test')]
check:
	{{cargo}} check --workspace --all-targets --all-features

[group('test')]
test:
	{{cargo}} test --workspace --all-targets --all-features --release

# Each optimizer internally forks rayon evaluators over all
# cores, so the effective thread count is num_cpus × num_cpus. On small-
# RAM boxes this OOMs. Cap via `RUST_TEST_THREADS` (default = 2 so BEM
# tests still interleave but memory stays bounded). Override with
# `just test-autoeq threads=N`.
[group('test')]
test-autoeq threads="2":
	RUST_TEST_THREADS={{threads}} {{cargo}} test --tests --release

[group('test')]
ntest:
	{{cargo}} nextest run --release --no-fail-fast --lib --bins --examples --tests --workspace --all-targets --all-features

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
	{{cargo}} build --profile dist --features cli --bin autoeq
	{{cargo}} build --profile dist --features cli --bin benchmark-autoeq-speaker
	{{cargo}} build --profile dist --features cli --bin autoeq-download-speakers

[group('dist')]
dist-roomeq:
	{{cargo}} build --profile dist --features cli --bin roomeq

# AutoEQ QA fuzzer is gated by the workspace `qa` feature.
[group('dist')]
dist-plot-bins:
	{{cargo}} build --profile dist --bin roomeq-fuzzer --features qa

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
	{{cargo}} run --features cli --bin autoeq-download-speakers --release

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

[group('bench')]
bench-autoeq: bench-autoeq-speaker

[group('bench')]
bench-autoeq-speaker:
	# either jobs=1 or --no-parallel ; or a mix if you have a lot of
	# CPU cores
	{{cargo}} run --release --features cli --bin benchmark-autoeq-speaker -- --qa --jobs 1

# ----------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------

[group('examples')]
examples-autoeq:
	{{cargo}} run --release --example headphone_loss_validation

# ----------------------------------------------------------------------
# PUBLISH
# ----------------------------------------------------------------------

[group('publish')]
publish-autoeq:
	{{cargo}} publish

# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

[group('demo')]
demo-headphone-loss:
	{{cargo}} run --release --example headphone_loss_demo -- \
	--spl "./data_tests/headphones/asr/bowerwilkins_p7/Bowers & Wilkins P7.csv" \
	--target "./data_tests/targets/harman-over-ear-2018.csv"

# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------

qa : qa-autoeq-all qa-roomeq-all qa-export-all qa-wolfram-validation
