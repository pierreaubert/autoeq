# --------------------------------------------------------- -*- just -*-
# How to install Just?
# cargo install just
# ----------------------------------------------------------------------
import 'builds/cross.just'
import 'builds/qa/qa-autoeq.just'
import 'builds/qa/qa-roomeq.just'
import 'builds/qa/qa-export.just'
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
	cargo build --release --features cli --bin autoeq
	cargo build --release --features cli --bin benchmark-autoeq-speaker
	cargo build --release --features cli --bin autoeq-download-speakers

[group('build')]
prod-roomeq:
	cargo build --release --features cli --bin roomeq
	cargo build --release --features qa --bin roomeq-qa-quality
	cargo build --release --features qa --bin roomeq-qa-coverage
	cargo build --release --features qa --bin roomeq-qa-features
	cargo build --release --features qa --bin roomeq-qa-synthetic
	cargo build --release --features cli --bin convert-recording

[group('build')]
dev:
	cargo build --bins --all-features

# ----------------------------------------------------------------------
# TEST
# we use --release (faster overall since the tests do some computations)
# ----------------------------------------------------------------------

[group('test')]
check:
	cargo check --workspace --all-targets --all-features

[group('test')]
test:
	cargo test --workspace --all-targets --all-features --release

# Each optimizer internally forks rayon evaluators over all
# cores, so the effective thread count is num_cpus × num_cpus. On small-
# RAM boxes this OOMs. Cap via `RUST_TEST_THREADS` (default = 2 so BEM
# tests still interleave but memory stays bounded). Override with
# `just test-autoeq threads=N`.
[group('test')]
test-autoeq threads="2":
	RUST_TEST_THREADS={{threads}} cargo test --tests --release

[group('test')]
ntest:
	cargo nextest run --release --no-fail-fast --lib --bins --examples

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

# ----------------------------------------------------------------------
# LINT / FORMAT
# ----------------------------------------------------------------------

[group('lint')]
lint:
	# The optional plotly dependency embeds templates from an external cache path
	# that is not available in all checkout environments. Keep lint hermetic and
	# lint the default production surface; plotly builds remain covered by CI.
	cargo clippy --all -- -D warnings

alias format := fmt

[group('lint')]
fmt:
	cargo fmt --all

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
	cargo build --profile dist --features cli --bin autoeq
	cargo build --profile dist --features cli --bin benchmark-autoeq-speaker
	cargo build --profile dist --features cli --bin autoeq-download-speakers

[group('dist')]
dist-roomeq:
	cargo build --profile dist --features cli --bin roomeq

# Plotly-gated bins (skipped by `--workspace` because of required-features).
[group('dist')]
dist-plot-bins:
	cargo build --profile dist --bin roomeq-fuzzer --features qa,plotly

# ----------------------------------------------------------------------
# CLEAN
# ----------------------------------------------------------------------

clean:
	cargo clean
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
	cargo run --features cli --bin autoeq-download-speakers --release

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

[group('bench')]
bench-autoeq: bench-autoeq-speaker

[group('bench')]
bench-autoeq-speaker:
	# either jobs=1 or --no-parallel ; or a mix if you have a lot of
	# CPU cores
	cargo run --release --features cli --bin benchmark-autoeq-speaker -- --qa --jobs 1

# ----------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------

[group('examples')]
examples-autoeq:
	cargo run --release --example headphone_loss_validation

# ----------------------------------------------------------------------
# PUBLISH
# ----------------------------------------------------------------------

[group('publish')]
publish-autoeq:
	cargo publish

# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

[group('demo')]
demo-headphone-loss:
	cargo run --release --example headphone_loss_demo --features="plotly" -- \
	--spl "./data_tests/headphones/asr/bowerwilkins_p7/Bowers & Wilkins P7.csv" \
	--target "./data_tests/targets/harman-over-ear-2018.csv"

# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------

qa : qa-autoeq-all qa-roomeq-all qa-export-all
