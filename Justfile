# --------------------------------------------------------- -*- just -*-
# How to install Just?
# cargo install just
# ----------------------------------------------------------------------
import 'builds/cross.just'
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
	cargo build --release --bin autoeq
	cargo build --release --bin benchmark-autoeq-speaker
	cargo build --release --bin autoeq-download-speakers

[group('build')]
prod-roomeq:
	cargo build --release --bin roomeq
	cargo build --release --bin roomeq-qa-quality
	cargo build --release --bin roomeq-qa-coverage
	cargo build --release --bin roomeq-qa-features
	cargo build --release --bin roomeq-qa-synthetic
	cargo build --release --bin convert-recording

[group('build')]
dev:
	cargo build --bins

# ----------------------------------------------------------------------
# TEST
# we use --release (faster overall since the tests do some computations)
# ----------------------------------------------------------------------

[group('test')]
check:
	cargo check --lib --bins --tests --examples

[group('test')]
test:
	cargo test --lib --bins --tests --examples --release

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

# ----------------------------------------------------------------------
# LINT / FORMAT
# ----------------------------------------------------------------------

[group('lint')]
lint:
	cargo clippy --all --features plotly -- -D warnings

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
	cargo build --profile dist --bin autoeq
	cargo build --profile dist --bin benchmark-autoeq-speaker
	cargo build --profile dist --bin autoeq-download-speakers

[group('dist')]
dist-roomeq:
	cargo build --profile dist --bin roomeq

# Plotly-gated bins (skipped by `--workspace` because of required-features).
[group('dist')]
dist-plot-bins:
	cargo build --profile dist --bin roomeq-fuzzer --features plotly

# ----------------------------------------------------------------------
# CLEAN
# ----------------------------------------------------------------------

clean:
	cargo clean
	find . -name '*~' -exec rm {} \; -print
	rm -f *.wav
	rm -fr fuzzer_output
	rm -fr data_generated

# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------

[group('download')]
download-speakers:
	cargo run --bin autoeq-download-speakers --release

# ----------------------------------------------------------------------
# BENCH
# ----------------------------------------------------------------------

[group('bench')]
bench-autoeq: bench-autoeq-speaker

[group('bench')]
bench-autoeq-speaker:
	# either jobs=1 or --no-parallel ; or a mix if you have a lot of
	# CPU cores
	cargo run --release --bin benchmark-autoeq-speaker -- --qa --jobs 1

# The `benchmark-convergence` binary lives in the `math-optimisation` crate
# (github.com/pierreaubert/math-audio). Clone that repository to run it.
[group('bench')]
bench-convergence:
	@echo "Run from the math-audio repository:"
	@echo "  cargo run --release --bin benchmark-convergence"

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

qa : qa-autoeq qa-roomeq

# ----------------------------------------------------------------------
# QA AUTOEQ
# ----------------------------------------------------------------------

[group('qa-autoeq')]
qa-autoeq: prod-autoeq \
	qa-ascilab-6b \
	qa-jbl-m2-flat qa-jbl-m2-score \
	qa-beyerdynamic-dt1990pro \
	qa-edifierw830nb

[group('qa-autoeq')]
qa-ascilab-6b:
	./target/release/autoeq --speaker="AsciLab F6B" --version asr --measurement CEA2034 \
	--algo autoeq:de --loss speaker-score -n 7 --min-freq=30 --max-q=6 \
	--maxeval 100000 --qa 0.5

[group('qa-autoeq')]
qa-jbl-m2-flat:
	./target/release/autoeq --speaker="JBL M2" --version eac --measurement CEA2034 \
	--algo autoeq:de --loss speaker-flat -n 7 --min-freq=20 --max-q=6 --peq-model hp-pk \
	--maxeval 100000 --qa 0.5

[group('qa-autoeq')]
qa-jbl-m2-score:
	./target/release/autoeq --speaker="JBL M2" --version eac --measurement CEA2034 \
	--algo autoeq:de --loss speaker-score -n 7 --min-freq=20 --max-q=6 --peq-model hp-pk \
	--maxeval 100000 --qa 0.5

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro: qa-beyerdynamic-dt1990pro-flat qa-beyerdynamic-dt1990pro-score	qa-beyerdynamic-dt1990pro-score2

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-score:
	./target/release/autoeq -n 5 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv --loss headphone-score  \
	--maxeval 100000 --qa 3.0

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-score2:
	./target/release/autoeq -n 7 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-score	--max-db 6 --max-q 6 --algo mh:rga --maxeval 100000 --min-freq=20 --max-freq 10000 --peq-model hp-pk-lp --min-q 0.6 --min-db 0.25 \
	--qa 1.5

[group('qa-autoeq')]
qa-beyerdynamic-dt1990pro-flat:
	./target/release/autoeq -n 5 \
	--curve ./data_tests/headphones/asr/beyerdynamic_dt1990pro/Beyerdynamic\ DT1990\ Pro\ Headphone\ Frequency\ Response\ Measurement.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--loss headphone-flat  --max-db 6 --max-q 6 --maxeval 100000 --algo mh:pso --min-freq=20 --max-freq 10000 --peq-model pk \
	--qa 0.5

[group('qa-autoeq')]
qa-edifierw830nb: qa-edifierw830nb-autoeqde qa-edifierw830nb-mhrga qa-edifierw830nb-mhfirefly

[group('qa-autoeq')]
qa-edifierw830nb-autoeqde:
	./target/release/autoeq -n 9 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.08 \
	--algo autoeq:de --population 70 --maxeval 80000 --seed 42 \
	--qa 6.0

[group('qa-autoeq')]
qa-edifierw830nb-mhrga:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.0000001 --algo mh:rga --population 100 --maxeval 80000 \
	--qa 2.5

[group('qa-autoeq')]
qa-edifierw830nb-mhfirefly:
	./target/release/autoeq -n 5 \
	--curve data_tests/headphones/asr/edifierw830nb/Edifier\ W830NB.csv \
	--target ./data_tests/targets/harman-over-ear-2018.csv \
	--min-freq 50 --max-freq 16000 --max-q 8 --max-db 8 \
	--loss headphone-score \
	--min-spacing-oct 0.04 --atolerance 0.00000001 --tolerance 0.000000001 --algo mh:rga --population 80 --maxeval 80000 \
	--qa 2.5

# ----------------------------------------------------------------------
# QA ROOMEQ
# ----------------------------------------------------------------------

# Ensure Python venv exists and has dependencies installed
[group('setup')]
ensure-venv:
	#!/usr/bin/env bash
	set -euo pipefail
	if [ ! -f ./venv/bin/python3 ]; then
		echo "Creating Python venv..."
		python3 -m venv ./venv
	fi
	if ! ./venv/bin/python3 -c "import plotly" 2>/dev/null; then
		echo "Installing Python dependencies..."
		./venv/bin/pip install -r ./scripts/requirements.txt
	fi

[group('setup')]
export-test-systems-help:
	@echo "RoomEQ external-export validator setup"
	@echo "======================================"
	@echo
	@echo "Structural checks, no external tools required:"
	@echo "  cargo test -p autoeq export::tests"
	@echo "  just qa-export-camilladsp  # requires the real CamillaDSP binary"
	@echo "  just qa-export-pipewire    # runs PipeWire validation in Docker"
	@echo "  just qa-export-equalizer-apo  # requires a Windows VM validator command"
	@echo
	@echo "Install helper:"
	@echo "  just install-export-test-systems"
	@echo
	@echo "Optional validator environment variables:"
	@echo "  ROOMEQ_CAMILLADSP_VALIDATE_CMD"
	@echo "  ROOMEQ_EQUALIZER_APO_VALIDATE_CMD"
	@echo "  ROOMEQ_EASYEFFECTS_VALIDATE_CMD"
	@echo "  ROOMEQ_WAVELET_VALIDATE_CMD"
	@echo "  ROOMEQ_PIPEWIRE_VALIDATE_CMD"
	@echo "  ROOMEQ_ROON_VALIDATE_CMD"
	@echo
	@echo "Each command may use {config} or {file} as the generated export path."
	@echo "If no placeholder is present, the path is appended as the final argument."
	@echo
	@echo "Examples:"
	@echo "  ROOMEQ_CAMILLADSP_VALIDATE_CMD='camilladsp --check {config}' cargo test -p autoeq tool_contract_camilladsp"
	@echo "  ROOMEQ_PIPEWIRE_VALIDATE_CMD='path/to/validate-pipewire {config}' cargo test -p autoeq tool_contract_pipewire"
	@echo "  ROOMEQ_EASYEFFECTS_VALIDATE_CMD='path/to/validate-easyeffects {config}' cargo test -p autoeq tool_contract_easyeffects"
	@echo
	@echo "Manual/application-backed validators:"
	@echo "  Equalizer APO: install on Windows and point ROOMEQ_EQUALIZER_APO_VALIDATE_CMD at an import/syntax-check script."
	@echo "  Wavelet: Android app; keep structural tests unless you provide a custom validator script."
	@echo "  Roon: normally manual import/configuration; keep structural JSON tests unless you provide a custom validator script."

[group('setup')]
install-export-test-systems:
	#!/usr/bin/env bash
	set -euo pipefail
	echo "Installing optional RoomEQ export validation tools where this platform supports them."
	echo "These are only needed for env-var-backed smoke tests; structural tests need no extra tools."
	echo
	if command -v cargo >/dev/null 2>&1; then
		echo "Installing CamillaDSP from its official repository..."
		cargo install --git https://github.com/HEnquist/camilladsp.git \
			--rev 05e9cfcdf43c0dfe078ed3feb8af4c8bd701fd74 --locked || {
			echo "cargo install camilladsp failed; install CamillaDSP manually and set ROOMEQ_CAMILLADSP_VALIDATE_CMD."
		}
	else
		echo "cargo not found; install Rust first if you want the CamillaDSP CLI via cargo."
	fi
	case "$(uname -s)" in
		Darwin)
			if command -v brew >/dev/null 2>&1; then
				echo "Installing PipeWire tools with Homebrew..."
				brew install pipewire || true
			else
				echo "Homebrew not found; install PipeWire manually if you want ROOMEQ_PIPEWIRE_VALIDATE_CMD."
			fi
			echo "EasyEffects is primarily a Linux desktop app; skipping automatic macOS install."
			;;
		Linux)
			if command -v apt-get >/dev/null 2>&1; then
				echo "Installing PipeWire and EasyEffects with apt..."
				sudo apt-get update
				sudo apt-get install -y pipewire-bin easyeffects
			elif command -v dnf >/dev/null 2>&1; then
				echo "Installing PipeWire and EasyEffects with dnf..."
				sudo dnf install -y pipewire easyeffects
			elif command -v pacman >/dev/null 2>&1; then
				echo "Installing PipeWire and EasyEffects with pacman..."
				sudo pacman -S --needed pipewire easyeffects
			else
				echo "No supported Linux package manager detected; install pipewire/pw-cli and EasyEffects manually."
			fi
			;;
		*)
			echo "Unsupported OS for automatic installs. Use export-test-systems-help for manual setup."
			;;
	esac
	echo
	echo "Next:"
	echo "  1. Run: just export-test-systems-help"
	echo "  2. Set ROOMEQ_*_VALIDATE_CMD variables for real tool smoke tests."
	echo "  3. Run: cargo test -p autoeq export::tests"

[group('qa')]
qa-export-camilladsp:
	#!/usr/bin/env bash
	set -euo pipefail
	validator="${ROOMEQ_CAMILLADSP_BIN:-camilladsp}"
	if ! command -v "$validator" >/dev/null 2>&1; then
		echo "CamillaDSP validator not found: $validator" >&2
		echo "Run 'just install-export-test-systems' or set ROOMEQ_CAMILLADSP_BIN." >&2
		exit 1
	fi
	ROOMEQ_CAMILLADSP_VALIDATE_CMD="$validator --check {config}" \
		cargo nextest run -p autoeq --lib --no-tests fail -E 'test(/camilladsp/)'

[group('qa')]
qa-export-equalizer-apo:
	#!/usr/bin/env bash
	set -euo pipefail
	validator="${ROOMEQ_EQUALIZER_APO_VALIDATE_CMD:-}"
	if [[ -z "$validator" ]]; then
		echo "A Windows Equalizer APO validator is required." >&2
		echo "Set ROOMEQ_EQUALIZER_APO_VALIDATE_CMD to a command that accepts {config}." >&2
		echo "The command should copy the generated file into a Windows VM and validate it with the installed Equalizer APO tooling." >&2
		exit 1
	fi
	ROOMEQ_EQUALIZER_APO_VALIDATE_CMD="$validator" \
		cargo nextest run -p autoeq --lib --no-tests fail -E 'test(/equalizer_apo/)'

[group('qa')]
qa-export-pipewire:
	#!/usr/bin/env bash
	set -euo pipefail
	if ! command -v docker >/dev/null 2>&1; then
		echo "Docker is required for Linux PipeWire export validation." >&2
		exit 1
	fi
	if ! docker info >/dev/null 2>&1; then
		echo "Docker is installed but its daemon is not available." >&2
		exit 1
	fi
	requested_base_image="${AUTOEQ_PIPEWIRE_BASE_IMAGE:-math-audio-base-linux-arm64:latest}"
	if [[ -z "$(docker image ls "$requested_base_image" -q | head -n 1)" ]]; then
		echo "AutoEQ Linux test base image not found: $requested_base_image" >&2
		echo "Set AUTOEQ_PIPEWIRE_BASE_IMAGE to an existing Linux test image." >&2
		exit 1
	fi
	base_image="$requested_base_image"
	toolchain="$(docker run --rm --entrypoint bash "$base_image" -lc 'rustup toolchain list | sed -n "s/ (active.*//p" | head -n 1')"
	if [[ -z "$toolchain" ]]; then
		echo "Could not determine the active Rust toolchain in $base_image." >&2
		exit 1
	fi
	docker build --build-arg BASE_IMAGE="$base_image" -t autoeq-qa-pipewire -f builds/qa/Dockerfile.pipewire .
	docker run --rm \
		-v "$(pwd)":/workspace \
		-v autoeq-pipewire-cargo-registry:/root/.cargo/registry \
		-v autoeq-pipewire-cargo-git:/root/.cargo/git \
		-v autoeq-pipewire-target:/target \
		-w /workspace \
		-e CARGO_TARGET_DIR=/target \
		-e RUSTUP_TOOLCHAIN="$toolchain" \
		-e CARGO_HTTP_MULTIPLEXING=false \
		-e CARGO_NET_RETRY=10 \
		autoeq-qa-pipewire \
		bash -lc 'ROOMEQ_PIPEWIRE_VALIDATE_CMD="bash scripts/validate-pipewire-config.sh {config}" bash scripts/run-pipewire-export-tests.sh'

[group('qa-roomeq')]
qa-roomeq: qa-roomeq-small-stereo-20 \
	qa-roomeq-small-stereo-21 \
	qa-roomeq-small-stereo-22 \
	qa-roomeq-convergence \
	qa-roomeq-coverage \
	qa-roomeq-synthetic \
	qa-roomeq-gd \
	qa-roomeq-features

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
	  cargo run --bin roomeq-qa-quality --release -- --jobs {{jobs}}
	else
	  cargo run --bin roomeq-qa-quality --release
	fi

[group('qa-roomeq')]
qa-roomeq-small-stereo-20: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	      mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_0; \
	      cargo run --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_0/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_0/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_$method.json; \
		  ./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_$method.json; \
	  done \
	done


[group('qa-roomeq')]
qa-roomeq-small-stereo-21: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_1; \
	    cargo run --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_1/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_1/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_$method.json; \
	  done \
	done


[group('qa-roomeq')]
qa-roomeq-small-stereo-22: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_2; \
	    cargo run --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_2/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/small_stereo_2_2/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_2/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/small_stereo_2_2/dsp_$method.json; \
	  done \
	done

[group('qa-roomeq')]
qa-roomeq-small-stereo-51: ensure-venv
	@for method in iir fir mixed; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/medium_surround_5_1; \
	    cargo run --bin roomeq --release -- \
	        --config       ./data_tests/roomeq/generate/$algo/medium_surround_5_1/config.json \
		    --override-config ./data_tests/roomeq/generate/optimiser-config/medium_surround_5_1/optimiser-$method.json \
		    --output       ./data_generated/roomeq/generated/$algo/medium_surround_5_1/dsp_$method.json; \
		./venv/bin/python3 ./scripts/display-roomeq.py \
		                   ./data_generated/roomeq/generated/$algo/medium_surround_5_1/dsp_$method.json; \
	  done \
	done

# ----------------------------------------------------------------------
# QA ROOMEQ — Multi-Measurement (5 listening positions per speaker)
# Tests the multi-objective optimization across all strategies.
# Requires data generated with 5 LPs (run `just generate-roomeq-data` first).
# ----------------------------------------------------------------------

[group('qa-roomeq-multi')]
qa-roomeq-multi-measurement: \
	qa-roomeq-multi-small-stereo-20 \
	qa-roomeq-multi-small-stereo-21 \
	qa-roomeq-multi-small-stereo-22-mso

[group('qa-roomeq-multi')]
qa-roomeq-multi-small-stereo-20: ensure-venv
	@for strategy in minimax weighted_sum variance_penalized; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_0; \
	    echo "=== Multi-measurement $strategy ($algo) small_stereo_2_0 ==="; \
	    cargo run --bin roomeq --release -- \
	      --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_0/config.json \
	      --override-config ./data_tests/roomeq/generate/optimiser-config/multi_measurement/$strategy.json \
	      --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_iir_multi_$strategy.json; \
	    ./venv/bin/python3 ./scripts/display-roomeq.py \
	                     ./data_generated/roomeq/generated/$algo/small_stereo_2_0/dsp_iir_multi_$strategy.json; \
	  done \
	done

[group('qa-roomeq-multi')]
qa-roomeq-multi-small-stereo-21: ensure-venv
	@for strategy in minimax weighted_sum variance_penalized; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_1; \
	    echo "=== Multi-measurement $strategy ($algo) small_stereo_2_1 ==="; \
	    cargo run --bin roomeq --release -- \
	      --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_1/config.json \
	      --override-config ./data_tests/roomeq/generate/optimiser-config/multi_measurement/$strategy.json \
	      --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_iir_multi_$strategy.json; \
	    ./venv/bin/python3 ./scripts/display-roomeq.py \
	                     ./data_generated/roomeq/generated/$algo/small_stereo_2_1/dsp_iir_multi_$strategy.json; \
	  done \
	done

[group('qa-roomeq-multi')]
qa-roomeq-multi-small-stereo-22-mso: ensure-venv
	@for strategy in minimax weighted_sum variance_penalized; do \
	  for algo in fem; do \
	    mkdir -p ./data_generated/roomeq/generated/$algo/small_stereo_2_2_mso; \
	    echo "=== Multi-measurement $strategy ($algo) small_stereo_2_2_mso ==="; \
	    cargo run --bin roomeq --release -- \
	      --config       ./data_tests/roomeq/generate/$algo/small_stereo_2_2_mso/config.json \
	      --override-config ./data_tests/roomeq/generate/optimiser-config/multi_measurement/$strategy.json \
	      --output       ./data_generated/roomeq/generated/$algo/small_stereo_2_2_mso/dsp_iir_multi_$strategy.json; \
	    ./venv/bin/python3 ./scripts/display-roomeq.py \
	                     ./data_generated/roomeq/generated/$algo/small_stereo_2_2_mso/dsp_iir_multi_$strategy.json; \
	  done \
	done

# New comprehensive QA using roomeq-qa-full binary
[group('qa-roomeq')]
qa-roomeq-coverage: prod-autoeq
	cargo run --bin roomeq-qa-coverage --release

# Hard unit-test line-coverage gate for the library.  The full 897-test suite
# is slow under LLVM instrumentation, so `--release` is used.  Once the crate
# reaches 90 % line coverage this target becomes the canonical gate.
[group('qa-roomeq')]
qa-roomeq-coverage-gate:
	cargo llvm-cov --lib --summary-only --release --fail-under-lines 90

[group('qa-roomeq')]
qa-roomeq-quick: prod-autoeq
	cargo run --bin roomeq-qa-coverage --release -- --quick --maxeval 200

[group('qa-roomeq')]
qa-roomeq-list:
	cargo run --bin roomeq-qa-coverage --release -- --list

[group('qa-roomeq')]
qa-roomeq-matrix:
	cargo run --bin roomeq-qa-coverage --release -- --matrix

[group('qa-roomeq')]
qa-roomeq-synthetic:
	cargo run --bin roomeq-qa-synthetic --no-default-features --release

[group('qa-roomeq')]
qa-roomeq-multiseat-guards:
	cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

[group('qa-roomeq')]
qa-roomeq-home-cinema:
	cargo test -p autoeq home_cinema --lib -- --nocapture
	cargo test -p autoeq validate_bass_management --lib -- --nocapture
	cargo test -p autoeq derives_all_channel_multiseat_primary_weights --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_on_grid_mismatch --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_on_invalid_weight_policy --lib -- --nocapture
	cargo test -p autoeq reports_grid_mismatch_as_channel_skip --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_when_primary_seat_is_invalid --lib -- --nocapture
	cargo test -p autoeq rejects_all_channel_multiseat_when_constraints_fail --lib -- --nocapture
	cargo test -p autoeq rejects_all_channel_multiseat_when_broadband_level_collapses --lib -- --nocapture
	cargo test -p autoeq reports_guardrail_rejection_without_claiming_applied --lib -- --nocapture
	cargo test -p autoeq home_cinema_all_channel_multiseat_guardrail_reruns_and_reports_rejection --lib -- --nocapture
	cargo test -p autoeq reports_all_channel_multiseat_null_guard --lib -- --nocapture
	cargo test -p autoeq reports_all_channel_multiseat_by_role_group_and_excludes_subs --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-all-channel-multiseat:
	cargo test -p autoeq derives_all_channel_multiseat_primary_weights --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_on_grid_mismatch --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_on_invalid_weight_policy --lib -- --nocapture
	cargo test -p autoeq reports_grid_mismatch_as_channel_skip --lib -- --nocapture
	cargo test -p autoeq skips_all_channel_multiseat_when_primary_seat_is_invalid --lib -- --nocapture
	cargo test -p autoeq rejects_all_channel_multiseat_when_constraints_fail --lib -- --nocapture
	cargo test -p autoeq rejects_all_channel_multiseat_when_broadband_level_collapses --lib -- --nocapture
	cargo test -p autoeq reports_guardrail_rejection_without_claiming_applied --lib -- --nocapture
	cargo test -p autoeq home_cinema_all_channel_multiseat_guardrail_reruns_and_reports_rejection --lib -- --nocapture
	cargo test -p autoeq reports_all_channel_multiseat_null_guard --lib -- --nocapture
	cargo test -p autoeq reports_all_channel_multiseat_by_role_group_and_excludes_subs --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-bass-management:
	cargo test -p autoeq home_cinema_bass_management --lib -- --nocapture
	cargo test -p autoeq bass_headroom_uses_supplied_sample_rate_for_crossover_response --lib -- --nocapture
	cargo test -p autoeq home_cinema_bass_management_sub_curve_is_predicted_from_routes --lib -- --nocapture
	cargo test -p autoeq home_cinema_bass_bus_curve_is_predicted_across_multiple_sub_outputs --lib -- --nocapture
	cargo test -p autoeq representative_bass_route_signature_uses_emitted_route_shape --lib -- --nocapture
	cargo test -p autoeq home_cinema_bass_management_workflow_applies_configured_group_crossovers_when_optimization_disabled --lib -- --nocapture
	cargo test -p autoeq validate_bass_management --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-dsp-consistency:
	cargo test -p autoeq reported_ --lib -- --nocapture
	cargo test -p autoeq timing_diagnostics --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-phase-critical:
	cargo test -p autoeq gd_opt --lib -- --nocapture
	cargo test -p autoeq phase_linear_gd_target --lib -- --nocapture
	cargo test -p autoeq frequency_grid --lib -- --nocapture
	cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

[group('qa-roomeq')]
qa-roomeq-perceptual:
	# nextest's explicit no-test failure prevents stale filters from producing a false-green QA run.
	cargo nextest run -p autoeq --lib --no-tests fail -E 'test(/loss::epa::/)'
	cargo nextest run -p autoeq --lib --no-tests fail -E 'test(/generate_validation_bundle_report_creates_json|correction_report_(rejected_guardrails|failed_constraints_and_null_advisory)/)'
	cargo nextest run -p autoeq --lib --no-tests fail -E 'test(/home_cinema_(no_sub|with_sub)_multiseat_rejection_reports|run_channel_via_generic_path_multiseat_rejected_recovers|all_channel_multiseat_acceptance_rejects_subs/)'
	cargo nextest run -p autoeq --bin roomeq-qa-quality --no-tests fail -E 'test(/scorecard_/)'

[group('qa-roomeq')]
qa-roomeq-gd:
	cargo test -p autoeq gd_opt -- --nocapture
	cargo test -p autoeq phase_linear_gd_target --lib -- --nocapture

[group('qa-roomeq')]
qa-roomeq-features:
	cargo run --bin roomeq-qa-features --no-default-features --release

# Supporting-source room compensation QA: integration test + targeted lib tests.
[group('qa-roomeq')]
qa-roomeq-supporting-source:
	cargo test -p autoeq --test roomeq_supporting_source -- --nocapture
	cargo test -p autoeq supporting_source --lib -- --nocapture

# Compact CI-friendly QA run: bounded fuzzer + small coverage subset.
# Typical wall time under 3 minutes on modern hardware.
[group('qa-roomeq')]
qa-roomeq-ci:
	cargo run --bin roomeq-fuzzer --release  --features="plotly" -- -n 50 --seed 42 --skip-kautz-modal
	cargo run --bin roomeq-qa-coverage --release -- --quick --maxeval 200
	cargo test -p autoeq reported_ --lib -- --nocapture
	cargo test -p autoeq timing_diagnostics --lib -- --nocapture
	cargo test -p autoeq gd_opt --lib -- --nocapture
	cargo test -p autoeq phase_linear_gd_target --lib -- --nocapture
	cargo test -p autoeq frequency_grid --lib -- --nocapture
	cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only
	cargo test -p autoeq home_cinema --lib -- --nocapture
	cargo test -p autoeq home_cinema_bass_management --lib -- --nocapture
	cargo test -p autoeq validate_bass_management --lib -- --nocapture
	just qa-roomeq-perceptual
