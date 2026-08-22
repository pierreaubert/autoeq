# --------------------------------------------------------- -*- just -*-
# QA ROOMEQ EXPORT
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
	@echo "  cargo test --release -p roomeq-export"
	@echo "  just qa-export-camilladsp  # requires the real CamillaDSP binary"
	@echo "  just qa-export-pipewire    # runs PipeWire validation in Docker"
	@echo "  just qa-export-portable    # semantic APO/EasyEffects/Wavelet/Roon checks"
	@echo "  just qa-export-all         # portable + CamillaDSP + PipeWire matrix"
	@echo "  just qa-export-equalizer-apo  # native Windows, or a Windows UTM VM on macOS"
	@echo "  just qa-export-roon-setup  # one-time private macOS Roon/BlackHole setup"
	@echo "  just qa-export-roon-app    # licensed interactive macOS Roon smoke test"
	@echo
	@echo "Install helper:"
	@echo "  just install-export-test-systems"
	@echo
	@echo "Optional validator environment variables:"
	@echo "  ROOMEQ_CAMILLADSP_VALIDATE_CMD"
	@echo "  ROOMEQ_EQUALIZER_APO_VALIDATE_CMD"
	@echo "  ROOMEQ_EQUALIZER_APO_UTM_VM"
	@echo "  ROOMEQ_EASYEFFECTS_VALIDATE_CMD"
	@echo "  ROOMEQ_WAVELET_VALIDATE_CMD"
	@echo "  ROOMEQ_PIPEWIRE_VALIDATE_CMD"
	@echo "  ROOMEQ_ROON_VALIDATE_CMD"
	@echo
	@echo "Each command may use {config} or {file} as the generated export path."
	@echo "If no placeholder is present, the path is appended as the final argument."
	@echo
	@echo "Examples:"
	@echo "  ROOMEQ_CAMILLADSP_VALIDATE_CMD='camilladsp --check {config}' cargo test --release -p roomeq-export tool_contract_camilladsp"
	@echo "  ROOMEQ_PIPEWIRE_VALIDATE_CMD='path/to/validate-pipewire {config}' cargo test --release -p roomeq-export tool_contract_pipewire"
	@echo "  ROOMEQ_EASYEFFECTS_VALIDATE_CMD='path/to/validate-easyeffects {config}' cargo test --release -p roomeq-export tool_contract_easyeffects"
	@echo
	@echo "Manual/application-backed validators:"
	@echo "  Equalizer APO: install in Windows; macOS defaults to the UTM VM named 'Win11 ARM AutoEQ'."
	@echo "  Set ROOMEQ_EQUALIZER_APO_UTM_VM when the VM has another complete name or UUID."
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
	echo "  3. Run: cargo test --release -p roomeq-export"

[group('qa-export')]
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
		cargo nextest run --release -p roomeq-export --lib --no-tests fail -E 'test(/camilladsp/)'

[group('qa-export')]
qa-export-portable:
	#!/usr/bin/env bash
	set -euo pipefail
	# Roon and Wavelet have no supported headless importer. These tests parse each
	# generated artifact independently and reconstruct its frequency response;
	# optional application validators remain available through ROOMEQ_*_VALIDATE_CMD.
	cargo nextest run --release -p roomeq-export --lib --no-tests fail \
		-E 'test(/equalizer_apo|easyeffects|wavelet|roon/)'

[group('qa-export')]
qa-export-roon-setup:
	#!/usr/bin/env bash
	set -euo pipefail
	scripts/roon-qa/setup.sh

[group('qa-export')]
qa-export-roon-app:
	#!/usr/bin/env bash
	set -euo pipefail
	scripts/roon-qa/run-app-qa.sh

[group('qa-export')]
qa-export-all: qa-export-portable qa-export-camilladsp qa-export-pipewire
	#!/usr/bin/env bash
	set -euo pipefail
	case "$(uname -s)" in
		Darwin|MINGW*|MSYS*|CYGWIN*) just qa-export-equalizer-apo ;;
		*) echo "Equalizer APO engine QA requires Windows; portable APO semantics passed." ;;
	esac

[group('qa-export')]
qa-export-equalizer-apo:
	#!/usr/bin/env bash
	set -euo pipefail
	case "$(uname -s)" in
		Darwin)
			runner="$(pwd)/scripts/run-equalizer-apo-utm.sh"
			ROOMEQ_EQUALIZER_APO_PCM_CMD="bash '$runner' '{config}' '{input}' '{output}'" \
				cargo nextest run --release -p roomeq-export --lib --no-tests fail -E 'test(/equalizer_apo/)'
			;;
		MINGW*|MSYS*|CYGWIN*)
			benchmark="${ROOMEQ_EQUALIZER_APO_BENCHMARK:-C:\\Program Files\\EqualizerAPO\\Benchmark.exe}"
			if [[ ! -f "$benchmark" ]]; then
				echo "Equalizer APO Benchmark.exe not found: $benchmark" >&2
				echo "Install Equalizer APO on Windows or set ROOMEQ_EQUALIZER_APO_BENCHMARK." >&2
				exit 1
			fi
			ROOMEQ_EQUALIZER_APO_PCM_CMD="powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-equalizer-apo-benchmark.ps1 -Benchmark '$benchmark' -Config '{config}' -InputFile '{input}' -OutputFile '{output}'" \
				cargo nextest run --release -p roomeq-export --lib --no-tests fail -E 'test(/equalizer_apo/)'
			;;
		*)
			echo "Equalizer APO engine QA requires Windows or macOS with a Windows UTM VM." >&2
			exit 1
			;;
	esac

[group('qa-export')]
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

# Backend semantic/PCM validation. External backends are mandatory here so a
# release cannot silently skip correctness checks.
[group('qa-export')]
qa-export-equivalence: qa-export-all
