#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <config.txt> <input.wav> <output.wav>" >&2
	exit 2
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "The UTM Equalizer APO bridge must run on macOS." >&2
	exit 1
fi

config=$1
input=$2
output=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
benchmark_script="$script_dir/run-equalizer-apo-benchmark.ps1"
vm=${ROOMEQ_EQUALIZER_APO_UTM_VM:-Win11 ARM AutoEQ}
benchmark=${ROOMEQ_EQUALIZER_APO_BENCHMARK:-C:\\Program Files\\EqualizerAPO\\Benchmark.exe}
wait_seconds=${ROOMEQ_EQUALIZER_APO_UTM_WAIT_SECONDS:-120}
keep_running=${ROOMEQ_EQUALIZER_APO_UTM_KEEP_RUNNING:-0}

for file in "$config" "$input" "$benchmark_script"; do
	if [[ ! -f "$file" ]]; then
		echo "Required Equalizer APO QA file not found: $file" >&2
		exit 1
	fi
done

if [[ -n "${ROOMEQ_EQUALIZER_APO_UTMCTL:-}" ]]; then
	utmctl=$ROOMEQ_EQUALIZER_APO_UTMCTL
elif command -v utmctl >/dev/null 2>&1; then
	utmctl=$(command -v utmctl)
elif [[ -x /Applications/UTM.app/Contents/MacOS/utmctl ]]; then
	utmctl=/Applications/UTM.app/Contents/MacOS/utmctl
else
	echo "utmctl not found. Install UTM or set ROOMEQ_EQUALIZER_APO_UTMCTL." >&2
	exit 1
fi

if ! [[ "$wait_seconds" =~ ^[0-9]+$ ]] || [[ "$wait_seconds" -eq 0 ]]; then
	echo "ROOMEQ_EQUALIZER_APO_UTM_WAIT_SECONDS must be a positive integer." >&2
	exit 1
fi

vm_status=$("$utmctl" status "$vm" 2>/dev/null) || {
	echo "UTM VM not found: $vm" >&2
	echo "Set ROOMEQ_EQUALIZER_APO_UTM_VM to its complete name or UUID." >&2
	exit 1
}

started_vm=0
guest_ready=0
job_id="roomeq-apo-qa-$$"
guest_dir="C:\\Windows\\Temp\\$job_id"
guest_config="$guest_dir\\config.txt"
guest_input="$guest_dir\\input.wav"
guest_output="$guest_dir\\output.wav"
guest_script="$guest_dir\\run-equalizer-apo-benchmark.ps1"
partial_output="$output.utm-$$"

cleanup() {
	status=$?
	trap - EXIT INT TERM
	set +e
	rm -f "$partial_output"
	if [[ "$guest_ready" -eq 1 ]]; then
		"$utmctl" exec --hide "$vm" --cmd powershell.exe -NoProfile -NonInteractive \
			-Command "Remove-Item -LiteralPath '$guest_dir' -Recurse -Force -ErrorAction SilentlyContinue" \
			>/dev/null 2>&1
	fi
	if [[ "$started_vm" -eq 1 && "$keep_running" != 1 ]]; then
		"$utmctl" stop --hide "$vm" >/dev/null 2>&1
	fi
	exit "$status"
}
trap cleanup EXIT INT TERM

case "$vm_status" in
	started)
		;;
	stopped|suspended|paused)
		echo "Starting UTM VM '$vm'..."
		"$utmctl" start --hide "$vm" >/dev/null
		started_vm=1
		;;
	*)
		echo "UTM VM '$vm' is in unsupported state: $vm_status" >&2
		exit 1
		;;
esac

echo "Waiting for the UTM Windows guest agent..."
deadline=$((SECONDS + wait_seconds))
while (( SECONDS < deadline )); do
	# utmctl can report guest-agent startup errors while still exiting zero, so
	# require a marker emitted by the guest process instead of trusting status.
	probe=$("$utmctl" exec --hide "$vm" --cmd cmd.exe /d /c echo ROOMEQ_UTM_READY 2>&1 || true)
	if [[ "$probe" == *ROOMEQ_UTM_READY* ]]; then
		guest_ready=1
		break
	fi
	sleep 2
done
if [[ "$guest_ready" -ne 1 ]]; then
	echo "UTM guest agent did not become ready within ${wait_seconds}s." >&2
	echo "Install the UTM Windows guest tools and confirm the VM can log in." >&2
	exit 1
fi

"$utmctl" exec --hide "$vm" --cmd powershell.exe -NoProfile -NonInteractive \
	-Command "New-Item -ItemType Directory -Force -Path '$guest_dir' | Out-Null"
"$utmctl" file push "$vm" "$guest_config" < "$config"
"$utmctl" file push "$vm" "$guest_input" < "$input"
"$utmctl" file push "$vm" "$guest_script" < "$benchmark_script"

echo "Running Equalizer APO Benchmark.exe in UTM VM '$vm'..."
"$utmctl" exec --hide "$vm" --cmd powershell.exe -NoProfile -NonInteractive \
	-ExecutionPolicy Bypass -File "$guest_script" \
	-Benchmark "$benchmark" -Config "$guest_config" \
	-InputFile "$guest_input" -OutputFile "$guest_output"
"$utmctl" file pull "$vm" "$guest_output" > "$partial_output"
if [[ ! -s "$partial_output" || "$(head -c 4 "$partial_output")" != RIFF ]]; then
	echo "UTM did not return a valid WAV from Equalizer APO." >&2
	exit 1
fi
mv "$partial_output" "$output"
