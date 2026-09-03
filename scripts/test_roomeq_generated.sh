#!/usr/bin/env bash
set -euo pipefail

IN=${IN:-./data_tests/roomeq/generate}
OUT=${OUT:-./data_generated/roomeq/generated}
LOG=${LOG:-warn}

command -v jq >/dev/null || {
    echo "test_roomeq_generated.sh requires jq to validate fixture JSON" >&2
    exit 1
}

cargo build --release --features cli --bin roomeq

# Validate the complete checked-in fixture set before starting expensive runs.
while IFS= read -r -d '' json; do
    jq empty "$json"
done < <(find "$IN" -type f -name '*.json' -print0 | sort -z)

run_case() {
    local family=$1
    local scenario=$2
    local base_config=$3
    local override=$4
    local stem=$5
    local scenario_out="$OUT/$family/$scenario"
    local output="$scenario_out/dsp-$stem.json"

    mkdir -p "$scenario_out"
    echo "RoomEQ generated: $family/$scenario ($stem)"
    if [[ -n "$override" ]]; then
        RUST_LOG="$LOG" ./target/release/roomeq \
            --config "$base_config" \
            --override-config "$override" \
            --output "$output"
    else
        RUST_LOG="$LOG" ./target/release/roomeq \
            --config "$base_config" \
            --output "$output"
    fi
    if [[ -n "$override" ]]; then
        ./venv/bin/python3 ./scripts/display-roomeq.py \
            "$output" \
            --output "$scenario_out/dsp-$stem.html" \
            --base-config "$base_config"
    else
        ./venv/bin/python3 ./scripts/display-roomeq.py \
            "$output" \
            --output "$scenario_out/dsp-$stem.html"
    fi
}

compare_cases() {
    local family=$1
    local scenario=$2
    shift 2
    (( $# > 1 )) || return 0
    local scenario_out="$OUT/$family/$scenario"
    local files=()
    local stem
    for stem in "$@"; do
        files+=("$scenario_out/dsp-$stem.json")
    done
    ./venv/bin/python3 ./scripts/display-roomeq.py \
        --compare "${files[@]}" \
        --output "$scenario_out/compare.html"
}

# FEM fixtures have a topology-specific override directory. Every override in
# that directory is a test, including configurations with only one supported
# processing mode.
for base_config in "$IN"/fem/*/config.json; do
    [[ -f "$base_config" ]] || continue
    scenario=$(basename "$(dirname "$base_config")")
    override_dir="$IN/optimiser-config/$scenario"
    stems=()
    if [[ ! -d "$override_dir" ]]; then
        # Some generated multi-sub fixtures intentionally carry their complete
        # optimizer configuration in config.json.
        run_case fem "$scenario" "$base_config" "" embedded
        continue
    fi
    while IFS= read -r -d '' override; do
        stem=$(basename "$override" .json)
        run_case fem "$scenario" "$base_config" "$override" "$stem"
        stems+=("$stem")
    done < <(find "$override_dir" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
    (( ${#stems[@]} > 0 )) || {
        echo "No optimizer fixtures found for FEM scenario: $scenario" >&2
        exit 1
    }
    compare_cases fem "$scenario" "${stems[@]}"
done

# The remaining FEM override directories are deliberate cross-cutting tests
# rather than topology names. Run each one against its compatible generated
# fixture so that every checked-in optimizer JSON is exercised.
run_extra_override() {
    local directory=$1
    local scenario=$2
    local base_config="$IN/fem/$scenario/config.json"
    local stems=()
    local override stem
    [[ -f "$base_config" ]] || {
        echo "Missing base fixture for optimizer directory $directory: $scenario" >&2
        exit 1
    }
    while IFS= read -r -d '' override; do
        stem=$(basename "$override" .json)
        run_case fem "$scenario-$directory" "$base_config" "$override" "$stem"
        stems+=("$stem")
    done < <(find "$IN/optimiser-config/$directory" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
    if (( ${#stems[@]} > 0 )); then
        compare_cases fem "$scenario-$directory" "${stems[@]}"
    fi
}

run_extra_override modes small_stereo_2_0
run_extra_override multi_measurement medium_multi_seat
run_extra_override small_stereo_2_2 small_stereo_2_2_mso
run_extra_override small_stereo_2_2_independent small_stereo_2_2_mso

# Home-cinema overrides are named feature/topology canaries. Keep these
# pairings in sync with qa/registry/roomeq.json so each override runs against
# the topology it was designed to exercise.
home_cinema_base() {
    local name=$1
    case "$name" in
        iir_lfe_only) echo "$IN/fem/medium_surround_5_1/config.json" ;;
        iir_redirected_bass|mixed_phase_redirected_bass)
            echo "$IN/fem/medium_surround_5_1_4/config.json"
            ;;
        hybrid_redirected_bass|phase_linear_fir_redirected_bass|height_alignment)
            echo "$IN/fem/large_surround_5_1_4/config.json"
            ;;
        coherence_adaptive_allpass|all_channel_multi_seat_mso)
            echo "$IN/fem/medium_surround_5_2_4_multi_seat/config.json"
            ;;
        *5_1_2*) echo "$IN/fast-hybrid/medium_surround_5_1_2_multi_seat/config.json" ;;
        *7_1_2*) echo "$IN/fast-hybrid/large_surround_7_1_2_multi_seat/config.json" ;;
        *7_1_6*) echo "$IN/fast-hybrid/large_surround_7_1_6_multi_seat/config.json" ;;
        *7_4_4*) echo "$IN/fast-hybrid/large_surround_7_4_4_multi_seat/config.json" ;;
        *9_1_6*) echo "$IN/fast-hybrid/large_surround_9_1_6_multi_seat/config.json" ;;
        *9_8_6*) echo "$IN/fast-hybrid/large_surround_9_8_6_multi_seat/config.json" ;;
        *) return 1 ;;
    esac
}

while IFS= read -r -d '' override; do
    stem=$(basename "$override" .json)
    base_config=$(home_cinema_base "$stem")
    [[ -f "$base_config" ]] || {
        echo "Missing home-cinema base fixture for optimizer JSON: $override" >&2
        exit 1
    }
    scenario=$(basename "$(dirname "$base_config")")
    if [[ "$base_config" == "$IN/fast-hybrid/"* ]]; then
        family=fast-hybrid
    else
        family=fem
    fi
    run_case "$family" "$scenario" "$base_config" "$override" "$stem"
done < <(find "$IN/optimiser-config/home_cinema" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)

echo "Generated RoomEQ QA completed successfully"
