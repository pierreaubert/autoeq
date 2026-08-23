# RoomEQ Input Format

RoomEQ consumes a JSON configuration file describing the room, speakers,
measurements, and optimizer settings. The top-level object is a `RoomConfig`.

## Measurement provenance

The optional top-level `provenance` object links each speaker measurement to a
versioned `.provenance.json` sidecar without embedding private acquisition data
in the RoomEQ config. `validation_mode` defaults to `"warn"` for legacy
configurations. RoomEQ currently records these references in the input/output
contract but does not validate sidecars at CLI runtime; do not rely on this
field as an integrity or SHA-256 enforcement mechanism.

```json
{
  "provenance": {
    "validation_mode": "strict",
    "measurements": {
      "L": {
        "record_id": "api:4c8d...",
        "content_hash": "4c8d7a2bfefb8e5b8e8f95f7b10f7c6dd5ac5aa938aeac181c35d7f94b34d0e2",
        "schema_version": 1,
        "sidecar_path": "measurements/left.csv.provenance.json"
      }
    }
  }
}
```

Upgrade a supported legacy sidecar without re-importing its measurement:

```bash
migrate_provenance measurements/left.csv.provenance.json
```

The command writes the current schema deterministically and creates a
`.bak` file for an in-place migration. Pass an explicit second path to retain
the original sidecar unchanged.

## Inter-channel timbre matching

Use `optimizer.inter_channel_timbre_matching` to reduce residual broadband
tonal differences after per-channel EQ. Each non-reference channel is compared
with the configured reference over the optimizer frequency range. RoomEQ only
applies the candidate shelf/gain correction when it reduces normalized timbre
spread by at least `min_improvement_db`.

```json
{
  "optimizer": {
    "inter_channel_timbre_matching": {
      "enabled": true,
      "reference_channel": "center",
      "min_improvement_db": 0.05
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enables the post-EQ timbre-matching stage. |
| `reference_channel` | string | required | Logical channel whose tonal balance is the target. |
| `min_improvement_db` | number | `0.05` | Minimum finite, non-negative reduction in normalized timbre spread required before applying DSP. |

The former `optimizer.vog` key is no longer accepted. Replace it directly with
`optimizer.inter_channel_timbre_matching`; its nested fields are unchanged.

## Height-channel alignment

Use `optimizer.height_channel_alignment` to align overhead channels with
role-appropriate bed-channel references. RoomEQ can match timbre, level, and
arrival time independently, and can optionally require trustworthy phase for
the phase-aware safety gate.

```json
{
  "optimizer": {
    "height_channel_alignment": {
      "enabled": true,
      "match_timbre": true,
      "match_level": true,
      "match_arrival_time": true,
      "match_phase": false,
      "min_timbre_improvement_db": 0.05,
      "max_delay_ms": 20.0,
      "reference_channels": {
        "top_front": "front_left",
        "top_middle": "side_left",
        "top_rear": "rear_left"
      }
    }
  }
}
```

At least one of `match_timbre`, `match_level`, or `match_arrival_time` must be
enabled. `max_delay_ms` must be finite and positive. Reference overrides may be
keyed by a height channel name or by `top_front`, `top_middle`, or `top_rear`.

## Configuration schema version

The top-level `version` is validated before paths are resolved or optimization
starts. RoomEQ accepts the historical `1.0.x` through `1.2.x` schema lines and
the `2.0.x` through `2.1.x` lines. The current default is `2.1.0`. Malformed
versions and unknown minor or major versions fail closed instead of being
interpreted with current defaults.

RoomEQ loads canonical configuration and override files with recursive strict
deserialization. Unknown, misspelled, or misplaced fields are errors rather
than ignored comments. This applies inside nested optimizer, FIR, group-delay,
speaker, system, and bass-management objects as well as at the root. Typed map
objects such as `speakers` still accept arbitrary map keys whose values conform
to the declared schema. The generated `input_schema.json` mirrors this contract
with `additionalProperties: false` on fixed-shape objects.

## Neutral objective, spatial risk, and preference layers

The neutral flat/asymmetric objective uses the versioned
`glasberg-moore-erb-rate-1990-v1` integration measure. Discrete ERB-rate cell
widths keep constant physical error invariant across linear, logarithmic,
sparse, and dense grids, and signed residuals are not smoothed before loss.

`optimizer.multi_measurement.weights` must be finite and nonnegative with at
least one positive entry; `variance_lambda` must be finite and nonnegative.
`spatial_robustness` evaluates every seat directly with a variance-penalized
per-seat risk measure and applies its correction-depth mask to each seat. It no
longer optimizes a power-averaged representative curve.

Bootstrap results are labelled `spatial_seat_sampling`. By default they assume
independent positions; when nearby seats are correlated, set
`bootstrap_uncertainty.effective_spatial_sample_size` below the nominal seat
count to draw fewer cases per resample and widen the confidence band. This is a
conservative correlation adjustment, not a spatial covariance model. Separately
measured `repeat_sweep_noise_std_db` and `calibration_uncertainty_std_db` are
reported as distinct nuisance sources rather than estimated by the seat
bootstrap.

The fixed `harman` house curve, `target_response.preference`, and role/content voicing are emitted as a
separately bypassable post-correction layer. They remain in the final DSP chain
but are excluded from neutral quality scores. The output report records
`neutral_target_response`, `preference_layer`, and
`excluded_from_neutral_quality_score` explicitly.

EPA optimization is experimental and uses spectral flatness only. Transfer-only
loudness, roughness, sharpness, and temporal values are diagnostics rather than
validated programme-audio or measured-decay objectives.

## Multi-measurement RIR prototype

When a speaker has several measurements captured at different positions, you
can ask RoomEQ to build a single distance- and directivity-weighted prototype
curve and then optimize that curve instead of each measurement individually.

Enable the prototype by adding a `rir_prototype` block inside the speaker's
`multi_measurement` configuration:

```json
{
  "version": "2.1.0",
  "speakers": {
    "left": {
      "measurements": [
        "measurements/left_pos1.csv",
        "measurements/left_pos2.csv",
        "measurements/left_pos3.csv"
      ]
    }
  },
  "optimizer": {
    "num_filters": 7,
    "multi_measurement": {
      "strategy": "weighted_sum",
      "rir_prototype": {
        "reference_position": [0.0, 0.0, 0.0],
        "source_position": [0.0, 2.5, 0.0],
        "microphone_positions": [
          [0.0, 0.0, 0.0],
          [0.15, 0.0, 0.0],
          [-0.15, 0.0, 0.0]
        ],
        "distance_mode": "inverse_square",
        "directivity": "omnidirectional",
        "frequency_dependent_directivity": false
      }
    }
  }
}
```

### `RirPrototypeConfig` fields

| Field | Type | Description |
|-------|------|-------------|
| `reference_position` | `[f64; 3]` | Optimal listening position, e.g. the center of the listener's head at the main seat. |
| `source_position` | `[f64; 3]` | Position of the main loudspeaker. Defines the forward axis used for directivity calculations. |
| `microphone_positions` | `[[f64; 3]]` | One position per measurement, in the same order as the measurements. |
| `distance_mode` | `DistanceWeightMode` | How distance from `reference_position` to each microphone affects its weight. |
| `directivity` | `DirectivityModel` | Directivity model applied to each microphone relative to the source axis. |
| `frequency_dependent_directivity` | `bool` | If `true`, directivity is evaluated at each frequency bin; otherwise it is evaluated once at 1 kHz. |

### `DistanceWeightMode`

```json
"distance_mode": "uniform"
"distance_mode": "inverse_square"
"distance_mode": { "gaussian": { "sigma_m": 0.3 } }
```

- `uniform` — all microphones weighted equally.
- `inverse_square` — weight is `1 / d²`, clipped at `1e-6` m to avoid infinities.
- `gaussian` — weight is `exp(-d² / (2·sigma²))`; `sigma_m` must be strictly positive.

### `DirectivityModel`

```json
"directivity": "omnidirectional"
"directivity": { "spherical_head": { "radius_m": 0.0875 } }
```

- `omnidirectional` — no directivity correction.
- `spherical_head` — rigid-sphere head-shadow approximation; `radius_m` must be strictly positive.

### Notes

- All measurements must share the same frequency grid (same length and same
  frequency values within tolerance). RoomEQ rejects mismatched grids.
- The prototype is built in the magnitude (SPL) domain. Phase and any other
  metadata from the first measurement are carried over unchanged.
- If `multi_measurement.weights` is supplied, it is ignored when `rir_prototype`
  is enabled, because the prototype builder has already collapsed the
  measurements into a single curve.
- Time-domain / IR averaging is not supported in this iteration.
