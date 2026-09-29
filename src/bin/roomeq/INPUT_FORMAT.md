# RoomEQ Input Format

## Recording configuration and room dimensions

The optional root `recording_config` object describes the room and capture
session. Its `room_dimensions` object requires finite positive `length`, `width`,
and `height` in meters, with finite positive volume. Example:

```json
{
  "recording_config": {
    "room_dimensions": { "length": 5.0, "width": 4.0, "height": 2.5 },
    "setup_description": "Main listening position",
    "recording_sample_rate": 48000,
    "channel_names": ["L", "R"]
  }
}
```

Schroeder split and decomposed correction both use these dimensions. Move legacy
`optimizer.schroeder_split.room_dimensions` and
`optimizer.decomposed_correction.room_dimensions` here; nested copies are rejected.
Supplying dimensions does not enable those modes. Without dimensions, existing
Schroeder frequency fallbacks remain unchanged. See the
[full recording configuration reference](../../../docs/ROOMEQ_INPUT_FORMAT.md#recording-configuration-recording_config)
for device, signal, calibration and timing evidence fields.

Report consumers evaluate serialized `kautz_filter` and `warped_biquad`
topologies rather than treating them as PEQs. Kautz section `gain` (and the
legacy single-section `db_gain`) is a **linear weight**; zero weights retain
their ordered allpass stages. Python channel EQ plots use the output sample
rate and reject malformed advanced-filter declarations.

Comparison-report fallback curves likewise use each output's `sample_rate`;
an absent rate retains the legacy 48 kHz default. A saved `eq_response` is
displayed without reconstruction. Summary counts include every Kautz section,
even when its weight is zero, and driver-local EQ sections; shared channel
entries count once and descriptive `route_owned` entries are excluded.

Parallel-driver waveform replay requires phase on every driver's initial capture,
with the same length as its finite, ordered frequency/SPL arrays. Missing or
invalid phase causes an explicit replay error rather than a zero-phase sum.
Phase presence is not a substitute for common capture timing provenance.
Parallel waveform reporting additionally requires a configured group, explicit
topology, multi-sub, DBA, or cardioid source mapping covering each driver, with matching
stationary timing/seat labels and valid support across the displayed measured
band. Missing or unsupported mappings withhold the summed prediction and record
a diagnostic. This report gate does not authenticate raw recordings.
Both acoustic traces are withheld when timing admission fails, since the group
reference may be synthesized; independent FIR-kernel diagnostics remain available.
System main-role and sub-output IDs are resolved to their configured sources;
conflicting aliases are unavailable. DBA checks every source even though the
delivered chain contains only front/rear aggregate branches. Cardioid front/rear
branch IDs and indices must agree with the production mapping.
The same source/timing admission is required before joint `per_driver` FIR
generation, over the target grid's full support. Failure rejects the requested
design before its artifact writes or chain mutation; no silent topology fallback
is installed. Existing configs with phase arrays but no declared shared timing
must supply appropriate acquisition evidence to use this coherent design.
Blank `timing_reference_id` values and the literal `unknown` (case-insensitive,
ignoring surrounding whitespace) are treated as absent by measurement admission.
The original declaration remains available for serialization and diagnostics.

Output `CurveData` preserves optional capture `noise_floor_db` and `coherence`
arrays through serialization and reconstruction. Each follows the frequency
grid; the noise floor uses the same level reference as SPL. Missing fields in
legacy output remain absent. This is output evidence transport, not a new
inline-measurement input option or independent phase-correction authorization.

For joint `per_driver` design, `mixed_phase` and Kirkeby with
`correct_excess_phase: true` additionally require a supported numerical
excess-phase assessment on every driver. The controls are
`optimizer.mixed_phase.assessment`, including its default SNR, coverage, taper,
and narrow/wide smoothing-consistency bounds. This assessed path uses those
smoothing widths rather than the legacy `fir.phase_smoothing` or
`mixed_phase.phase_smoothing_octaves` decomposition controls. An unknown or
unsupported assessment rejects the explicit request before sidecar writes;
missing noise-floor data is not replaced by a guessed SNR. Magnitude-only
requests are assessed separately and are not implicitly substituted.

For `per_driver` placement only, set
`optimizer.mixed_phase.assessment.retain_magnitude_on_refusal: true` to retain
the existing magnitude-processing chain after a numerical phase-evidence or
direct-sound target-policy refusal. The default is `false` (abort). This opt-in
does not design a replacement magnitude FIR: it records an insufficient-evidence
phase decision, writes no phase sidecars, and subjects the retained graph to
the normal final safety checks. Shared-FIR behavior is unchanged. Missing
coherent capture phase/timing, malformed inputs/policy, and operational failures
still abort. The requested target and source identities remain unchanged.

The same joint excess-phase path checks source-bound direct-sound target policy
over the active correction band. Above the target transition, every contributing
capture must provide assessed direct-sound/angular support for that band,
including every cabinet contributing to a DBA aggregate. The configured
Schroeder transition (or the target model's documented 300 Hz fallback) applies.
Missing support refuses the request before sidecar writes without replacing the
target. These checks do not authenticate capture declarations.

Successful joint designs retain source/target decisions for final-ledger
reconciliation, with selected phase strength and branch-specific FIR references.
Zero phase selection is unresolved, not an applied phase correction. Requested
bands are reported as design bounds, not exact finite-FIR support. Missing or
reassigned branch FIRs invalidate the original applied claim. Admission errors
still abort the request without producing a successful output ledger.

Parallel-driver waveform pairs share their uncorrected peak reference, so
post-correction gains remain visible. This changes report reconstruction only;
serialized playback filters and their gain units are unchanged.

Non-FIR parallel reports also replay individual branch captures and serialized
common processing. Missing capture/phase evidence withholds the predicted pair
and logs a reason; no combined-curve PEQ-summary fallback is used.

Output `metadata.stage_outcomes` records current waveform availability in the
`waveform_views` stage, using check IDs `pre_ir:<channel>` and `post_ir:<channel>`.
Failed checks carry a diagnostic reason; successful checks indicate a saved
prediction only, not acoustic acceptance. Refresh replaces prior stage records.
The Python main and comparison reports display these reasons; legacy outputs
without this stage omit the panel. No input configuration fields are added.
When present, its checks must cover both views for each delivered channel exactly
once. Malformed or incomplete inventories render an unavailable explanation.

Playback-verification plant inputs are a separate CLI handoff, not `RoomConfig`.
Use `--verification-prediction-inputs <JSON>` with `--verification-bundle <DIR>`;
`--verification-graph <JSON>` generates from saved native DSP without optimization.
The strict `physical-ir-prediction-v1` handoff and capture-plane/calibration/timing
requirements are documented in the manual's capture-verification section. It
requires explicit physical-output-by-seat unit-transfer IRs and never treats
ordinary room-response plots as raw recordings. Bundle creation is not a playback
or safety approval and refuses an existing output file.

RoomEQ consumes a JSON configuration file describing the room, speakers,
measurements, and optimizer settings. The top-level object is a `RoomConfig`.

`optimizer.high_frequency_correction.max_q` limits Q locally in the guarded
frequency region through optimizer bounds and returned-candidate checks. It no
longer replaces `optimizer.max_q` globally when defaults or policy overrides are
applied. Bass retains its configured global cap; a stricter global cap still
wins. These are engineering constraints, not audibility thresholds or evidence
of valid direct-sound capture.

## Processing mode names

The preset names FIR, MIXED, and MIXED-PHASE are not interchangeable:

| Preset name | JSON `optimizer.processing_mode` | Correction structure |
|---|---|---|
| IIR | `low_latency` (alias `iir`) | IIR magnitude EQ. |
| FIR | `phase_linear` (alias `fir`) | FIR correction; `fir.phase` selects linear, minimum-phase, or Kirkeby design. The mode name alone does not guarantee linear phase. |
| MIXED | `hybrid` (alias `mixed`) | With `mixed_config`: frequency-split FIR/IIR. Without it: IIR followed by residual FIR magnitude correction. |
| MIXED-PHASE | `mixed_phase` | IIR magnitude EQ plus a short excess-phase FIR; may fall back to IIR-only if phase is unavailable or the phase FIR is rejected. |

For frequency-split hybrid, `mixed_config.fir_band: "low"` assigns the lower band
to FIR and the upper band to IIR; `"high"` reverses them. Its `crossover_freq`
must lie strictly inside the usable correction range so both bands exist.
A 300 Hz split is not valid for a 40–200 Hz correction range; 100 Hz is an
example of an in-range split. Do not extend correction bandwidth just to make
an old split setting pass validation.

This processing split is **not** the speaker/subwoofer crossover configured in
`crossovers` and referenced by bass management. It is also not the full speaker
passband used to assess corrected playback.

Use `fir.taps` for FIR/hybrid FIR length and
`mixed_phase.max_fir_length_ms` for the MIXED-PHASE excess-phase stage.
`fir.phase: "kirkeby"` with `correct_excess_phase: true` requires measured phase;
this differs from the MIXED-PHASE path's IIR-only fallback. Length limits are not
guarantees of total playback latency. See the manual's
[mode comparison and examples](../../../docs/ROOMEQ_MANUAL.md#fir-mixed-and-mixed-phase-which-mode-should-i-use)
for phase behaviour, latency, and choosing a mode.

Standalone phase correction (`phase_correction`) runs only on
assessment-backed bands: the excess-phase assessment in
`phase_correction.assessment` must support each gate-authorized band on
measured phase and noise-floor SNR evidence, and the target chain must
not limit the proposed detail band (room-curve-only upper-band detail
is refused; in-situ bass detail may proceed). Refusal keeps the
independently supported magnitude correction. `assessment` defaults
mirror the analysis-validated evidence bar and stay user-overridable;
`max_correction_latency_ms`, when set, refuses the action if the
generated FIR's causal centering delay exceeds the budget. This is not
the acoustic propagation estimate. Reports retain `estimated_delay_ms`
for propagation and separately expose `causal_center_delay_ms` for the
standalone phase FIR. Other DSP stages and backend buffering are separate.

Permission records must cite nonempty evidence reference IDs and match the
channel's measurement identity. Malformed or mismatched phase/direct-sound
records do not authorize correction. Structural checks are not authentication
of the underlying capture; independently supported magnitude processing remains
available when phase is refused.

`kautz_modal` remains experimental. Its in-repository deterministic fixed-pole
search fits the actual magnitude of complex basis functions **plus a unity dry
path**, rather than using the pinned library's incompatible dB approximation.
Section `gain` is a linear
basis coefficient, not a PEQ dB gain. The legacy single-section `db_gain` field
is also interpreted as that coefficient by the playback consumer. Serialized
evaluation preserves the dry path, canonical/legacy section fields, and zero-gain
defaults; malformed values are rejected. The search uses log-frequency-weighted
squared dB error, prepared targets, and at most `max_iter` coordinate sweeps.
It checks composite `min_db`/`max_db` on measurement, digital-band, and dense
pole-neighborhood samples; those bounds must contain unity. Outside correction
support, measured bins are compared to unity correction rather than flattened.
An already-met target can return a zero-weight bank. This is not a certified
continuous-frequency bound or a guarantee of zero out-of-band response.
Engine/workflow PEQ caches no longer contain synthetic Kautz biquads. An empty
PEQ cache does not mean the Kautz bank is absent: use serialized `kautz_sections`
for linear weights, pole parameters, and section counts. Zero-weight sections
still count because their allpass stage can affect subsequent basis functions.
Final correction-strength trials scale the linear bank weights, so strength
`s` realizes `H_s = 1 + s * (H_full - 1)`. Zero means unity and one retains the
full bank; intermediate strength is not a fraction of the magnitude in dB.
Poles, Q, and allpass ordering remain unchanged. Legacy single-section weights
and section aliases follow the same rule; malformed weights reject the trial.
The resulting complete graph still requires the ordinary safety/seat replay.
Phase-supported serial-channel waveform views also use the complete serialized
transfer (including gain and delay), not a PEQ approximation of Kautz weights.
They retain a common pre-correction peak reference and are model-derived
65,536-point finite-period reconstructions cropped to 400 ms, not raw captures
or proof of long-response linear-convolution support. Failed realization or
missing phase withholds the waveform pair.
The unchanged multirate gain-budget canary now passes, but its unshifted analytic
modal RMS remains slightly worse; correction quality is not established.
Pole selection honors `num_filters`, the
observation band narrowed by `correction_band` when supplied, Nyquist, and
`min_q`/`max_q` (the playback basis supports Q >= 0.1). When oversubscribed,
the strongest eligible peaks are retained with deterministic frequency ordering.
An empty eligible inventory or invalid bounds fails explicitly. Small-weight
sections remain in the chain because removing them changes later basis functions.
Pole-band limits do not guarantee zero response outside that band or compliance
with realized gain limits. See the [review evidence log](../../../docs/ROOMEQ_REVIEW_20260916.md).

## Report policy

The optional top-level `reporting` object declares report policies that need
an explicit operator statement. It changes no acceptance math; absent policies
leave the corresponding viewer summary cells pending.

- `reporting.t60_flatness_tolerance_s`: declared ±tolerance in seconds for
  the report Section 1 "T60 flatness in window" share (nine measured octave
  fits within tolerance of the complete-channel room mean). A present value
  must be finite and positive, otherwise structural validation fails. When
  absent, the viewer applies the ITU-R BS.1116-2 §8.2.3.1 Fig. 1 midband
  default of ±0.05 s (uniformly, as a documented simplification) and labels
  the cell accordingly. The declared value is carried into output
  `metadata.t60_flatness_tolerance_s`; the cell still needs nine valid
  measured fits before rendering.

## Final multi-position validation

Bass-managed crossover alignment uses the configured primary seat's measured
complex response for both mains and a single subwoofer, while retaining all
subwoofer seats for spatial magnitude EQ. Use identical seat ordering and a
shared timing reference across speakers. Spatial magnitude averages are not
valid complex transfers.

Both main/sub alignment searches and per-source coherent route optimization
require matching, nonempty `provenance.timing_reference_id` values with
`capture_kind` set to `stationary_ir` or `direct_sound`. This includes every
capture in a grouped physical sub output. Declared valid bands must cover the
crossover overlap. Missing, mismatched, or spatial-magnitude references skip
these phase-sensitive actions with explicit advisories; phase arrays or a
coherent-looking response alone cannot authorize them.

For a multi-sub group, explicit `joint_optimization: true` takes precedence
over legacy `optimizer.multi_seat` processing. Its capture-reference and phase
requirements still apply; legacy settings cannot bypass those refusals.
Missing seat matrices or timing references return a configuration error;
the selected joint mode never invokes the detailed optimizer as a fallback.
Measurement-backed `allpass_optimization` also requires labeled stationary
captures with a shared timing reference across sources at each seat. Unknown,
mismatched, or incomplete reference matrices are refused before dispatch.
Curve-only numerical APIs do not establish acquisition provenance.
In routed systems, select subwoofer strategy `mso`; combining explicit joint
mode with the independent `single` strategy is rejected instead of silently
running independent processing.

Joint output includes optional `channels.<channel>.joint_sub` diagnostics:
array gains/delays, objective components, physical driver IDs, gain-stage
applications, and every seat before control, after array control, and after
shared EQ. Levels retain their measurement reference and are not automatically
calibrated SPL. The `output_drive_penalty` is an optimizer output-loss proxy,
not an amplifier/driver safety assessment. `channel_processing_matches` is
recomputed on output conversion and packaging; false marks historical stage
predictions after later processing changes. True checks channel controls only,
not global routing, external resource bytes, hardware playback, or preference.

Raw captures are checked before phase-sensitive bass processing over the
configured crossover overlap (half the lowest to twice the highest candidate
frequency, including group/sub overrides). Declared coherence below
`recording_config.coherence_threshold` (default 0.9), declared SNR below 10 dB,
or malformed confidence evidence causes an explicit measurement error.
Absent coherence/noise-floor metadata remains supported with unverified
advisories; it does not establish measurement confidence or timing compatibility.
Subwoofer treble data is not required by this check.

### Crossover cancellation

`optimizer.max_crossover_cancellation_db` is a finite, nonnegative dB value
(default `3.0`). It measures coherent main/sub cancellation below the louder
realized branch, not error below the target curve. No input-version change is
required; existing configurations use the default.

```json
{"optimizer": {"max_crossover_cancellation_db": 3.0}}
```

Cancellation within the limit plus 0.05 dB numerical tolerance passes this check.
Above the limit, each logical input must improve by more than 0.05 dB relative to
its fixed pre-optimization routing baseline: 10→4 dB passes with an
`improved_residual_cancellation` advisory; 10→10 and 10→11 dB fail. Passing this
check does not bypass electrical headroom, target-quality, or multi-seat checks.

The optional common main Post-EQ uses a separate fixed target-shortfall rule:
accept at or below 3.05 dB, or improve by at least 20% and at least 1 dB against
the immediate input. These constants have no configuration fields. Its main
response above twice the crossover must not regress against the requested
target (or flatness without a target). Active cancellation screening also
rejects regression greater than 0.05 dB against that immediate input. Final
selection compares alternatives with and without the optional pass under the
same electrical and retained-seat acceptance requirements.

The baseline is captured from the measurements before automatic array alignment,
route optimization, level alignment, and EQ. Configured crossover ranges use
their geometric centre and automatic filter types use the first search candidate;
DBA uses its initial -3 dB rear gain (bounded by `min_db`), 10 ms delay, and inverted
rear polarity. These initial controls are frozen rather than replaced by later
optimized controls. Cardioid geometry remains structural.
For multi-seat cardioid measurements, `front` and `rear` must declare matching
seat counts in the same order, with measured phase at every seat. Preprocessing
combines each pair before spatial EQ aggregation and uses
`optimizer.multi_seat.primary_seat` (default 0) for the routing reference.
Declare both physical outputs; singleton captures are not broadcast across seats.
Non-routed physical outputs are replayed through their owning DSP chain. Each
capture must identify exactly one owner; grouped multi-seat drivers require
explicit stable driver IDs rather than inferred ordering.
In output scorecards, `measurement_overlap_hz` is optional for aggregates whose
independent logical inputs have disjoint frequency support. Existing shared-band
reports retain the two-element array. Each `final_seats` record still identifies
its assessed band; all seats of one logical input must share supported frequencies.

Comparisons use matching frequency grids and the union of the baseline and final
half-to-twice-crossover windows, bounded by measured support and 20–2000 Hz.
Unavailable baseline or phase evidence cannot authorize an above-limit exception.
Per-input baseline/final deficits, worst frequencies, comparison band, limit,
improvement, and acceptance reason appear in
`metadata.bass_management.crossover_cancellation`; frozen baseline spectra are
retained under `metadata.bass_management.optimization.crossover_cancellation`.

### Final electrical limits and cumulative correction selection

`optimizer.finalization.subwoofer_limiter` (default `false`) opts into native
runtime limiting after each physical sub output's sum and EQ, instead of permanent
sub attenuation. The fully wet hard limiter uses a ceiling of
`min(output_ceiling_dbfs, -1)` dBFS, 5 ms lookahead, and 100 ms release. Limiter
mode supports ceiling values in −20..0 dBFS. Other outputs receive matching
latency; the native host supplies compensation without duplicating replay delays.
Reported acoustic curves are small-signal responses, not predictions of gain
reduction during loud programme material. This is sample-peak protection, not a
true-peak or driver-excursion guarantee. Updated native playback is required;
unsupported external exports fail rather than dropping the limiter.

`optimizer.finalization.max_useful_output_loss_db` sets the permitted unexplained
acoustic output loss for mains, surrounds, and heights over the usable playback
band, with a default of 3 dB. Subwoofers are exempt from this SPL-loss allowance;
their electrical safety and crossover/response-quality checks remain active.
For routed mains, SPL-loss evidence uses the physical main branch above its
structural crossover, not its sum with redirected subwoofer bass. Combined
response quality is checked separately. Independent stereo speakers are
observed over their measured passbands even for bass-only correction requests.
For example, `"max_useful_output_loss_db": 5.0` permits a 5 dB tradeoff while
retaining the other quality gates. It is independent of PEQ boost/headroom,
electrical `max_attenuation_db`, and `default_input_peak` assumptions. Never
derive an input-peak budget from a PEQ headroom reserve.

`optimizer.finalization.min_improvement_lower_bound_db` (default `0.0`) is the
benefit floor: every training seat's uncertainty-adjusted improvement
(`improvement_lower_bound_db`, after subtracting pre/post summation
uncertainty budgets) must exceed it, or the candidate is rejected as showing
no demonstrated benefit and selection falls back to a simpler protected
result. Identity candidates (no meaningful correction applied) skip the
floor and flow through as the protected baseline. The default is the
measurement-uncertainty boundary, not a perceptual threshold: raising it
needs repeat-capture and listening evidence (see
`docs/ROOMEQ_LISTENING_PLAN.md`), never a guess.

`optimizer.finalization` defines the electrical assumptions used after all
processing and artifact assembly:

Home-cinema level calibration does not use the requested correction bounds.
It uses the shared measured main-speaker passband above crossover transitions,
preferring 500–2,000 Hz and requiring at least one supported octave. Subwoofer
and LFE stopbands are not included in main-speaker level matching. All main
channels share that calibration reference, including surround and height groups.

If final acceptance is rejected or lacks evidence, the CLI exits nonzero. Any
native JSON written in that case is diagnostic only: its manifest has status
`rejected`, and external playback export is not attempted. Artifact existence
alone is not evidence of a successful correction.

Final routed-seat quality is evaluated over measured playback support, not
only `optimizer.min_freq..max_freq`. The correction band remains separately
reported: a bass-only request must not conceal upper-passband damage. Native
LFE assessment stops at its low-pass cutoff; redirected main inputs include the
combined main/sub response through the main speaker's measured passband.

```json
{
  "default_input_peak": 1.0,
  "input_peak_limits": {"L": 1.0, "R": 1.0, "LFE": 1.0},
  "output_ceiling_dbfs": 0.0,
  "max_attenuation_db": 12.0
}
```

Input names refer to logical inputs. Omitted inputs use `default_input_peak`
(1.0 when unspecified);
unknown names and peaks outside `(0, 1]` are errors. The ceiling must be finite
and at most 0 dBFS. The attenuation limit is finite and between 0 and 60 dB.
These defaults are active even when the object is omitted.

Optional `physical_drive` adds operator-declared sampled steady-sine limits.
Optional `physical_drive_weight` (default `0.0`, finite and nonnegative) adds a
final-candidate ranking cost: weight times squared worst declared demand/limit
ratio, added to mean seat target error in dB. Positive weights require valid
`physical_drive` declarations. Hard safety and acoustic constraints are unchanged;
this does not authorize additional output loss. It searches the existing bounded
final-graph candidate family plus budget-fraction common cuts (except with runtime
sub-output limiting). For retained joint-sub groups it also tries two bounded
gain alternatives per nonreference driver, evaluated on the complete graph.
The reference driver, delays, and shared-EQ design stay fixed; all existing
seat/output/physical gates apply against the original baseline. Trial IDs start
with `joint_drive_gain_`. This does not change the earlier acoustic joint-sub
objective or claim an unrestricted physical optimum.
For example, this is the shape for a **single physical output** (illustrative
numbers only; use measured demand and compatible hardware limits):

```json
{
  "physical_drive": {
    "outputs": {
      "[\"channel\",\"L\"]": [{
        "quantity": "voltage_rms",
        "calibration_id": "voltmeter-session-1",
        "reference_conditions_id": "amplifier-load-protection-session-1",
        "limit_conditions_id": "amplifier-load-protection-session-1",
        "sine_duration_seconds": 1.0,
        "reference_output_peak": 0.1,
        "linear_valid_output_peak": 0.5,
        "frequencies_hz": [40.0, 80.0],
        "demand_at_reference": [1.0, 1.0],
        "limits": [2.0, 2.0]
      }]
    }
  }
}
```

Every emitted physical output needs an envelope. Use exact output IDs from
electrical-headroom diagnostics, not guessed speaker aliases. Other quantities
are `current_rms` (A RMS) and `excursion_peak_mm` (mm peak); watts/SPL are not
accepted as interchangeable units. Missing calibration, different reference
conditions, incomplete output coverage, exceeded limits, and out-of-linear-range
demand refuse delivery after final routed pruning. No physical interpolation
or nonlinear limiter credit is used. Passing records remain scoped to declared
samples/quantities and do not establish thermal, transient, or program capacity.
These same constraints participate in final candidate selection, including
output-specific, common, and spectral attenuation trials. They do not increase
`max_attenuation_db` or authorize useful-output loss. Physical demand and linear
validity are replayed after candidate changes and again after routed pruning.
Runtime limiter action is not credited; required sub-output cuts precede it.
This does not add calibrated physical drive to the joint-array objective.
See [the manual](../../../docs/ROOMEQ_MANUAL.md#declared-physical-drive-limits)
for provenance, bounds, output naming, and remaining limitations.

Finalization tries the complete correction and reduced strengths, preserving
structural routing, crossover and polarity controls. Main and sub correction
strengths can vary separately. Each candidate is replayed with per-output,
common, or cut-only frequency-selective headroom attenuation. Configured
channel-level alignment is reapplied before final checks. Every available
training and held-out seat is checked against the same structural acoustic
baseline, including single-seat systems. No valid candidate means an
optimization error. Rejected alternatives are diagnostic search outcomes;
the selected graph has separate enforced safety checks.
Frequency-selective trials add at most twelve common PEQ/shelf sections, keep
their centers inside the declared correction band, and bound the sum
of their cuts by `max_attenuation_db`. Complete-graph replay, rather than section
gain alone, decides whether the electrical ceiling is satisfied.
Electrical overload outside that band uses a shelf anchored at the band edge;
its in-band acoustic effect must still pass the same seat checks.

`max_attenuation_db` bounds additional attenuation on mains, surrounds, and
heights. Cuts confined to physical subwoofer outputs are exempt. Common cuts
remain bounded because they also attenuate mains. If a subwoofer requires a cut
beyond that budget, spectral trials first apply its output-only safety cut,
then use the bounded common PEQ budget for remaining overloads. All physical
outputs must still satisfy the electrical ceiling, and
every candidate must pass the acoustic checks. The budget does **not** grant an
acoustic output-loss allowance. An intended level reduction must be declared
separately through `optimizer.permitted_output_gain_db`. The selector never
derives that allowance from its candidate. Cutting excess output above the
target remains permitted by the existing useful-output metric.

The electrical bound is for independently phased simultaneous sinusoids on the
reported frequency grid, including serialized filter centers. It is not a
continuous-frequency, transient, true-peak or physical-device certificate.
Final electrical evidence and candidate decisions are retained in
`metadata.stage_outcomes`; rejected trial checks describe alternatives, not the
selected graph. Missing electrical or physical replay evidence is an error.

Multi-position training captures are retained on their native grids until final
DSP replay. The final gate evaluates each logical input at each capture index,
including its contributing physical outputs after routed gain, crossover, delay,
and pre/post-route DSP. Held-out measurements use physical-output names; a routed
validation seat must include every contributing output. Single-seat captures are
not broadcast to missing positions. Named branch captures must have identical seat
labels/order; unnamed captures retain the explicit positional array-order contract.

`metadata.correction_acceptance.acoustic_quality.final_seats` records partition,
logical input, capture index, optional seat label, physical outputs, actual supported
evaluation band, and pre/post target error. Each position's errors use its own
reported band; the aggregate reports the union and common overlap separately.
The runtime worst-position budget is enforced after post-processing;
a regression or missing required replay evidence fails the run, rather than
certifying only an averaged channel. The comparison removes EQ/convolution correction
while retaining the final structural routing for its baseline. Ear-identified CTC
replay and ambiguous multi-seat legacy driver groups are not certified; use explicit
driver topology IDs for the latter. Missing phase cannot establish a coherent sum.
Supporting-source outputs use their separate reference-seat acoustic contract;
they are not certified by the ordinary multi-position correction scorecard.

### Explicit correction gain allowances

Final-seat useful-output assessment accepts explicit broadband correction gain
allowances through `optimizer.permitted_output_gain_db`, keyed by logical input,
for example `{"left": -6.0, "right": -6.0}`. The default is 0 dB. This declares
an intended trim or headroom attenuation; it does not install gain DSP and is
never inferred from the candidate. The measured correction is compared with the
allowance at every training and held-out seat. Unexplained full-band or bass loss
above the 3 dB engineering budget fails final-seat validation. Structural routing
gain is retained in both baseline and candidate and is not a correction gain.
Reports retain logical-input, partition, seat, and calibrated target-shortfall
evidence separately from normalized shape error. Single-seat-only and other
paths outside final-seat replay still require the broader runtime audit.

### Explicit correction band and natural roll-off

`optimizer.correction_band` is optional. It narrows where EQ may actively
shape a response while leaving the configured observation band
(`optimizer.min_freq..=max_freq`) intact for before/after scoring:

```json
"optimizer": {
  "min_freq": 20,
  "max_freq": 20000,
  "correction_band": {
    "min_hz": 40,
    "max_hz": 16000,
    "allow_natural_rolloff": true
  }
}
```

This is an explicit capability compromise for sources that cannot usefully
deliver the full requested extension (for example, leaving a subwoofer's
natural 20–40 Hz roll-off alone). It never installs a high-pass or low-pass
by itself, and it cannot move outside the observation band. A narrowed range
must set `allow_natural_rolloff: true`; otherwise configuration validation
fails rather than silently leaving scored bins untreated. Reports retain the
fixed evaluated band and the requested correction band separately.

### Unequal upper-band measurement support

For ordinary channel processing, measurement provenance `valid_band_hz`
narrows the effective correction bounds to the intersection of the declared
band, loaded frequency grid, and requested optimizer band. Edges must be
finite with `0 < low < high`; an empty intersection is rejected. The raw
measurement and requested configuration are retained. This declaration does
not authorize direct-sound or phase correction or guarantee zero filter
tails outside the band. Ordinary-channel level/passband analysis and IIR/FIR
design select only loaded samples inside the usable band, preserving aligned
phase, coherence, and noise-floor values. At least two loaded samples are
required; edges between bins narrow to the retained grid without extrapolation.
For an acquisition with internal gaps, use `valid_bands_hz` as ordered,
nonoverlapping `[low, high]` intervals instead of `valid_band_hz`. The
measurement loader aligns each interval independently and retains a support
mask for the gap. Single-channel correction consumes the mask end to end:
each segment is conditioned independently, optimization and scoring see only
the union of measured samples, PEQ centers in a gap refuse the channel, and
a per-channel segment report records per-segment scores with observed gap
leakage. Curve-only single-curve and averaging (group/multisub/home-cinema)
paths still refuse this declaration until they consume band-specific
authorization.
Full-band derived phase caches are not reused on the sliced data. This does
not validate upstream acquisition, averaging, or resampling provenance.
Workflow dense-grid conditioning isolates the declared region before reduction,
smoothing, and phase reconstruction, retaining conditioned outer samples for
reporting. Source alignment likewise isolates retained usable samples before
averaging and rejects sparse/disjoint usable support. Its usable grid narrows
to common native sample support, never interpolating from excluded neighbors.
Joint multi-sub search evaluates only shared bins in the active correction band
(at least two required), while retaining full responses for protected-seat
assessment. Its output-loss proxy does not certify physical drive capability.
Native snapshots stay unchanged; conditioned curves are not raw captures and
these rules do not establish calibration validity or complete lineage.
Detailed source and ordinary-channel preparation retain internal, hash-linked
receipts for executed alignment, averaging, and dense-grid conditioning. Ordinary
channel results transport these to the final `measurement_input_conditioning`
stage; report explanations summarize its operations. Missing path coverage and
representative-identity mismatches are explicit, not successful no-op claims.
The stage verifies ordered hash reachability from loader-recorded native curve
identities to all prepared outputs. Known preparation operations also reconstruct final hashes
per source position: an earlier reachable hash cannot stand in for its prepared
output, and missing positions or inconsistent operation roles degrade the stage.
Legacy receipts without roots still parse
but cannot pass that check. Roots are not inferred from unrecognized inputs.
For ordinary single-source channels, root content and order must match the
frozen parsed source used by the run, including configured logical channel
mapping. Missing snapshot attribution degrades the check; final reporting never
reopens source paths. This is not authentication of raw recording bytes.
These records do not authenticate capture/calibration declarations or establish
a verified raw-snapshot chain. Other engine conditioning remains incomplete.
Shared level and `from_measurement` slope preparation apply the same usable
sample selection. Shared level comparison requires common support with at
least two samples per contributing channel; otherwise no shared level is used.
Measured target-slope fallback skips sources without sufficient samples in the
regression window and tries the next non-subwoofer candidate in name order.
Usable bed-channel estimates still take priority; explicit overrides and genuine
zero estimates are retained. The zero default is used only after all eligible
candidates lack an estimate, not merely because the first loadable curve does.

Intake assessment records for magnitude, decay, and absolute loudness use
this usable-band intersection, with explicit unsupported records for the
excluded measured ranges. Assessment edges narrow to the first and last loaded
samples inside the declaration, not unsampled declared boundaries; at least
two retained samples are required. The original declaration stays in provenance.
Missing SNR or decay evidence remains unknown
inside the usable band; declaring a band does not supply that evidence.

Final coherent replay does not extrapolate subwoofer phase or silently truncate
a full-range main. For a routed subwoofer measured through twice its deployed
low-pass frequency, replay bounds the unmeasured stopband from the capture's
final half-octave: a falling measured tail continues at its fitted rolloff in
dB per octave, recorded as `measured_subwoofer_stopband_rolloff`. A rising tail
or fewer than two points in the window keeps the older flat peak-hold
assumption, recorded as `assumed_subwoofer_stopband_below_measured_tail`.
The actual branch DSP is applied to this envelope; omission still requires at
most 0.1 dB aggregate magnitude uncertainty. Both forms apply to route-owned
and per-driver low-passes, not to mains or missing crossover-band measurements.

An explicit calibrated acoustic upper bound overrides this assumption. It is
identified separately for each physical output, partition (`training` or
`held_out`), and seat index:

```json
"optimizer": {
  "upper_band_acoustic_bounds": {
    "sub": [{
      "partition": "training",
      "seat_index": 0,
      "band_hz": [200, 20000],
      "max_spl_db": 20,
      "evidence_id": "calibrated-source-upper-bound"
    }]
  }
}
```

An explicit bound must use the original capture calibration and input reference,
overlap its measured endpoint, and cover the assessment band. Bounds cannot be
broadcast across seats. An explicit declaration is a flat cap unless it carries
a non-positive `rolloff_db_per_oct`, in which case the bound declines at that
rate above `band_hz[0]`; a rising slope is rejected. Replay applies the actual
branch DSP to the bound and allows omission only when aggregate magnitude
uncertainty is at most 0.1 dB.
It does not invent unmeasured phase. Final-seat support evidence records both
magnitude and phase uncertainty, and the improvement lower bound includes
baseline and candidate uncertainty. Missing bounds outside the automatic
subwoofer case prevent an accepted correction. Contradictory declarations
remain configuration errors.

Final coherent replay also checks lower measurement support. A driver or routed
output starting above another branch's measured bass is insufficient evidence
within the requested optimizer band. `upper_band_acoustic_bounds` cannot be
used as a lower-band bound. Supply lower-frequency measurements or explicitly
restrict `optimizer.min_freq` to the common supported band; no lower-band
roll-off is inferred from crossover settings.

## Supporting-source acoustic contract

`supporting_source.delay_ms` is the requested reference-seat propagation-plus-
electrical lag. Set `acoustic_arrival_offset_ms` to measured unfiltered support
arrival minus primary arrival on a common time reference. The installed delay is
their difference; negative delay and a conflicting `optimizer.allow_delay: false`
are errors. Set `shared_phase_reference: true` only for time-referenced measured
phases. The actual FIR/gain/delay are included in `coherent_sum` reporting and the
`max_coherent_cancellation_db` engineering budget (default 3 dB).

Without this evidence, explicit `allow_unverified_acoustics: true` is required for
experimental operation. This opt-in does not establish acoustic validity. Reports
separate power-average design from coherent prediction, flag missing evidence or
budget overruns, and do not claim measured onset, fusion, DRR, or listening benefit.
FIR energy spread and spatial validity require final-chain measurements/listening.

## Measurement provenance

### Per-source direct-sound capture facts

Each measurement source may carry its own `provenance`. For gated/direct-sound
claims, `provenance.direct_sound` contains raw `facts` and an explicitly selected
`policy`. This is separate from the top-level sidecar references described below.
For example, the following **synthetic policy example is not a recommended
measurement standard or universal threshold**:

```json
{
  "path": "measurements/left.csv",
  "provenance": {
    "capture_kind": "direct_sound",
    "timing_reference_id": "loopback-session-1",
    "direct_sound": {
      "facts": {
        "gate_s": 0.002,
        "direct_path_m": 1.0,
        "first_reflection_path_m": 2.0,
        "sound_speed_m_s": 343.0,
        "angular": {"angles_deg": [0.0, -30.0, 30.0]},
        "averaging": "stationary",
        "capture_kind": "direct_sound",
        "sample_rate_hz": 48000.0
      },
      "policy": {
        "version": "quasi-anechoic-v1",
        "cycles_for_valid_band": 2.0,
        "min_off_axis_count": 2,
        "min_off_axis_abs_deg": 15.0
      }
    }
  }
}
```

The validator checks the gate against `(first_reflection_path_m - direct_path_m)
/ sound_speed_m_s`. In this example, two cycles in 2 ms limit support to 1 kHz
and above, intersected with the loaded grid, declared `valid_band_hz`, and capture
Nyquist frequency. Acquisition sample rate is not assumed equal to playback rate.
Off-axis directions must be distinct signed azimuths in [-180, 180]; duplicate
captures at one angle do not increase coverage. Unknown averaging, missing policy,
invalid geometry/rate, or contradictory capture kinds cannot authorize phase/detail
claims. Missing angular coverage prevents detail correction without invalidating
otherwise supported stationary phase evidence. Ordinary stationary room IRs do not
need quasi-anechoic facts unless they also assert direct-sound evidence.

Legacy files remain readable, but `has_direct_angular: true` alone is no longer
proof of detail eligibility. Reasons and assessed bands appear in operation-gate
metadata. These checks establish consistency of supplied evidence, not capture
authenticity, acoustic improvement, or listening benefit. Magnitude target-policy
integration remains a separate requirement; a gate verdict alone is not a claim
that every magnitude-processing branch enforces it.

### Top-level sidecar references

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

Schema **3.0.0** separates logical programme inputs from measured physical
outputs. Stereo declares exactly `L` and `R`; one or two sub measurements are
listed under `system.subwoofers.outputs`. Home cinema adds `LFE` implicitly.
Legacy v2 configurations, stereo `system.speakers.LFE`, configurable
`bass_management.lfe_channel`, flattened sub mappings, and shared-string
subwoofer crossovers are rejected with a migration example.

Schema **2.2.0** adds `optimizer.fir.placement`:

```json
{
  "version": "3.0.0",
  "optimizer": {
    "processing_mode": "phase_linear",
    "min_freq": 20.0,
    "max_freq": 200.0,
    "fir": {
      "placement": "per_driver",
      "taps": 4096,
      "phase": "kirkeby",
      "correct_excess_phase": true,
      "max_boost_db": 4.0
    }
  }
}
```

This is a configuration fragment; supply the usual `speakers` measurements.
`shared` is the default and preserves existing behavior. `per_driver` requires
version 2.2.x and `phase_linear`, `hybrid`, or `mixed_phase` processing. It
currently supports independent legacy or explicit-driver groups (such as paired
main/sub outputs) and standalone speakers. Standalone speakers keep their
existing single-FIR design. Bass-management and shared multi-sub outputs deploy
the common post-route FIR on each physical driver with distinct sidecars and
`room_eq_fir_design_scope: shared_kernel_per_physical_output`. This preserves
every logical-input transfer, rather than independently inverting source curves
that share an output. Pre-route FIRs stay on their logical source. Arrays and
supporting-source outputs remain unsupported for this placement.
Routed frequency-split hybrid blocks also remain unsupported: moving only their
convolution outside the split/merge block would change the realized transfer.
Group captures must have phase and a common timing reference. An output
directory is required for the physical FIR sidecars.

For a group, each physical driver's ordered plugin list receives a convolution
and a unique WAV; no additional shared residual FIR is generated. Retained
gain, delay, crossover and IIR stages remain in place. All physical FIRs in a
group use the same causal support. `mixed_phase` designs phase-only correction
using `mixed_phase.max_fir_length_ms`; the other modes use `fir.taps`.

The joint objective uses the calibrated complex sum and absolute group target.
Magnitude correction is tapered inside the configured frequency band and
limited by both optimizer and FIR boost bounds. Deep local dips (6 dB below
a half-octave neighbourhood envelope), coherence below 0.8, and less than
10 dB signal/noise margin are conservatively protected. Their FIR boost is
limited to 0 dB (0.1 dB finite-realization tolerance); combined-response nulls
are excluded from the target-error objective. This does not identify SBIR with
certainty or remove existing IIR boost. Independent phase adjustment may repair
main/sub interference without adding electrical energy at a room null.

Every candidate is evaluated from its actual finite taps. Non-improving or
unsafe sets are replaced with identity FIRs carrying the same causal delay;
logs and convolution parameters record before/after RMS, protected-bin count,
tap count and latency. Outside-band magnitude leakage is limited to 0.5 dB
on the evaluation grid (finite FIRs cannot have brick-wall correction support).
If a later runtime safety gate removes correction, it preserves each physical
FIR's design delay as an explicit delay stage. Delay metadata on a retained
convolution is descriptive: do not add that delay a second time in the host.
Placement is not automatic: the engine does not compare shared/per-driver modes.

The top-level `version` is validated before paths are resolved or optimization
starts. RoomEQ accepts only the `3.0.x` schema line. The current default is
`3.0.0`. Malformed versions and unknown minor or major versions fail closed
instead of being interpreted with current defaults.

RoomEQ loads canonical configuration and override files with recursive strict
deserialization. Unknown, misspelled, or misplaced fields are errors rather
than ignored comments. This applies inside nested optimizer, FIR, group-delay,
speaker, system, and bass-management objects as well as at the root. Typed map
objects such as `speakers` still accept arbitrary map keys whose values conform
to the declared schema. The generated `input_schema.json` mirrors this contract
with `additionalProperties: false` on fixed-shape objects.
The optional top-level `description` string from older authored configurations
is accepted as prose metadata and ignored during optimization. Other unknown
root fields still fail validation.

For home-cinema bass management, `system.bass_management.lfe_low_pass_hz`
controls the LFE programme path independently of the redirected-bass speaker
crossover and defaults to 120 Hz.

`system.subwoofers.crossover` is a positional list with exactly one key per
entry in `system.subwoofers.outputs`. Every listed key must exist in
`crossovers`. Each selected per-sub low-pass `LP_i` deploys as a low-pass
`crossover` plugin on `channels.<SUB>.drivers[i].plugins` (pre-sum; the
redirected-bass routes omit a duplicate group low-pass) and is reported in
`bass_management.groups[].selected_sub_low_pass_hz` and
`sub_output_results[].selected_low_pass_hz`.

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

## Audibility policy selection and staged rollout

`optimizer.filter_audibility` selects the per-filter audibility veto. Absent
(`None`, the default) disables the veto entirely: legacy output, byte-for-byte.
When present, the default evaluates every filter and records reason-coded
verdicts but never removes anything (`report_only: true`); enforcement
(`report_only: false` plus `allow_enforcement_with_experimental_proxy: true`)
is an explicit experimental opt-in — `report_only: false` alone stays
advisory. Thresholds are implementation-time starting calibrations,
not reference psychoacoustic values — see the tracked
[audibility acceptance contract](../../../docs/ROOMEQ_MANUAL.md#audibility-acceptance-contract-2026-09-20)
and its explicit validation deferrals. A nominal `listening_level_phon` is not
physical SPL calibration, and filter-only proxy adjudication is not proof of
equivalence across seats, programmes, and playback levels. These contract
decisions preserve enforcement defaults and thresholds. See also the manual's
"Staged rollout and release gates" section.

`optimizer.pruning_budget.aggregation` selects `sum` or `max` of cumulative
condition differences; neither averages away a failing condition. For routed
systems and workflows supplied with held-out validation captures, removal is
deferred until final graph selection and evaluated against
complete delivered playback. An explicit `evaluation` declaration is required;
correlated logical inputs additionally require measured phase at every seat.
Missing evidence preserves F0. Final routed assessments use metadata key
`final_routed_graph` and remain Low confidence.

Declared `conditions` must have matching evaluation evidence.
Native single/multiple-measurement EQ workflows can resolve the complete
condition set through this optional declaration:

```json
"pruning_budget": {
  "aggregation": "max",
  "evaluation": {
    "version": "spectral-v1",
    "measurement_ids": ["seat-0", "seat-1"],
    "programmes": [
      {"id": "music", "frequencies_hz": [20, 20000], "spectrum_db": [0, -9]}
    ],
    "listening_levels_phon": [55, 85]
  }
}
```

Measurement IDs name every supplied curve in input order, including zero-weight
curves. Programme arrays must align, contain at least two finite samples, and
use positive strictly increasing frequencies covering the evaluation grid.
IDs must be unique, nonempty, slash-free, and have no surrounding whitespace or
control characters. Levels must be positive, finite, and unique. Nominal phon
values remain experimental proxy assumptions, not calibrated playback levels.

Empty `conditions` uses the complete measurement/programme/level product when
`evaluation` is supplied. An explicit list must match that product exactly;
condition IDs look like `seat-0/music/55phon`. Missing native evidence retains
filters. Without `evaluation`, only `flat-background` is resolved and empty
`conditions` retains that legacy scope.

Policy records are versioned and explicit: promotion needs the release gates
its claim requires (correctness and physical safety for safeguards, plus
perceptual validation for perceptual claims, plus recorded listening outcomes
for benefit claims). Advisory selections and elapsed warning cycles never
promote. Deleting a policy selection restores the legacy behavior.

## Multi-measurement RIR prototype

When a speaker has several measurements captured at different positions, you
can ask RoomEQ to build a single distance- and directivity-weighted prototype
curve and then optimize that curve instead of each measurement individually.

Enable the prototype by adding a `rir_prototype` block inside the speaker's
`multi_measurement` configuration:

```json
{
  "version": "3.0.0",
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

## Measured room impulse responses

`python3 scripts/mdat2csv.py measurements.mdat output_dir --timing-reference-id session-1`
exports supported REW `IRData/SampledData` impulses as `<measurement>__ir.csv`
and adds their declarations to `recordings.json`. Install
`scripts/requirements.txt` first. The reference ID is an operator declaration:
use it only when the captures share that timing reference. Without the option,
IR CSVs are still exported, but are not added to the generated configuration.
Native amplitudes and start times are retained without normalization or
recentering. Legacy IRFloat, minimum-phase reconstruction and filtered/smoothed
storage requiring interpretation are reported as unavailable rather than guessed.

Channels can declare a measured room IR backing the R1–R5 acoustic report
fields (early reflections, early/late curves, octave T60, waterfall with
resonance decays, wavelet). Keys are output channel names; each entry is a
`time_ms,amplitude` CSV captured in the room (swept-sine deconvolution or
equivalent), resolved against the configuration directory:

```json
{
  "measured_impulse_responses": {
    "left": {
      "path": "measurements/left__ir.csv",
      "sample_rate_hz": 48000.0,
      "timing_reference_id": "clock-1"
    }
  }
}
```

- `path` (required): IR file with a `time_ms,amplitude` header.
- `sample_rate_hz` (optional): verified against the file's time grid;
  a mismatch fails ingestion fail-closed.
- `timing_reference_id` (optional for one channel, required on every entry
  when several channels declare IRs): shared clock identity so no
  cross-channel analysis can silently mix clocks.

Ingestion gates (all fail-closed): uniform time grid, finite samples with a
nonzero peak, a declaration for a channel the run actually outputs. The
measured waveform replaces the synthesized `pre_ir`; the R1–R5 acoustic
analyses (reflection table, early/late, octave T60, waterfall with
resonances, wavelet) run on it at finalization, before measurement
extraction and ledger binding. Each analysis that declines on a valid file
leaves its cells pending with a warning. Channels without a declaration
keep synthesized IRs and their report cells stay pending: room-acoustic
analysis never runs on a synthesized IR.

Bass measurements retain their native frequency resolution. Main analysis keeps
its full measured range even when a sub measurement ends near 200 Hz; only a
sufficiently attenuated, falling sub tail is extrapolated for prediction. Sub EQ
stays within trustworthy measured support. Routed multi-seat/all-pass processing
preserves the dedicated engine's filters, primary seat, and global-EQ selection.
`multi_seat.max_deviation_db` defines a soft penalty for
`primary_with_constraints`, not a per-seat hard bound.

### Measurement response snapshots

The public workflow freezes configured numerical measurement responses before
optimization starts. Subsequent numerical loads use the same full native curves
for optimization and final replay, even if a source CSV changes during the run.
Original paths, measurement labels, speaker names, and declared provenance are
retained as metadata. Malformed inline phase arrays are rejected at this boundary.
The internal loaded-response handoff is not a new public configuration syntax.
Associated recording WAVs, external CEA2034/target assets, authenticated capture
identity, and subsequent conditioning are outside this snapshot guarantee.

Output optimizer evidence may contain `input_normalization`, describing analysis
normalization only. Its gain is not an emitted playback trim or an SPL calibration.
The `optimizer_input_conditioning` structural stage retains canonical per-attempt
gain ledgers and lists runs without such evidence. Legacy absence remains unknown.
Multi-measurement PEQ and spatial FIR runs may also contain
`multi_input_normalization`: ordered objective-curve receipts and their population
(aligned measurements, prototype, bootstrap, or bootstrap of a prototype).
Objective indices do not identify physical seats; no acquisition or calibration
authenticity is implied. This is output evidence, not an input configuration field.

Capture-verification manifests (separate from RoomConfig) can supply
`ir_analysis.baseline` for matched raw IR/step and supported octave ETC views. Its baseline graph identity,
source/seat/stimulus, SHA-256, settings, usable support, rate, and record length
are checked against the candidate and verification plan. See the manual's
"Declared calibrated IR comparisons" section for the exact handoff and
`display-roomeq.py --capture-verification` report command. No baseline means
unavailable, not a reconstructed pre-correction capture.
The viewer can also place those bound diagnostics after the plots in one
optimization report with
`display-roomeq.py room-output.json --capture-verification verification-report.json -o room-report.html`.
It verifies the saved optimization payload and referenced FIR bytes first,
then requires the verification candidate graph ID to match. A mismatch or
stale payload yields an unavailable capture section, never an adopted view.
ETC requires declared nominal support for all 500–4000 Hz octave bands and
the full 0–40 ms window plus finite-filter lookahead. The report preserves
filter method, common baseline reference, display floor, and boundary caveats;
it does not infer room decay or acoustic acceptance from those envelopes.

Optional `ir_analysis.decay` adds matched finite-window octave decay views.
It requires `noise_window_ms: [start, end]`, `minimum_fit_margin_db`, and
`minimum_r_squared` with no implicit budgets. The same declared signal-free
window must be applicable to both captures. The report separates common-reference
levels from individually normalized tails and retains per-band noise/fit facts
or explicit unavailable reasons. The T20 slope extrapolation is not passive-room
RT or acoustic acceptance. See the manual for filter support, noise assumptions,
finite-window limits, and the example handoff. These fields belong to the capture
manifest, not RoomConfig; existing RoomConfig schemas are unchanged.

With a matched baseline/candidate IR pair, verification also emits an advisory
`octave_t60` view over 63 Hz–16 kHz. It reports valid T30/T20 fits, fit quality,
and explicit unavailable reasons for missing band or decay support. Its
automatic noise-cutoff and 0.90 fit threshold are analysis conventions, not
acceptance limits; the separate `ir_analysis.decay` path above remains the
declared-budget comparison. See the manual for interpretation and limitations.
The capture HTML may additionally average accepted octave T60 fits across
distinct sources at the same seat when stimulus, baseline graph, sample rate,
capture kind, settings, and declared band match; each mean carries its
contributing source count.
The same pair can emit `early_reflections` from 1–8 kHz filtered IRs when
the full band is declared usable; the report lists baseline/candidate
post-direct peaks within 15 ms above −15 dB relative to direct and retains
unavailable reasons when the evidence is insufficient.
The pair can emit bound `early_late_curves` on a common full peak-band
reference within each capture. It uses a 20 ms early split from the
broadband envelope peak, or the 120 Hz lowpass envelope peak for sub/LFE,
and plots full, early, and late third-octave energy contributions. The
1–8 kHz mean is shown only with full declared band support; it is not an
absolute between-capture level comparison or an optimization DSP JSON field.
The pair can also emit a bound `waterfall` diagnostic if both IRs include
the full 500 ms post-peak interval and analysis half-window. It carries
decimated frequency × time grids within the declared usable band and
60 ms resonance candidates with fitted decay times. Each grid has its own
level reference, so the report does not treat the plot as an absolute
between-capture level or passive-room damping comparison.
The same complete pair can emit a `wavelet` heatmap with three-cycle Morlet
analysis, 64 × 100 maximum displayed cells and −30…0 dB colors relative
to each capture's own wavelet-grid peak. Its presence is an advisory capture
diagnostic, not a per-channel optimization output field.


Optional `ir_analysis.ambient_noise` references a separate silent-playback mono
WAV and a numeric pressure-calibration JSON resource, each with its own SHA-256.
It declares source/seat/candidate graph, acquisition gain, operating conditions,
synthetic status, and explicit FFT frame/band settings. Calibration includes
Pa/full-scale-sample sensitivity and a frequency-response correction table;
the IR magnitude offset is never reused as pressure calibration. Unsupported
numeric support yields an unavailable noise view; identity/hash mismatch rejects
the import. See the manual's complete handoff/resource examples and estimator
conventions. Noise views work without a baseline IR and do not alter acceptance.

### Independent-clock microphone capture provenance

A multi-microphone `measurements` source may carry `provenance.capture`.
`capture.geometry` is `spread` or `compact`; `capture.takes` contains exactly one
entry per measurement in the same order. This preserves failed timing fits
instead of dropping microphones or implying a zero timing error.

Each take records `microphone_id`, actual `device_id`, `offset_samples`,
`skew_ppm`, `residual_uncertainty_us`, `correction_applied` (`none` or `resampled`),
`timing_reference_id`, `calibration_id` (the frozen file SHA-256 for SOTF captures),
`calibration_orientation` (`on_axis` or `ninety_degrees`), `gain_db`, `position_m`,
and `position_uncertainty_mm`. `preserves_acoustic_delay` distinguishes surveyed
reference correction from alignment that removed unknown propagation delay.
`quality_passed` defaults to false: a clock fit alone never passes measurement QA.
Unknown offsets, skews, and bounds are JSON null, not zero.

Clock eligibility requires every take to have a finite residual bound strictly
below **50 microseconds**, finite offset/skew, resampling that preserves acoustic
delay, passed quality, and a common reference matching the source's
`provenance.timing_reference_id`. This bound corresponds to 9 degrees at 500 Hz;
higher-frequency phase work and direction estimation require additional limits.
Compact surveys require position uncertainty no greater than 1 mm.

When this block is present and any check fails, measurement consumers receive
spatial-magnitude provenance with no timing reference. A top-level stationary-IR
label cannot override a failed microphone. Serialization retains the original
block for viewer diagnostics. Legacy sources without the block retain their
existing acquisition contract; single-microphone REW import is unchanged.

SOTF exports shared-origin calibrated phase CSVs and `quality_passed: true`
only when every microphone for a source has accepted common-reference artifacts,
no clipping or analysis issues, and broadband plus nonempty frequency-band SNR
of at least 30 dB. Otherwise it retains calibrated magnitude CSVs with
`quality_passed: false`. Incomplete magnitude analysis keeps the canonical export
pending. A valid phase export still requires the frequency-specific clock gate
below before complex pair sums or direction overlays are permitted.

The HTML report's **Capture clock QA** table displays session geometry and each
microphone's device ID, sample offset, ppm skew, residual bound, correction, and
coherent-use status. Comparison reports evaluate clock evidence independently for
each mode. L+R becomes a magnitude-only power sum when bounds, quality, common
reference/layout, or measured phase are missing or invalid, and the report states
the reason. A power sum does not predict coherent pressure cancellation. The
fallback contains no phase, so it cannot feed phase or group-delay overlays.
The viewer also refuses coherent L+R for older reports without clock evidence;
their individual-channel and magnitude views remain available.

Capture phase consumers additionally validate each residual at the requested
upper frequency: the conditional clock phase error must not exceed nine
degrees (frequency in Hz times residual in microseconds must not exceed
25000). Missing or zero uncertainty does not authorize a frequency band.
This check applies to phase correction, coherent summation, multisub reference
scopes and physical crossover reference scopes. It does not narrow magnitude
evidence. Capture viewer L+R summation uses the sum of the two source bounds
for a worst-case relative-error check over the plotted band.

Optional `provenance.capture.reflection_report` preserves the source identity,
direct-arrival candidate, early-reflection candidates, and refusal reasons.
Each event records shared-origin arrival time, direct-relative time and level,
optional source-facing direction, planar mirror ambiguity, pair-fit residual,
and analysis band. Direction observations remain conditional on the plane-wave
and microphone-phase assumptions; fit residuals are not angular uncertainty.
Reports render these observations per logical channel, matching the configured
source mapping. Direction plots suppress failed clock evidence, out-of-band
timing accuracy, malformed vectors, and unresolved planar mirrors.

Arrival `microphone_energy_db` values are integrated event energies relative
to each microphones direct SSIR segment, in capture microphone order. Omitted
or empty vectors cannot supply measured prototype weights. When RIR prototype
weighting is requested, valid per-event angles form an energy-weighted mixture
of the configured directivity model, replacing geometric angles in supported
bins only. This remains a magnitude-only spatial-weight approximation, not
coherent reconstruction of the reflected field. Invalid capture evidence or
ambiguous geometry retains the geometric weighting path.
