# RoomEQ - Multi-channel Room Equalization Optimizer

The HTML report starts with playback status, followed by collapsible evidence,
summary, speaker details, timing, symmetric monitors, time-domain analysis,
EPA, and DSP signal-flow sections. Each section's speaker/group selector is independent.
The final section opens with an All channels view of the complete saved graph,
with shared inputs branching to every physical output. Individual output tabs
remain available. It shows numbered processing nodes, including every
contributing input and explicit routing sums. Expand the matching table rows
for saved plugin parameters. Routed diagrams separate pre-route processing,
route gain/crossover/delay, and post-route processing, without duplicating
route-owned plugins. Unsupported legacy ownership is reported, not guessed.
Diagrams use the same d3rs/WASM renderer with or without WebGPU.
Waterfall surfaces support mouse dragging with Rotate enabled and Reset restores
the initial view. Wavelets show frequency horizontally and early time vertically,
with a shared full-grid peak reference and a −30…0 dB color scale.

Regenerate the DSP output to obtain dense measured-IR diagnostics: replotting old
bundles cannot recover discarded samples. New exports retain positive waterfall
FFT bins and 2 ms frames; wavelets use 48 samples/octave, 0.1 ms early samples
through 15 ms, then 1 ms samples through 500 ms. `times_ms` is authoritative;
wavelet `hop_ms` describes the minimum nominal step, not uniform spacing.
This increases display sampling, not the acoustic resolution of the analysis
window. Until the dense-wavelet API is published and pinned, `.cargo/config.toml`
uses sibling `../math-audio` DSP and IIR/FIR crates for development builds.

Python report channel EQ plots and phase-aware replay evaluate serialized
Kautz and warped topologies. Kautz parameters are linear basis weights with a
unity dry path, not independent PEQ dB gains. Channel EQ decomposition uses the
output sample rate (48 kHz only for legacy outputs without a rate). These are
transfer predictions, not measured playback or listening evidence. Comparison
overlays retain saved EQ curves when available; otherwise they reconstruct each
mode at that output's sample rate, also defaulting to 48 kHz for legacy outputs
without a rate. Generated frequency grids do not exceed that mode's Nyquist.
Comparison summaries count emitted EQ sections per channel, including
driver-local entries and zero-weight Kautz basis sections. Shared channel
entries count once; descriptive `route_owned` markers are excluded. These are
inventory counts, not estimates of CPU cost or acoustic benefit.

Parallel-driver waveform replay requires finite, grid-aligned phase for every
individual capture. Missing phase is not assumed to be zero: coherent replay
fails with the driver name, and report refresh clears its predicted post-IR.
An independently phase-supported combined reference IR and filter-kernel
temporal metrics may remain available; neither establishes the missing acoustic
sum. Valid phase arrays alone also do not establish synchronized captures:
common timing provenance remains a separate prerequisite. Report refresh now
checks the configured group/topology/multi-sub/DBA/cardioid source mapping, complete driver
indices (and explicit topology IDs), matching stationary timing and seat labels,
and support across the waveform's full measured band. Missing, mismatched, or
unsupported mappings withhold the summed prediction and save the reason. This
declaration gate is not raw-recording authentication or complete resource binding.
When this timing gate fails, both acoustic traces are withheld: the group
reference may itself be a synthesized sum. Independent FIR-kernel metrics remain.
Main-role and declared sub-output aliases resolve to their source configuration;
conflicting mappings are refused. DBA uses the production front/rear aggregate
branches, but timing admission covers every contributing source in both arrays.
When separate single-capture sub outputs are grouped into one internal processing
chain, timing validation resolves all declared outputs in driver-index order and
checks each output ID. It does not mistake the chain's owning `Sub1` alias for
the sole contributing capture. Repeated report refreshes retain errors in the
waveform status but warn only when the replay failure changes; a recovered view
allows a later recurrence to be reported again.
Cardioid uses its front/rear individual branches. Supporting-source processing
emits separate channels and is not a parallel-driver mapping in this gate.
Joint per-driver FIR generation now uses the same admission check before design
or sidecar writes. An explicitly requested per-driver design with missing or
incompatible source timing is rejected, rather than silently generating coherent
filters from phase arrays alone. This covers the design target's full grid;
source declarations still do not authenticate the underlying recording.
Both FIR and non-FIR
parallel predictions replay individual captures through branch processing and
the common chain once, rather than applying a PEQ summary to a combined curve.
Non-FIR waveform pairs are withheld if required branch evidence is unavailable;
the reason is logged. They are not replaced by a combined-curve PEQ fallback.

Availability is also saved in `metadata.stage_outcomes` under `waveform_views`,
with `pre_ir:<channel>` and `post_ir:<channel>` checks. Each refresh replaces old
checks and clears stale waveform views, including channels without measurements.
Failed checks retain the initial-evidence or replay error; passing checks mean
only that a predicted view was saved, not that acoustic acceptance passed.
Main and comparison HTML reports display these reasons near the top. Conflicting
stage records or disagreement with saved view presence are shown as unavailable.
This diagnostic panel does not independently verify capture timing or graph binding.
Malformed diagnostic containers, duplicate checks, or checks that do not cover
both waveform views for every delivered channel make the panel unavailable.
Legacy outputs without a waveform stage continue to omit this panel.

Parallel-driver pre/post waveform pairs now use the same uncorrected peak
reference. Attenuation, boost, and polarity remain visible; post curves are not
independently peak-normalized. Each acoustic curve is reconstructed directly,
without dividing by the uncorrected response at cancellation nulls. The view
still uses a finite-period 65,536-point IFFT and a 400 ms crop, with endpoint
extrapolation outside the measured band; it is not a full-length raw recording
or an independent physical-room decay measurement.

`roomeq` is a command-line tool for optimizing multi-channel speaker systems. It analyzes frequency response measurements and generates optimal DSP chains (EQ, crossovers, gains) for each channel.

## FIR, MIXED, and MIXED-PHASE: which mode should I use?

Main/sub phase alignment requires declared common stationary capture timing
for the mains and every contributing physical sub source, with a meaningful
reference identity. Blank references and the literal `unknown`
(ignoring case and surrounding whitespace) are absent for admission, even when
all sources use the same placeholder. The source declaration is retained; valid
magnitude evidence is assessed separately from phase permissions.
Matching-looking phase curves do not establish synchronization. Missing or incompatible
references leave supported magnitude processing available and report why
phase-sensitive alignment was skipped. Explicit joint-sub selection takes
precedence over legacy multi-seat processing for that group.

Joint-sub search scores
only shared measured bins inside the configured active correction band and
requires at least two such bins. Before/after objective components use that
same band; changing an outside-band target cannot steer the search. Full-band
responses are retained because array gain/delay controls can affect them.
The output/drive score is a measurement-relative output-loss proxy, not a
calibrated amplifier/driver-demand objective; physical limits remain separate.

Joint-sub array proposals must also pass the existing runtime per-seat
ERB-weighted target-error check, not only improve the aggregate objective.
If a retained seat regresses beyond the numerical tolerance, the array gains,
delays, responses, and objective values revert together. The stage diagnostics
carry `array_rejection_reason`. Absence of that field in legacy results does
not prove that this check ran. This is an engineering array-stage check, not
an audibility guarantee or acceptance of later shared EQ and final routes.

The following shared-EQ stage is checked separately against every retained
seat, using that optimizer's frozen target and common level reference. The
target is explicitly aligned from the canonical optimization grid to the
retained measurement grid. A failed check removes the shared filters before
graph construction, recomputes identity loss, and records
`shared_eq_rejection_reason`. Useful shared EQ remains available when all
seats pass. These two stage checks still do not certify later trims, routed
coherent sums, limiter operation, physical capacity, or actual playback.

Normal finalization also carries these rejection reasons into the bound
decision ledger. The top-of-report explanation shows one historical record
per affected physical sub output, its assessed measurement-frequency support,
and the stage correction-band limits separately. Seat references combine the
recorded timing-reference label with the retained positional index; they are
not invented acquisition IDs. Rejected stage proposals remain history even
if later processing changes, and are never counted as applied delivery claims.

Joint-sub runs retain `channels.<channel>.joint_sub` stage diagnostics for all
seats, including raw-reference output levels and the shared-EQ response.
These are predictions before subsequent trims and routing, not a recorded
playback verdict or calibrated output-capability proof. Changed channel
controls mark them as historical; they are not discarded. Physical sub IDs
follow emitted driver IDs, so two subs may share the same speaker model name.

These names describe **how the correction is built**, not which speakers receive
bass. In particular, **MIXED and MIXED-PHASE are different modes**.

| Name in examples / filenames | `optimizer.processing_mode` | What it does | Main controls |
|---|---|---|---|
| IIR | `low_latency` (alias `iir`) | Parametric IIR EQ for magnitude correction. Lowest filter latency of these choices. | `num_filters`, gain/Q bounds |
| FIR | `phase_linear` (alias `fir`) | Designs the correction as an FIR convolution filter. Its phase behaviour depends on `fir.phase`. | `fir.taps`, `fir.phase`, `fir.correct_excess_phase` |
| MIXED / hybrid | `hybrid` (alias `mixed`) | Combines IIR and FIR **magnitude correction**, either in separate frequency bands or as IIR followed by residual FIR correction; see below. | `mixed_config` when splitting bands, plus `fir` |
| MIXED-PHASE | `mixed_phase` | Uses IIR for magnitude EQ, then a short, approximately unity-magnitude FIR to correct residual **excess phase** where supported. | `mixed_phase.max_fir_length_ms` and phase safeguards |

Use the JSON spellings above: the display/file label `MIXED-PHASE` is written
`"mixed_phase"`, with an underscore, in configuration.

### FIR: convolution does not necessarily mean linear phase

`phase_linear` is the historical processing-mode name. It selects the FIR
correction path, but does **not** override `optimizer.fir.phase`:

- `"linear"`: linear-phase FIR magnitude correction. The filter has constant
  delay; this alone does not remove the loudspeaker/room's existing excess phase.
- `"minimum"`: minimum-phase FIR magnitude correction. FIR is a filter
  implementation, not a promise of constant group delay.
- `"kirkeby"`: regularized inversion; with `correct_excess_phase: true`, the
  design also uses measured acoustic phase. That combination requires phase
  evidence and is not simply a symmetric linear-phase magnitude EQ.

For example, `optimiser-fir.json` in the Genelec measured case currently selects
`phase_linear` **and** `fir.phase: "kirkeby"` with excess-phase correction enabled.
Calling that preset “FIR” is correct; calling its correction “linear phase” just
because of the mode key is misleading.

### MIXED: IIR and FIR both participate in magnitude correction

There are two forms of `hybrid`:

1. **With `mixed_config`: frequency-split processing.** RoomEQ splits and
   recombines the signal. `fir_band: "low"` means FIR below
   `mixed_config.crossover_freq` and IIR above it; `"high"` reverses the assignment.
2. **Without `mixed_config`: serial residual correction.** RoomEQ first fits IIR
   EQ, then designs FIR correction for the remaining error against the same
   calibrated target. There is no user-selected FIR/IIR split frequency.

The FIR part still follows `fir.phase`; hybrid is not necessarily linear-phase
and does not have a fixed latency advantage over FIR-only processing.

### MIXED-PHASE: divide the work by magnitude versus phase, not by frequency

For standalone `phase_correction`, direct-sound evidence must cover the whole
requested detail band without gaps. `max_correction_latency_ms` bounds the
generated FIR's causal centering delay, not measured acoustic propagation.
The report distinguishes `causal_center_delay_ms` from `estimated_delay_ms`;
neither alone describes latency from all DSP stages and backend buffering.

Serial FIR temporal analysis composes all emitted convolution resources;
retained in-memory taps are used only when one convolution owns them.
Missing resources make the evidence unavailable rather than certifying a
partial chain. These FIR metrics still exclude other filter types, alignment
delays, and backend buffering; they are not a total-system latency guarantee.

Phase and direct-sound permission records must be structurally valid, cite
nonempty evidence reference IDs, and match the channel's measurement identity.
A permission label alone cannot authorize the FIR. These checks do not
authenticate a capture; its acquisition and usable-band assessment remain
separate requirements. Refusal leaves existing magnitude processing intact.

“Excess phase” is the part of the measured phase that is not explained by the
minimum-phase response associated with its magnitude. MIXED-PHASE uses IIR EQ
for magnitude and a separate short FIR for the residual excess phase. It does
**not** mean “FIR for bass, IIR for treble” and does not use `mixed_config` to
choose a split.

The short FIR is intended to leave magnitude approximately unchanged. RoomEQ
can reduce its correction depth or omit it when the magnitude/pre-ringing
safeguards cannot be met. With no phase data, this path falls back to IIR-only
processing; selecting the mode is not evidence that a phase FIR was delivered.
Check the exported chain and report.

Use `mixed_phase.max_fir_length_ms` to control this phase FIR's length; changing
`fir.taps` is not the control for that stage. A 10 ms length limit is not a
guarantee of 10 ms total playback latency or complete room-phase correction.

### Three different frequency settings: do not confuse them

| Setting | Meaning |
|---|---|
| Correction bounds (`optimizer.min_freq` / `max_freq`, or an explicit `correction_band`) | Where EQ is allowed to make corrections; not the complete measured speaker passband used for playback assessment. |
| `optimizer.mixed_config.crossover_freq` | Internal FIR/IIR processing split, used only by frequency-split hybrid. |
| Speaker/subwoofer crossover (`crossovers` referenced by the system's bass management) | Routes bass between physical speakers and subs. Independent of the FIR/IIR split. |

For the Genelec **40–200 Hz** correction range, a hybrid split at **300 Hz**
leaves no upper correction band. Keep the intended correction ceiling and put
the split strictly inside the usable correction interval, for example:

```json
{
  "optimizer": {
    "processing_mode": "hybrid",
    "min_freq": 40.0,
    "max_freq": 200.0,
    "mixed_config": {
      "crossover_freq": 100.0,
      "crossover_type": "LR24",
      "fir_band": "low"
    },
    "fir": { "taps": 4096, "phase": "linear" }
  }
}
```

This fragment assigns FIR correction to 40–100 Hz and IIR correction to
100–200 Hz, with crossover overlap; it does not set the speaker/sub crossover
to 100 Hz. Both processing bands also need usable measurement support.

Ordinary channel preparation intersects requested correction bounds with
the loaded grid and provenance `valid_band_hz`, when declared. Invalid or
non-overlapping declared bands are rejected without altering the original
measurement or requested configuration. This bounds the correction range;
it does not grant phase/direct-sound permission or eliminate filter tails.
Ordinary-channel level/passband analysis and IIR/FIR design use only loaded
samples inside that band, including per-position optimizer inputs. At least
two samples must remain; the selected grid is not extrapolated to declared
edges. Raw curves remain available for reporting/replay, and measured phase,
coherence, and noise-floor arrays are sliced together. Upstream acquisition
and conditioning provenance still require separate validation. Workflow
dense-grid reduction, smoothing, and phase reconstruction condition the
declared usable region independently of excluded samples, then retain the
conditioned outer response for reporting. These loaded curves are not immutable
raw captures. Source loaders likewise align each declared usable region
independently before spatial or coherent averaging. Their shared usable grid
is restricted to the intersection of retained native sample supports; declared
edges are not filled from excluded neighbors. Native source snapshots remain
unchanged, and loaded outer responses remain available. This does not establish
calibration validity or complete conditioning lineage.
Detailed source loading now produces canonical hash-linked operation receipts
for alignment and averaging. Ordinary channel preparation adds dense-grid
conditioning receipts and retains them on the prepared engine input. These are
producer-level records, transported by ordinary channel results to the final
`measurement_input_conditioning` structural stage before final identity binding.
The stage checks receipt shape, ordered hash reachability from producer-recorded
native curve roots to every prepared curve, and representative identity against
the reported initial curve. Prepared representative and individual hashes must match
the final identities reconstructed per source position from recorded alignment,
spatial averaging and dense-grid operations. Missing positions, stale intermediate
hashes, unknown operation versions and inconsistent operation roles degrade the
stage; hash reachability alone is insufficient. Native roots separately match
the frozen parsed input curves in source order, using system channel mappings
where configured. Final reporting never reopens the original measurement files.
This binds numerical input, not raw recording bytes or acquisition declarations.
Missing roots (including legacy receipts), broken
dependencies, missing receipts and mismatches produce a degraded stage, not a
claim of unchanged input. The opening explanation summarizes recorded operations.
Grouped/routed paths without receipts remain explicitly unreported. These records
bind numerical inputs and outputs, not authenticated recording provenance, a
verified raw-snapshot chain, or a fully pinned implementation build.
Shared room-level and measurement-derived slope references also honor declared
usable support. Shared levels are compared only over common loaded support;
disjoint overlap or fewer than two retained samples on a contributing curve
leaves the shared reference unavailable. This does not establish SPL calibration.
For measurement-derived target slope, usable bed-channel estimates take priority.
If none are available, RoomEQ tries non-subwoofer fallback channels in name order
until one has enough usable samples in the existing regression window. Loading
a curve without a slope estimate does not make it a measured flat reference.
A genuine zero estimate remains valid; only exhausting all eligible candidates
uses the existing zero-slope default. Explicit slope overrides are preserved.
Magnitude, decay, and absolute-loudness intake assessments likewise retain
the usable band and explicitly mark excluded measured ranges unsupported.
An otherwise unknown assessment remains unknown within the usable band.
When declared edges fall between loaded bins, the assessed range narrows to
the first and last retained samples, matching the selection used for processing.
Fewer than two retained samples cannot establish an assessed band. The original
declaration remains in provenance; excluded edge intervals are explicitly
unsupported rather than presented as measured assessment coverage.

### Practical choice and latency

- Start with **IIR** for bass-peak correction and a low-latency baseline.
- Choose **FIR** when you want convolution-based correction and can accommodate
  its length, latency, and phase-design trade-offs.
- Choose **MIXED** when you specifically want frequency-split or serial
  IIR-plus-FIR magnitude correction.
- Choose **MIXED-PHASE** when you have trustworthy, timing-referenced phase
  measurements and want bounded excess-phase correction alongside IIR EQ.

A symmetric linear-phase FIR of `N` taps has filter delay `(N−1)/(2 Fs)`:
4096 taps at 48 kHz is about **42.7 ms**, before host buffering and other stages.
That formula does not describe every minimum-phase or Kirkeby design. Compare
the delivered chain's latency, level-matched response, crossover summation,
pre-ringing, and individual seats—not just the selected mode or tap count.
None of these modes can reliably invert deep room cancellations or repair
unknown measurement timing. Phase-sensitive summation needs matching seats and
a common timing reference; a spatial magnitude average is not a phase capture.

## Bass-only target normalization

For correction below 500 Hz, target normalization uses the supported
500–2,000 Hz reference band when available. A band-limited speaker instead uses
its measured usable passband if that extends beyond correction. Full-band EQ
retains its in-band normalization. IIR and FIR share this rule; neither requires
subwoofer treble measurements. Passband detection uses an octave-smoothed
peak-relative threshold so a long measured stopband cannot lower the reference.

### Explicit input-peak budgets

`optimizer.finalization.max_useful_output_loss_db` is the separate acoustic
output-loss allowance for mains, surrounds, and heights (default **3 dB**; set **5 dB** for a more
permissive correction/output tradeoff). It applies to unexplained loss after
accounting for the intended target correction and explicitly authorized gain
changes, over each speaker's supported playback band, including a bass check.
For a routed main, this evidence replays only that physical main output over
its measured passband above the structural crossover. Redirected subwoofer bass
is excluded from the main's SPL budget, but remains in the independent combined
response and crossover checks. Independent stereo speakers are also observed
over their measured passbands, not merely the requested correction range.
Subwoofers are exempt from this allowance; their loss evidence remains visible.
Electrical safety, response-shape, and crossover checks still apply independently.
The structural fallback installs necessary attenuation on the overloaded physical
outputs instead of lowering every input to the worst output's headroom requirement.
It does not authorize loss without the other correction-quality checks passing.

`optimizer.finalization.default_input_peak` declares a linear-amplitude input
bound; `input_peak_limits` can override individual logical inputs. For example,
6 dB of guaranteed upstream reserve corresponds to `10^(-6/20)`, or
`0.5011872336272722`. This is only valid when the reserve is actually enforced
upstream; a PEQ headroom setting does not justify it. The LFE bound is applied
before its playback gain.

These are playback assumptions, not an inserted attenuator or limiter. An
upstream chain must honor them; otherwise the electrical guarantee does not
apply. `system.bass_management.headroom_margin_db` alone does not establish an
input bound. Unspecified budgets retain the default full-scale assumption.

### Finalization attenuation by output role

`optimizer.finalization.max_attenuation_db` limits additional attenuation on
mains, surrounds, and heights. Physical subwoofer-only cuts have no attenuation
budget; their electrical output ceiling and final acoustic checks still apply.
A shared cut remains bounded because it also reduces main output. Rejection
diagnostics name the constrained output and list the required cuts by role.

### Declared physical drive limits

`optimizer.finalization.physical_drive_weight` optionally ranks feasible final
graph candidates by mean seat target error plus this weight times the square of
the worst declared physical demand/limit ratio. Its default is zero (the existing
acoustic-only ranking). A positive weight requires `physical_drive` declarations;
it is a user-selected engineering trade-off, not a perceptual threshold. Every
electrical, physical, output-preservation, and seat gate remains mandatory.
The selector evaluates its bounded candidate family instead of stopping at the
first acoustically adequate result. Unless runtime sub-output limiting is selected,
it additionally tries common cuts of 1/64, 1/16, 1/4, and the full existing
attenuation budget at each correction strength. These are bounded search samples,
not audibility thresholds; exceeding an acoustic limit still rejects the candidate.
`final_candidate_objective` records both components as selection-stage history,
not approval of later pruning. For retained joint-sub groups, positive weight
also reopens nonreference driver gains at the complete-graph boundary: two
midpoint proposals toward the configured gain bounds per driver, keeping the
reference driver fixed. Each proposal passes through the same electrical,
physical, primary-target, protected-seat, and output-preservation gates against
the frozen original baseline. These trials carry `joint_drive_gain_*` IDs.
The earlier acoustic array search and its historical diagnostics are not
relabelled as calibrated; changed controls invalidate the stage's current-chain
binding. This bounded refinement does not reopen delays or redesign shared EQ,
and does not establish an unrestricted physical optimum.

Routed joint groups may use configured physical-output names different from
the optimizer's positional group IDs. Refinement validates this mapping and the
canonical routing graph; historical stage IDs remain unchanged. Refinement
trims are correction-owned post-route gains, not a second application of the
array gains already included in the routing matrix.

`optimizer.finalization.physical_drive` optionally enforces operator-declared
physical demand at the final serialized output boundary. Supported quantities
are `voltage_rms` (V RMS), `current_rms` (A RMS), and `excursion_peak_mm` (mm peak).
These are not microphone SPL values. For every declared frequency, demand is
`demand_at_reference * digital_output_peak / reference_output_peak`.
Limits must use the same units, hardware/load/protection conditions, and steady
sine duration. No physical envelope is interpolated or extrapolated.
The reference is at the output **after serialized DSP**: do not include those
filters again in the demand envelope. Keep all required driver protection
active during acquisition and distinguish external protection from the
serialized chain. This option does not authorize playback or new measurements.

The policy must cover **every physical output** with at least one envelope.
Use the exact output IDs in the sampled-electrical-headroom diagnostics:
independent channels use JSON-string IDs such as `["channel","L"]`, independent
drivers use `["driver", channel, index, driver_name]`, and routed graphs use
their declared output names. These are emitted graph identities, which may
differ from input speaker aliases. Unknown/missing output IDs are errors.

Each envelope requires `quantity`, `calibration_id`, `reference_conditions_id`,
`limit_conditions_id`, `sine_duration_seconds`, `reference_output_peak`,
`linear_valid_output_peak`, `frequencies_hz`, `demand_at_reference`, and `limits`.
Reference/limit condition IDs must match; blank/unknown calibration is refused.
Both digital peaks are positive linear amplitudes relative to full scale,
with `reference_output_peak <= linear_valid_output_peak <= 1`.
Frequency, demand, and limit vectors must have matching lengths (at least two),
with increasing positive frequencies, nonnegative demand, and positive limits.
Equality to a limit passes; greater demand or a peak beyond the declared linear
range refuses delivery. The declaration budget is 65,536 samples in total.

The check replays actual routes, PEQ/FIR, gain, delay, and physical driver paths
after routed pruning. Independent inputs retain their independent peak bounds;
opposing polarities cannot claim cancellation between unrelated signals.
Nonlinear limiter attenuation is not credited. Passing records appear under
`metadata.stage_outcomes` as `final_graph_declared_physical_drive`, with demand,
units, utilization, limits, and the original declaration retained in checks.
No physical check or capacity evidence is invented when the policy is absent.

This is **sampled, steady-sine, operator-declared evidence only**. It does not
authenticate calibration, assess unlisted quantities/frequencies, shared power
supplies, thermal limits, program peaks, distortion/compression, or real playback.
It does not substitute for acquisition-dependent acceptance views or authorize
additional attenuation/output loss. Final candidate selection includes these
limits in its output-specific, common-gain, and spectral attenuation trials.
Required attenuation uses same-unit demand/limit amplitude ratios and the
declared linear-valid output peak. It is combined with digital requirements
by taking the stricter cut, not by adding unlike units. Each changed candidate
is replayed through the physical gate and the existing seat, target, crossover,
and useful-output checks; the post-pruning physical gate remains mandatory.
Physical sub-output cuts precede the terminal limiter and do not credit its
nonlinear action. The structural fallback is checked too: making it physically
feasible does not turn an acoustically rejected fallback into accepted EQ.
Candidate diagnostics retain required output and physical attenuation in dB.
The joint-array optimizer itself still uses an acoustic output-loss proxy,
not calibrated physical drive; bounded final selection does not prove that no
other feasible design exists when its candidates are exhausted.
The assessment concerns the native serialized graph; backend conversion and
actual hardware operation still require independent verification.

The Python HTML report opens with the saved playback verdict followed by
**Why this correction?**, before scores and plots. This explanation shows
recorded correction bands, final acceptance, unassessed seat bands, and
advisory filter decisions. Expand its constraints and stage history to inspect
reversions and failed checks. Comparison reports include a section for each mode.
Configured bands describe correction scope, not proof of applied changes;
filter-center verdicts are not frequency intervals. Missing reasons remain
explicitly unavailable rather than being inferred from response curves.
The per-speaker report renders a 1–8 kHz early-reflection section only when
an optional `early_reflections` output field declares a measured-room IR,
the specified band-limited method, and valid pre/post event rows. It shows
time/relative-level plots, six-column event tables, and post-event first-dip
markers on the valid final response curve. These markers are two-path
estimates, not measured spectral nulls. Missing or invalid evidence stays
pending; the optimization producer of this field remains a separate task.

The HTML report displays the saved playback verdict before its scores. Rejected
or unverified results are diagnostic only. Reduced input-peak assumptions are
shown explicitly in dBFS in both single-run and comparison reports; displaying
a report does not enforce those assumptions or independently replay the graph.
The measured-result audit also requires unique useful-output evidence for every
replayed seat and checks non-subwoofer passband and bass loss against the
configured allowance, even when the saved verdict says `accepted`.

The single-run report's **Level Gains and Safety Attenuation** table lists global,
channel, and driver gain plugins, including their stage and purpose. The EQ-only
overview does not show these gains. Do not sum entries across owners or count
route-owned entries twice: routed input-to-output transfer also includes routing
gains and crossovers. The post-DSP response is the appropriate view of their
combined acoustic effect.

### Runtime sub-output limiting

Set `optimizer.finalization.subwoofer_limiter: true` to preserve normal bass level
and limit overloaded **physical subwoofer outputs after routing, summation, and
output EQ**. The default is `false`, retaining static electrical attenuation.
Mains, surrounds, and heights retain their separate 3 dB useful-output-loss budget.
Input peaks remain explicitly configured; this mode does not pretend full-scale
simultaneous input peaks are impossible.

The native limiter uses a sample-peak ceiling of the lower of −1 dBFS and
`output_ceiling_dbfs`, 5 ms lookahead, 100 ms release, hard limiting, and 100% wet
mix. Limiter mode requires an output ceiling in −20..0 dBFS. Its latency is
matched across the physical outputs, preserving crossover timing. The native
host supplies this compensation; offline replay represents it with explicit
delays (integer-sample lookahead, e.g. 220 samples at 44.1 kHz).

Acoustic plots and quality scores describe **small-signal playback below the
limiting threshold**. During overload, bass gain is reduced dynamically. The
electrical report retains the unclipped linear sum as pre-limiter evidence and
labels protected outputs separately. This is sample-peak protection, not a
true-peak, amplifier-power, or driver-excursion guarantee. Do not bypass the
limiter or add gain after it. The updated native player is required; external
exports are rejected until they can preserve the same protection contract.

Rebuild the native player against the matching updated AutoEQ model as well as
the updated RoomEQ adapter. Older AutoEQ dependencies require the obsolete
`physical_sub_output` field and cannot deserialize the new physical-output schema.
The limiter plugin itself is unchanged; no installed playback application is
automatically upgraded by running the RoomEQ CLI.

### Subwoofer hardware gain and LFE playback gain

Measurements include the physical subwoofer's fixed gain. Keep that measured
transfer intact when the hardware gain stays unchanged during playback: it
affects both native LFE and redirected main-channel bass. If that hardware gain
already supplies the intended LFE boost, set
`system.bass_management.lfe_playback_gain_db` to `0.0` to avoid another software
boost. This does not declare digital input headroom. A gain applied only by an
AVR's LFE input path is different and must not be treated as a common physical
subwoofer gain.

## Crossover cancellation and baseline acceptance

Stereo systems with subwoofers share the home-cinema bass-management executor.
The workflow announcement identifies the configured layout and physical sub
count. Post-EQ decision logs use the shared `roomeq_workflow::bass_management`
logging target.

An explicit multi-sub `joint_optimization: true` request requires a prepared
seat matrix and verified shared timing evidence. Missing evidence returns an
error without substituting the detailed optimizer. Selecting legacy detailed
processing is a separate request, not a way for joint admission to report success.
Measurement-backed all-pass MSO similarly requires labeled stationary captures
with matching timing references at each seat; phase arrays alone do not authorize
it. Low-level numerical APIs remain separate from measurement-backed admission.

Home-cinema level calibration is independent of the EQ frequency limits. It
uses the shared measured main-speaker passband above crossover transitions,
preferring 500–2,000 Hz with at least one supported octave. Surround and height
speakers need not be full-band. All main speakers share the reference; matching
each left/right pair alone does not establish overall calibration.

Fallback removes level and safety trims derived from discarded correction and
recomputes alignment before electrical safety replay. Rejected or unverified
results cause a nonzero CLI exit and skip external playback export. Native JSON
may remain for diagnosis with a manifest whose status is `rejected`; that file
is not an approved playback configuration.

Final routed quality checks observe each input's measured playback passband,
not just the EQ frequency bounds. A bass-only correction therefore still fails
if a late stage damages the treble. Native LFE is evaluated only through its
low-pass band; a main input includes its redirected subwoofer contribution and
extends through the main speaker's supported passband. The terminal alignment
check replays the final graph instead of accepting an earlier cached success.

Routed snapshots, correction replay, and final crossover validation share one
numeric cancellation policy. `optimizer.max_crossover_cancellation_db` defaults
to 3 dB and accepts any finite nonnegative value. This measures cancellation below
the louder main/sub branch, not deviation below the target curve or clipping.

Within the configured limit plus 0.05 dB numerical tolerance, cancellation passes.
Above it, cancellation must improve by more than 0.05 dB relative to the fixed
configured routing before automatic array/route optimization and EQ. For example,
10→4 dB passes with an `improved_residual_cancellation` advisory, whereas 10→10
and 10→11 dB fail. Each logical input is checked independently. The original
baseline persists through every correction and safety stage; intermediate
improvements never become a new baseline.

Post-EQ decision logs separately compare the immediate input **without** this
pass and the candidate **with** this pass: target shortfall, main/sub cancellation,
combined objective, and main-only score. The frozen configured cancellation
baseline is labelled separately. The reported percentage is the reduction in
the target shortfall measured in dB, not a percentage change in pressure, power,
or perceived loudness. Peak EQ gain and useful-output loss expose additional
trade-offs, and every failed acceptance check is listed.

The optional Post-EQ target-shortfall gate accepts either a result at or below
3.05 dB, or a reduction of at least 20% **and** at least 1 dB against the immediate
input. These are fixed product thresholds, not audibility thresholds. The main
response above twice the crossover frequency must not regress: this check uses
error against the requested target when one is available, otherwise flatness.
When cancellation screening is active, this pass must not worsen cancellation
by more than 0.05 dB against its immediate input. The final routed cancellation
check still uses the frozen configured baseline described above. Common
pre-route EQ can reduce target shortfall while leaving relative main/sub
cancellation unchanged.

Final selection evaluates candidates both with and without an accepted optional
main Post-EQ pass. Removing this pass preserves preceding EQ, routing, and
physical-sub correction. Electrical drive, useful-output, temporal, and retained
seat checks still apply. Passing the relative target rule does not guarantee
that the optional pass will be exported.

Finalization reports candidate counts and elapsed time at info level. During
long searches with warning-only logging, it reports progress at candidate
boundaries about every 30 seconds. Waveform replay uses FFT evaluation on its
exact uniform grid, including FIRs longer than the transform. Electrical grids
use the same acceleration while retaining direct evaluation at extra EQ and
crossover frequencies. Arbitrary grids retain scalar evaluation; safety samples
are never interpolated. Temporal evidence is reused across a safety check
only when the channel state, source curves, and retained FIRs are unchanged.
These optimizations do not reduce the candidate search or relax acceptance.

For multi-position measurements, baseline construction selects
`optimizer.multi_seat.primary_seat` (default 0) from each main and sub source,
matching crossover alignment. Spatial power averages remain magnitude-only
and do not supply phase. An unavailable index in a multi-capture source is
an error; a single-capture source retains its sole capture. Missing usable
phase leaves the coherent baseline unavailable. This reference-seat baseline
does not replace final replay at every retained seat or establish timing
provenance by itself.

When crossover frequency changes, both responses are compared on the same grid
over the union of their half-to-twice-crossover windows within measured support
and 20–2000 Hz. Missing baseline evidence cannot authorize an above-limit result.
`metadata.bass_management.crossover_cancellation` records per-source baseline and
final deficits, worst frequencies, evaluated band, configured limit, improvement,
and acceptance reason. Above-limit acceptance still requires all other acoustic
and electrical checks. See `src/bin/roomeq/INPUT_FORMAT.md` for initialization of
automatic crossover ranges and structured array baselines.

## Final convolution resource integrity

When combined-boost limiting changes a channel's PEQs, its original optimizer
run remains recorded but is no longer marked selected for output. The applied
limiting stage records the gain scale and boost limit separately.

Successful limiting also updates retained biquad coefficients. Serial-channel
IR reports evaluate the final serialized topology, including Kautz/warped
sections, gain, and delay, instead of reconstructing it from PEQ summaries.
Previous waveform and early/late reports are invalidated until they
are rebuilt. If phase evidence is unavailable at refresh, the waveform pair is
absent rather than retained from an older measurement or correction. These are
model-derived waveform reports, not a replacement for measured temporal or
listening evidence.

The model-derived waveform view uses a 65,536-point inverse FFT, applies the
complete serial transfer on its uniform frequency grid, and crops to 400 ms.
Both views share the uncorrected peak reference; the corrected view is not
independently normalized. Missing phase, malformed transfer, or unresolved
resources leave the pair unavailable rather than falling back to an approximate
PEQ chain. This finite-period reconstruction can wrap sufficiently long
responses and is not proof of linear-convolution support or measured decay.
Parallel-driver acoustic reconstruction remains a separate path; a combined
capture is not assumed to be each driver's transfer.

Combined-boost limiting commits a changed channel only after successful DSP
replay. Replay failure retains both its original chain and cached response;
a failed replay is not evidence that the channel meets the boost limit.

When final-seat shape and useful-output checks both fail, the rejection report
retains both violation codes and the error includes both diagnostics. A shape
failure does not suppress evidence of lost output.

Final-seat validation retains each loaded response on its original frequency
grid and support, including phase, coherence, and noise-floor arrays when
present. Loading does not align or clip different seats to a shared grid;
physical-branch summation performs its own supported alignment later. These
parsed responses are not authenticated raw recordings, and this retention
does not establish calibration, timing, or acquisition provenance.

The `final_seat_input_retention` stage retains those exact parsed response arrays
using the workflow intake contracts, including source/driver role, configuration
source key, source-scoped positional indices, declared measurement labels, and
declared provenance. Training and held-out inputs have separate identities and
partitions. Its passing checks mean structural retention only, not acoustic or
physical acceptance. Loaded-response hashes are not raw recording hashes.
At the public workflow boundary, configured measurement responses are frozen
before optimization callbacks run. Numerical loaders for optimization and final
replay reuse those full native responses; original paths, labels, speaker names,
and provenance declarations remain metadata. Changing a CSV after this point
cannot change the run's numerical measurement input. Malformed inline phase
arrays are rejected before freezing rather than silently discarded. Associated
recording WAVs, CEA2034 data, external target assets, and processing performed
after loading are outside this snapshot contract. It is not an atomic acquisition
session or an authenticated recording archive.
The receipt's empty conditioning ledger says snapshot retention applied no
gain, alignment, or calibration; it does not assert that acquisition or other
processing was unconditioned. Unit take weights are storage facts, not optimizer
weights, and no averaging is performed by the receipt producer.

For shared single-channel EQ preparation, optimizer runs now carry optional
`input_normalization` evidence: the signed dB gain actually added to the analysis
curve, reference-selection policy, active correction bounds, and parsed input /
normalized-unsmoothed curve identities. The reference can be a caller-supplied
common reference, a limited-band target-relative reference, or the arithmetic
mean of dB samples in the active correction band. The correction bounds are not
asserted to be the reference-estimation band. Input identity describes the curve
at normalization, which may already have been resampled or otherwise prepared.

`optimizer_input_conditioning` translates recorded offsets into the canonical
gain ledger per channel and optimizer attempt. Repeated preparation identities
across adaptive/refinement attempts are not cumulative gain steps. Historical
normalization remains a conditioning fact even if its candidate was not selected.
The stage is structural, not a safety/acoustic verdict; it lists runs lacking
receipts and is degraded when such runs exist. Missing optional evidence in old
files means unknown, not zero gain. Multi-measurement PEQ and spatial FIR searches
also retain `multi_input_normalization`, one record per actual objective curve.
The population distinguishes aligned measurements, an RIR prototype, bootstrap
resamples, and bootstrap resamples of a prototype. Objective indices are not
physical seat identities. Linear, minimum-phase, and Kirkeby spatial FIR searches
carry these records through the same per-attempt conditioning stage.
Other preprocessing, non-spatial FIR normalization, calibration application, and timing/recentering still
require their own conditioning records; these fields do not claim full coverage.

With verified shared timing, legacy MSO with multiple measured seats retains
one coherent combined response per seat after selected sub gains/delays for
shared sub EQ. The
configured multi-measurement strategy receives those responses. Routing keeps
its separate complex representative for crossover timing. Shared-EQ support
is restricted to common measured support; missing phase or inconsistent
seat identities/counts cannot be replaced by an invented coherent response.
This does not change legacy MSO's primary-seat gain/delay search into a
multi-seat alignment optimizer; use the explicit multi-seat mode for that.

Legacy MSO enables phase-dependent sub delays only when every source has
matching labeled stationary seats and a declared shared timing reference
covering the correction band. Phase arrays without that provenance use
gain-only optimization, with `unverified_timing_gain_only` in advisories;
coherent per-seat shared-EQ responses are unavailable and explicitly marked
`unverified_timing_shared_eq_seats_unavailable` rather than inferred from
unverified relative phase. The original seats remain available for validation.
The final routed crossover and electrical safety checks still apply. This
does not authenticate a synthetic fixture as an acoustic capture or establish
measured playback improvement.

If correction-safety rollback breaks routed playback, the workflow fails.
The restored pre-gate graph is retained only as diagnostic state: restoring
routing does not prove that the rejected correction is safe. No validation
bundle is newly published for this failed workflow.

Final-seat quality rejection sets correction acceptance to `accepted: false`
and `decision: "rejected"`, retaining seat evidence and violation reasons.
The workflow returns an error; this decision does not claim that a rollback or
identity fallback was executed. Consumers of the output decision enum must
handle the additive `rejected` value.

Channel-matching EQ has explicit logical-input (`pre_route`) ownership, so routed
playback applies it to both the main and redirected-bass branches. Deployed-source
curve caches follow the same correction and are restored if matching is discarded.

Routed finalization and direct physical replay reject channel plugins without a
recognized `pre_route`, `post_route` or `route_owned` stage. CamillaDSP's public
export validator already enforces this requirement. Driver plugins are owned by
their explicit physical branch; non-routed chains do not need routing stage tags.

Post-EQ acceptance checks useful-output loss across the representative stage's
declared band, including bass below the mains scoring band. A rejected candidate
is discarded through the existing stage path, preserving previous correction,
routing, gains and delays; its optimizer evidence is not selected for output.
Stage metadata records output-loss rejections. This necessary stage guard does
not replace final native-seat checks for cumulative or position-specific loss.

Routed graphs keep logical source inputs separate from physical output-only sub
channels. Adding independently controlled sub outputs extends the destination
list, not the list of signals to replay or capture. Route source indices address
the logical input list; destination indices address the physical output list.

Enabled validation bundles are written only after final-seat validation succeeds
and final provenance, conditioning receipts, and decision-ledger attachment finish.
They include the workflow sample rate, requested optimizer configuration, final
DSP graph with acceptance/seat evidence and resource identities, and final
combined scores. The embedded graph omits the bundle's own just-created report
pointer. A workflow error does not publish a new bundle; it does not delete an
older bundle already present in a reused output directory. These descriptors
are not rendered, level-matched listening assets or proof of perceptual benefit.

Finalization and packaging require every global, channel and driver convolution
stage to declare a string `ir_file` that is nonblank and contains no NUL byte.
Malformed declarations are errors, not absent resources; valid references to
unavailable files remain explicitly unbound at finalization.

Before binding available resources, finalization decodes every referenced WAV
at the workflow sample rate and requires nonempty, equal-length channels with
finite samples. This applies even without retained coefficient ownership, such
as driver or multi-FIR resources. It validates WAV data, not equivalence to
an accepted acoustic transfer; retained-tap comparisons remain ownership-scoped.

Completed workflows record `metadata.final_convolution_sha256` after the final
DSP stages, using the configured artifact store. Each reference maps to its
SHA-256, or `null` when the resource was unavailable at finalization. Packaged
exports verify this inventory before writing and retain it when sidecars are
renamed. Changed, missing, or unbound resources require a new validated workflow
result; exporting must not silently assign replacement files a new identity.
The export API without sidecar verification rejects bound convolution graphs.

Older/manual graphs without this optional field remain readable and exportable,
but do not carry this final-workflow integrity guarantee. The inventory records
byte identity, not acoustic acceptance, audibility, or protection against an
actor who edits both the file and its recorded digest.

Retained channel FIR coefficients are associated only with that channel's
single declared channel-level convolution reference. A driver's separate FIR
must use its own sidecar; it is neither compared against unrelated parent taps
nor replaced with them when missing. This ownership check does not infer
coefficient ownership for arbitrary multi-FIR chains.

## CamillaDSP delay realization


CamillaDSP exports preserve integer delays in samples. Fractional delays use
a causal FIR rather than silently rounding to whole samples. Every parallel
branch receives the same required stage padding, preserving relative timing.
The YAML `roomeq_delay_realization` declaration records per-stage padding,
total additional latency, sample rate, and usable frequency band. The same
typed report is available through the export API and packaged artifact; the
workflow logs it when fractional-delay processing is present.

The fractional-delay kernel is specified through 0.46 × sample rate, with
at most 0.01 dB magnitude error; Nyquist is not certified. Added backend
padding is **in addition to** requested delays, existing correction-FIR
latency, and device buffering. It is not evidence of acoustic improvement,
transient headroom, or listener preference. The required matrix backend QA
checks sampled electrical transfer independently of optimization scores.

### Physical-routing API

`roomeq-model::PhysicalRoutingGraph` is the backend-neutral contract for resolved
input-to-physical-output routing. Input plugins run before fan-out; every route
retains its own gain, polarity, crossover family/frequency and delay. Contributions
sum at each physical output before its ordered output plugins run. Routes may be
canonicalized for deterministic serialization, but distinct transfers are never
merged solely because they share an input. Port names are graph identities, not
sound-card assignments.

`roomeq-engine::physical_routing::resolve_physical_routing` translates supported
legacy home-cinema channel/driver ownership into this contract. Shared sub EQ is
assigned to each physical sub, while driver gain/delay/polarity already included
in route values is not applied again. Unresolved ownership is an error. Electrical
diagnostics and the sibling SOTF native adapter use this resolver. Native lowering
preserves signed gains and uses explicit wet-only, zero-feedback alignment delays.
Unsupported crossover families are rejected rather than replaced with LR24.
Local PCM regressions cover integer delays, LR24 crossover transfer, shared EQ,
and route-order invariance at 44.1/48/96 kHz and multiple block sizes. Deployment
still requires the matching AutoEQ dependency revision. Arbitrary fractional-delay
and FIR conformance, device routing, acoustic improvement, and transient/continuous
headroom are not certified by those tests or by model replay alone.

This adds a library contract, not a new RoomEQ input configuration field or a
replacement of the existing exported `DspGraph` JSON schema. AutoEQ has no new
dependency on SOTF's player, engine, or plugins.

## Features

- **Single speaker optimization**: Optimize EQ for individual speakers
- **Multi-driver crossover optimization**: Optimize crossovers for multi-driver speakers (woofer + tweeter, etc.)
- **Group delay alignment**: Optimize time alignment between subwoofers and main speakers
- **Multiple optimization algorithms**: Support for COBYLA, Differential Evolution, and other optimizers
- **AudioEngine-compatible output**: Generates JSON DSP chains compatible with the AudioEngine plugin system
- **Target curve tilt**: Harman-style tilted target curves with optional bass shelf
- **Excursion protection**: Automatic F3 detection and highpass filter generation
- **Schroeder frequency split**: Separate EQ strategies for modal and statistical room behavior
- **Phase alignment**: Subwoofer/speaker phase and polarity optimization
- **Multi-seat optimization**: Minimize response variance across multiple listening positions
- **Supporting-source room compensation**: Delayed, decorrelated supporting loudspeaker to fill reverberant energy while preserving the primary source's direct sound

## Usage

```bash
cargo run --features cli --bin roomeq -- --config <config.json> --output <output.json> [OPTIONS]
```

### Options

- `--config <CONFIG>`: Path to room configuration JSON file (required)
- `--output <OUTPUT>`: Path to output DSP chain JSON file (required)
- `--sample-rate <RATE>`: Sample rate for filter design (default: 48000 Hz)
- `--freq-samples <N>`: Number of log-frequency points used when reducing dense
  measurements for interpolation (default: 200)
- `--export-format <FORMAT>`: Also export `camilladsp`, `apo`, `easyeffects`,
  `wavelet`, `pipewire`, `roon`, `rew`, or `coefficients`
- `--export-path <PATH>`: Override the derived external-export path
- `--convert <DSP_CHAIN_JSON>`: Convert an existing RoomEQ DSP chain without
  running optimization
- `--verbose`: Enable verbose output
- `--help`: Print help information

`rew` emits a single-channel REW Generic EQ filter-settings file.
`coefficients` emits normalized `a0=1, a1, a2, b0, b1, b2` sections using the
same canonical biquad implementation as runtime DSP. Both formats reject
convolution, crossovers, routing, or unknown stages instead of silently
dropping them.

### Output bundle layout

A run writes a small DSP JSON plus a sibling assets directory named after
the output stem: `--output dsp.json` produces `dsp.json` and `dsp_files/`
(generally `<stem>_files/`). Nothing is written to the process working
directory. The assets directory holds every run-generated file:

- convolution FIR WAV sidecars (`ir_file` references stay bare filenames
  and resolve against the assets directory, with the JSON's parent
  directory as a legacy fallback);
- extracted measurement curves as CSV (`<channel>__initial.csv`,
  `<channel>__final.csv`, `<channel>__eq.csv`, `<channel>__target.csv`,
  `<channel>__pre_ir.csv`, `<channel>__post_ir.csv`,
  `deployed__<channel>.csv`, per-driver curves) plus
  `measurements_index.json` mapping each channel/field to its file;
- diagnostic grids (`<channel>__waterfall.json`,
  `<channel>__wavelet.json`, `<channel>__early_late_curves.json`,
  `<channel>__resonance_decays.json`) when the run produced them;
- `manifest.json` (run status and exact asset ownership) and `roomeq.log`
  (run summary lines; detailed logs remain on stderr via `RUST_LOG`).

### HTML report format

The report's waveform availability distinguishes an imported room IR from a reconstructed
input response and a predicted post-DSP response. Post-DSP level offsets include
headroom attenuation; the separate balance column compares final monitor levels.
Landmark plots follow the landmark table, and arrival plots follow the timing
table. The symmetric-monitors tab uses Rust-exported magnitude sums over
overlapping frequency support, not phase-coherent acoustic sums. Existing
bundles can refresh this
derived report data without rerunning optimization:
`cargo run -p roomeq-workflow --example refresh_report_pairs -- path/to/dsp.json`.
Then regenerate the HTML with `ui/display-roomeq` and the intended `-o` path.

`utils/mdat2csv.py measurements.mdat output_dir` exports
frequency-response CSVs and supported measured impulse responses as
`<measurement>__ir.csv` (`time_ms,amplitude`). Install `scripts/requirements.txt`
for the Java serialization decoder. Native IR amplitudes, sample intervals and
start times are preserved. Add `--timing-reference-id session-1` to declare a
shared acquisition timing reference and include the IRs in `recordings.json`
for room-acoustic analysis. Without that declaration, the IR files are exported
but not bound into the configuration. Unsupported legacy/derived IR storage is
reported explicitly.

Parent configurations can declare native measured IRs directly, including
3 kHz subwoofer captures. Use `output_channel` and `driver` to attach a capture
to its exact delivered driver, not the summed parent speaker. Independent
acoustic diagnostics can omit unknown timing IDs; this does not establish a
shared clock for sums or alignment. Driver IRs and diagnostics are exported
as acoustic sidecars and shown on their report tabs, with bands above Nyquist
unavailable. See [measured-case imports](MEASURED_IR_IMPORT.md) and the
[input format](../src/bin/roomeq/INPUT_FORMAT.md) for mappings and details.

`ui/display-roomeq room-output.json -o room-report.html` writes a
self-contained HTML file: the plots render from an embedded versioned JSON
payload (`autoeq-report-data-v1`, documented in
`crates/autoeq-report-wasm/SCHEMA.md`) with a WebAssembly 2D canvas renderer,
so the file needs no network access and no Plotly install. The title line
carries a renderer status at its end, with a provenance line above it naming
the RoomEQ version, the DSP data date (`metadata.timestamp`), and the report
render date. WebGPU is selected automatically when
an adapter and device are available; dense waterfall triangles are batched on
the GPU. The report layout, d3rs/WASM geometry, axes, legends, colors, and
rotation controls are shared with the Canvas fallback. There is no separate
GPUI view. Missing adapters, initialization failures, or device loss fall back
without changing the report content. Append `?renderer=canvas` to a served
report URL to compare the portable backend for diagnostics. Surface projection
and depth sorting still run in WASM; rasterization is GPU-accelerated.
The playback verdict stays pinned at the top, followed by three centered
tabs — DSP analysis (why this correction, DSP signal flow), Acoustics
analysis (summary, speakers, time of flight, symmetric monitors, time
domain), and Psychoacoustic report (EPA scores). Inside a tab the flow is
flat: multiple subheads open as tabs reusing the channel-tab styling (a
lone subhead renders directly), per-channel sections open under shell
tabs placed below the summary sections, and every table or graph sits in
exactly one card. Legends toggle series by click. The combined overview is three
stacked panels sharing the frequency axis — Before EQ (all channels plus
dotted targets), EQ (all shaping curves), Corrected (all post-DSP channels
plus dotted targets) — with the Before/Corrected panels on one SPL range.
**All EQ Filters** lists every channel behind a channel button set. A
frequency-landmarks table reports per-speaker peaks, notches and −6 dB LF
extension from the measured response; level residuals are predicted from the
post-DSP curve. The summary operational share reads emitted
`channel_summaries` when present and otherwise derives the same share from
legacy final `decisions` (provisional history never enters the scope).
Rebuild the embedded bundles with `just report-dist` (stable) and
`just report-dist-gpui` (nightly) after touching the renderer crates.

Explicit `--export-path` / `--verification-bundle` artifacts go where
requested (an `--export-format` default lands next to the JSON) and are
tracked in `manifest.json` asset ownership.

The saved JSON keeps the exact `DspGraph` schema with DSP data (plugins,
metadata, decision ledger) but without the heavy measurement blobs, so it
stays small. The Python viewer (`ui/display-roomeq`, via
`ui/display-roomeq/loaders.py`) re-injects the external curves from the sibling
directory automatically, so plots are unchanged. Legacy outputs with
embedded curves and sidecars next to the JSON keep loading as before.

### Interactive diagnostic plots

The room-mean T60 graph overlays the ITU-R BS.1116-3
§8.2.3.1 Figure 1 tolerance envelope; channel tabs reuse that block (its
table already carries per-speaker columns) while per-driver tabs keep
their own measured-acoustics T60. Its reference is
`0.25 * (volume_m3 / 100)^(1/3)` seconds when the saved effective configuration
contains valid `recording_config.room_dimensions`. Otherwise, the viewer uses
the measured 250–4000 Hz octave centers as an estimate of the 200 Hz–4 kHz mean,
averaging only channels with all five valid midband fits. That fallback is
explicitly labelled as a relative-shape reference, not volume compliance.
Without either reference no limits are invented. The upper curve starts at
63 Hz, the lower at 100 Hz, both change tolerance at 4 kHz and stop at 8 kHz.
The existing fixed-window summary score is separate from this recommendation.

RoomEQ report graphs use d3rs through the shared WASM renderer, including
capture diagnostics. Waterfalls default to REW-style time-slice contours with
frequency, time and relative-level axes; the toolbar offers a filled-surface
toggle, a d3rs colormap dropdown (Turbo, Viridis, Plasma, Inferno, Magma,
Rainbow) and a contour toggle, and coloured ridge lines trace the exported
resonance candidates over the slices. Wavelets use logarithmic frequency horizontally,
time vertically (initially -1 to 15 ms), and a labelled -30 to 0 dB colour scale.
A cyan dashed ridge marks peak energy arrival time per frequency column
(earliest time holding the column maximum). Mouse wheel zooms at the cursor,
plain left-drag crops a zoom box on line plots, Shift/right-drag pans,
double-click resets the view, and the 20–200 / 20–20k overlay presets jump the
frequency axis; the Move toggle keeps drag-to-pan for touch. Smoothing applies
to eligible frequency-response line plots, not time traces or diagnostic grids:
1/48 through 1/1 octave plus REW-style Var (1/48 below 100 Hz to 1/3 above
10 kHz), Psychoacoustic (1/3 below 100 Hz to 1/6 above 1 kHz, cubic-mean peaks)
and ERB (±half equivalent rectangular bandwidth) modes.

Every SPL-vs-frequency plot spans 50 dB vertically with a horizontal gridline
every dB (labelled every 5 dB). Comparison-report design targets are
level-matched once per report: the main L/R target shape is shifted so its
100 Hz–10 kHz mean equals the measured L+R pair mean over the same band, and
that single offset applies to every channel tab (the L+R tab reuses the main
shape on its own grid), preserving designed inter-channel differences.

The waterfall and decay plot highlight up to two exported resonance candidates,
ranked by longest fitted decay then highest level. Their colours match the final
all-speaker table. These are detected candidates, not confirmed geometric room
modes. Levels retain each grid's own peak reference; they do not compare absolute
speaker output. No fitted line is invented when the export lacks its intercept.

Rendering cannot recover discarded samples: sparse exported early-time wavelet
samples or missing low-frequency waterfall bins remain sparse. The wavelet panel
reports its displayed sample count. Regenerating HTML does not recompute IR data.

### Reading the playback summary

`metadata.playback_summary` is the claim-level answer to "what did this run
ship, and what does it cost". It restates the acceptance report without new
judgment: the shipped outcome, how many training seats improved beyond
uncertainty (`training_seats_improved/total`), the worst seat (report it next
to the average), the modeled latency and headroom cost, the correction
family actually present in the graph (`realized_processing`, which can differ
from the requested mode — see `processing_fallback`), and the limits in
force. `headlines` renders these as one deterministic sentence per outcome;
rejected and unchanged runs name their violations there.

EPA numbers (`epa_per_channel`, `epa_multichannel`,
`perceptual_metrics.epa_preference_delta`) are configured model predictions
from frequency response, labeled as such by `metadata.epa_provenance` — not
measured audibility. EPA preference can fall while the acceptance metric
improves: acceptance measures target-weighted RMS shape improvement, while
EPA weights loudness, sharpness, and roughness dimensions that move with
level balance and tilt. When they disagree, the correction changed something
one objective weights and the other does not. Neither number proves
audibility; see `docs/ROOMEQ_LISTENING_PLAN.md` for the claim wording each
evidence level supports.

### Playback verification (operator captures)

The following flags make capture verification usable from the binary; the same
functions back the library helpers, and every check below refuses loudly
instead of comparing mismatched evidence.

- `--verification-bundle <DIR>` (runs after optimization and save):
  writes `<DIR>/verification-bundle.json` for the finalized graph.
  Requires `--baseline-graph <FINGERPRINT>` (16 hex chars from the
  referenced run), `--calibration-id <ID>`, `--stimulus-hash <HASH>`,
  and `--verification-seats <SEATS>` (comma-separated seat IDs). Sources
  come from the graph's channel names; convolution sidecars resolve
  against the sibling assets directory (legacy: the output directory) and
  are hashed with tap counts. Driver-
  level convolution requires the explicit prediction handoff described below.
  The bundle is always small-signal:
  limiter trials need a separate operator protocol and bundle.
  Channel `delay_ms` is the sum of serial explicit delay plugins, not total
  FIR/driver/limiter or routed-system latency. Malformed, overflowing, or
  branch-local delays that cannot be represented by that scalar are refused.
- `--verify-captures <MANIFEST>` (standalone, needs no
  `--config`/`--output`): imports the operator manifest, decodes every
  capture file (corrupted takes rejected), and — with `--coverage-plan
  <BUNDLE_JSON>` — checks the manifest trial against the plan, the
  required source/seat coverage, the plan's candidate graph and stimulus
  binding, and capture sample rates. `--expected-graph` /
  `--expected-stimulus` check identities without a plan. The report goes
  to `--verification-report <PATH>` (never a raw take, manifest, or plan).
  Without numerical predictions the import stays `insufficient_evidence`.
  Plans with `ir_comparisons` additionally run the declared IR comparison
  described below. Synthetic captures stay explicitly synthetic. The binary
  never starts playback or recording.

To generate numerical `ir_comparisons`, add
`--verification-prediction-inputs <JSON>` to bundle generation. You can also
generate from an already saved native graph without running the optimizer:

```bash
roomeq --verification-graph final.json --verification-bundle new-bundle \
  --verification-prediction-inputs plant-inputs.json --sample-rate 48000 \
  --baseline-graph 0123456789abcdef --calibration-id session-calibration \
  --stimulus-hash STIMULUS_SHA256 --verification-seats MLP
```

The handoff is a `physical-ir-prediction-v1` JSON document. Required fields:

- `capture_plane`: exactly `unit_physical_output_transfer_after_serialized_dsp`.
  Each IR describes the external plant per unit physical digital-output input,
  **after all DSP represented by the saved graph**. An already-corrected or
  baseline whole-chain IR is not interchangeable: it would apply processing
  twice. Required external driver protection remains part of the plant, unchanged
  during acquisition and later playback. Do not bypass necessary protection to
  produce these files; unsupported capture planes require a different handoff.
- `sample_rate_hz`, `settings` (`method: full_ir_dtft_v1`, `calibration_id`,
  `timing_reference_id`, `magnitude_offset_db`), `frequencies_hz`, `band_hz`,
  `max_capture_samples`, `alignment`, `tolerances`, and `synthetic`.
  Grid endpoints must equal the band; samples/rate/settings are not resampled,
  recentered, independently normalized, or fitted. `alignment` and `tolerances`
  use the existing IR-comparison fields documented below.
- `output_assignments`: for independent channels, objects with `channel`,
  nullable `driver`, and unique physical `output`. Every driver needs its own
  assignment. For bass-managed graphs this array is empty: serialized routing
  owns the physical-output identities and pre/route/post processing.
- `captures`: exactly one object for every physical output at every requested
  seat, with `output`, `seat`, `file`, `file_sha256`, `valid_band_hz`, `settings`,
  and a nonempty `protection_chain_id`. Paths resolve relative to this manifest.
  Mono WAVs must retain common timing/gain references and usable support across
  the entire requested band. Calibration/protection declarations are operator
  supplied, not independently authenticated by the software.
- `coherent_trials`: additional objects such as
  `{"source":"mono","inputs":{"L":1.0,"R":1.0}}`. Gains are signed linear
  amplitudes of simultaneous coherent inputs, not dB or energy weights. All
  isolated logical-input trials are generated automatically. Capture the named
  additional trials too; they become required `source` IDs in the bundle.

The result retains plant-manifest and DSP-resource hashes, declared acquisition
facts, trial inputs, and generated complex-response predictions. Synthetic plant
inputs force simulated evidence even if subsequent files claim acoustic origin.
Generated bundles also carry the processing graph (without decision-ledger
metadata) and a `sha256-json-typed-v1` payload binding. Capture import verifies
that binding, graph identity, resource-list consistency, and agreement between
plan settings/evidence kind and acquisition declarations before comparison.
Missing or modified binding data is rejected. The digest detects stale/edited
artifacts; it is not a signature or independent authentication of an operator's
claims. Legacy manually supplied comparisons remain operator-declared plans.
Numerical verification reports retain the exact coverage-plan file SHA-256.
No stimulus is played and no recording is started. Existing bundle destinations
and report destinations are refused, including symlinks/hard links to existing
files. Choose a new report path for each verification attempt. The current
direct-DTFT implementation has a total 16,777,216-work
budget and a 1,048,576-sample per-IR ceiling; it refuses oversized work without
truncation. The grid must resolve the declared maximum capture duration, which
must contain known plant/FIR/delay latency. This is not a guarantee about infinite
IIR tails, acoustic decay completeness, maximum output, or audibility.

Exit codes: `0` successful standalone bundle creation (not playback approval),
an accepted outcome (optimization), or a complete passing
comparison of operator-declared acoustic IR captures; `1`
the procedure ran but nothing is approved (rejected outcome,
validated-but-unapproved import); `2` rejected operator input (bad
manifest, stale identity, missing coverage, corrupted capture,
unmatched settings or trial type, missing required verification flags).

#### Declared calibrated IR comparisons

The coverage-plan JSON can carry a top-level `ir_comparisons` array with
exactly one entry per required source/seat pair. Each entry contains:

- `source`, `seat`, and `prediction` (`freq` in Hz, `spl` in calibrated dB,
  and `phase` in degrees; retain absolute reference, not a display-normalized curve).
- `settings`: `method: "full_ir_dtft_v1"`, `calibration_id`,
  `timing_reference_id`, and `magnitude_offset_db` (added to the IR's
  `20 log10(|H|)`, never fitted).
- `band_hz: [low, high]`, `alignment: {"gain_db": ..., "delay_ms": ...}`,
  and `tolerances` containing `max_magnitude_deviation_db`,
  `max_timing_error_ms`, and `max_output_loss_db`.

Every capture-manifest entry must then include `ir_analysis` with
`file_sha256` (exact WAV bytes), identical `settings`, and `valid_band_hz`
covering the whole planned band. This mode accepts already acquired/deconvolved,
calibrated mono IR WAVs at the planned sample rate—not raw sweep recordings.
It applies no normalization, recentering, resampling, or fitted gain/delay.
Missing prediction phase cannot yield complete verification.

For matched before/after IR and step diagnostics, `ir_analysis` may additionally
contain `baseline` with `path`, `file_sha256`, `graph_id`, `source`, `seat`,
`stimulus_hash`, `settings`, `valid_band_hz`, and `synthetic`. Its graph must equal
the bundle's baseline graph. Source, seat, stimulus, settings, support, sample
rate, and observation length must match the candidate. A relative baseline path
resolves beside the candidate WAV, not beside the coverage plan. Both raw files
are protected against report overwrite. Omission remains supported and yields
an explicit unavailable IR/step view.

The verification report includes `comparisons[].capture_views` and a typed-JSON
SHA-256 integrity binding. The canonical IR/step view retains both file hashes,
sample-zero timing reference, declared stimulus identity, sample rate, settings identity, and unmodified WAV
amplitudes; the step is the discrete cumulative sum. These are raw sample units,
not an inferred pascal/SPL scale. The declared magnitude offset is retained but
not applied to these raw traces. Pairs containing a synthetic capture stay
synthetic. Matching operator declarations is not authenticated acquisition.
Records over 65,536 samples yield an unavailable raw trace display rather than
truncated data; the compact octave T60 diagnostic can still be calculated from
the complete record. Matched octave ETCs are available when the declared capture support includes
500/sqrt(2) through 4000*sqrt(2) Hz below Nyquist, and both records include the
complete 0–40 ms window plus filter lookahead. The method uses finite Hann-windowed
analytic sinc filters, four center-frequency periods on each side, with
zero-extended **linear** convolution. FFT padding is at least record length plus
filter length minus one, and inverse scaling is 1/N. The known filter indexing
delay is removed, not the capture's timing. Filter skirts and symmetric time
spreading remain; the first half-support interval is affected by zero extension
before record start. Do not interpret that analysis spreading as room pre-ringing.

Both ETC traces use the same per-band baseline peak, retained in raw amplitude
units, so a playback level change stays visible. The -160 dB display floor is
explicit, not a measured noise floor or audibility threshold. The method, support,
window, and reference travel in the bound view payload. The report shows 500,
1000, 2000, and 4000 Hz bands separately. Missing support, insufficient lookahead,
or a silent baseline yields unavailable rather than an invented trace. These
views do not establish left/right similarity. Ambient noise needs the separate
calibrated recording described below. Full physical headroom remains unavailable
on this path; an ETC does not supply missing calibration or capacity evidence.

Matched decay diagnostics require an additional explicit `ir_analysis.decay`:

```json
{"noise_window_ms": [900.0, 1200.0], "minimum_fit_margin_db": 10.0, "minimum_r_squared": 0.95}
```

These example budgets are engineering choices, not audibility thresholds or
universal room-acceptance defaults. The operator declares that this interval
contains stationary, signal-free noise in **both** captures; that declaration
is not authenticated by a file hash. The record must contain the entire window.
Nominal octaves are selected from 63, 125, 250, 500, 1000, 2000, and 4000 Hz
inside the declared usable band. Each uses the same finite analytic filters
described above. Noise power is estimated only from the window interior with
full filter support; reverse energy integration stops one filter half-support
before the window begins. No unobserved tail is appended and no noise is
subtracted. Unsupported individual bands remain explicitly unavailable.

The bound payload retains the raw analytic-envelope-squared energy reference,
separate noise estimates, analysis support, common-reference dB tails, and
individually normalized tails. It is not calibrated acoustic energy or ambient
SPL. Common-reference traces preserve level changes; normalization cannot turn
lower output into a faster decay. A silent candidate has no normalized trace.
The -160 dB floor affects rendering only, not slope fitting.

A -5 to -25 dB least-squares slope may be extrapolated to 60 dB only when the
observation spans that interval, at least three fit samples exist, the estimated
remaining signal/noise energy meets the declared margin at every fit sample,
and R² meets the declared budget. Fit duration, actual span, sample count, R²,
and minimum estimated margin are retained. An exactly zero noise estimate has
no finite margin value; it does not prove noiseless acquisition. Failed fits
remain unavailable while valid traces are retained. This finite-window T20
diagnostic is **not passive-room RT**, an ISO-certified measurement, a safety
gate, or evidence of perceptual benefit; filter spreading, record truncation,
and the declared noise assumptions still matter. Missing `decay` settings keeps
legacy input readable and decay explicitly unavailable.

When a matched baseline/candidate IR pair is supplied, the verification report
also carries an advisory nine-band `octave_t60` diagnostic (63 Hz–16 kHz).
`math-rir` applies octave filters, automatic per-band noise cutoff, and a
Schroeder slope: T30 is preferred and T20 is the fallback. EDT-only estimates,
poor fits, filter-ring-limited estimates, bands above Nyquist, and bands whose
full octave extends beyond the declared usable capture band remain unavailable
with a reason. The HTML view retains gaps and shows both fits and their R².
This automatic diagnostic has an algorithmic fit threshold of 0.90; it does
not replace the separately declared noise-window and fit budgets above,
prove a passive-room damping change, certify an ISO measurement, or affect a
playback gate. The capture pair's synthetic or operator-declared status and
the existing payload binding apply to the new view as well.
When two or more distinct sources at one seat share the same declared
stimulus, baseline graph, sample rate, settings, usable band, and capture evidence kind,
the HTML capture report
also shows their room-mean octave T60 diagnostic. Each band averages only
accepted T30/T20 fits and displays the contributing source count for
baseline and candidate separately. Missing fits remain gaps, and differing
settings or capture kinds are never pooled. This is an advisory capture
summary, not a passive-room damping or playback verdict.

The same pair can provide an advisory STFT waterfall when each IR contains
the complete 500 ms post-peak analysis interval plus the 16 ms Hann half-window.
The bound `capture_views.waterfall` payload carries separate baseline and
candidate grids, each with at most 100 time frames and 64 frequency bins inside
the declared capture band. It uses a 32 ms window, 2 ms hop, and a −5…500 ms
axis relative to each broadband absolute peak. Each grid's dB values use its
own full-grid peak; they cannot show an absolute between-capture output change.
The HTML report projects frequency, time, and relative level as an oblique
wireframe and shows the sampled time trace for each resonance picked at 60 ms,
alongside the fitted 20–200 ms decay time. Peaks are diagnostic candidates,
not confirmed room modes or evidence of changed passive-room damping. A short
or unsupported pair stays explicitly unavailable. This capture verification
field is separate from the optimization DSP JSON's per-channel `waterfall`.

The same complete pair can also emit `capture_views.wavelet`: a complex Morlet
transform with three cycles, six logarithmic centers per octave, a 1 ms raw
frame hop, and at most 64 frequency × 100 time display cells. The view is
restricted to declared capture support, with −30…0 dB blue-to-red colors
relative to each capture's own full wavelet-grid peak. The implementation
normalizes to the grid peak after its wavelet response calculation, so these
colors are not calibrated SPL or absolute between-capture levels. The
time-frequency blur and boundary support affect apparent early energy; the
heatmap alone cannot identify a physical reflection path. This verification
field is separate from the optimization DSP JSON's per-channel `wavelet`.

The same matched IR pair also yields an advisory 1–8 kHz early-reflection
table when the declared usable capture band contains that whole range and
both IRs have a detectable direct sound. Candidates within 15 ms and at
least −15 dB relative to the filtered direct peak are listed separately for
baseline and candidate. The table gives delay, extra path length at 343 m/s,
first comb-dip frequency, and estimated peak-to-peak ripple. A gain at the
direct level has an unbounded idealized ripple estimate, shown as unavailable.
These are finite-window band-limited peaks, not verified geometric paths or
an audibility result; their gain is relative to the direct peak, not an
absolute microphone dBFS level. Missing 1–8 kHz support leaves the view
unavailable.

The matched pair can also emit `capture_views.early_late_curves` when both
recordings have a direct reference, a complete 20 ms early segment, a
nonempty late segment, and at least two fully supported third-octave bands.
Full, early, and late are energy contributions on the same full peak-band
reference **within each capture**; full is their incoherent energy sum, not
a complex pressure sum. Both this view and the optimization per-channel
figure refuse data where full sits below either part in any band and leave
the cell pending instead of plotting impossible physics. The split is 20 ms
after the broadband envelope
peak, or after the 120 Hz lowpass envelope peak for subwoofer/LFE sources.
The report labels the separate baseline/candidate references and shows a
1–8 kHz early-minus-late mean only when that complete range is supported.
It does not establish absolute between-capture level or a passive-room
damping change. This bound capture view is separate from the optimization
DSP JSON's per-channel `early_late_curves` field.

Calibrated ambient noise uses an optional **separate silent-playback WAV**, not
an IR tail or the IR magnitude calibration offset. Supply `ir_analysis.ambient_noise`:

```json
{
  "path": "silent-playback.wav",
  "file_sha256": "<SHA-256 of noise WAV bytes>",
  "calibration_path": "pressure-calibration.json",
  "calibration_sha256": "<SHA-256 of calibration JSON bytes>",
  "graph_id": "<candidate graph identity>",
  "source": "<comparison source>",
  "seat": "<comparison seat>",
  "acquisition_gain_id": "<fixed microphone/interface gain and path identity>",
  "playback_state": "silent",
  "conditions": "<stationary microphone, orientation, room/HVAC/equipment state and acquisition conditions>",
  "synthetic": false,
  "settings": {"frame_samples": 4096, "valid_band_hz": [20.0, 20000.0]}
}
```

Paths are absolute or relative to the associated candidate WAV directory.
`silent` declares that the stimulus is off while the stated playback path and
equipment conditions are retained; it is not verified automatically. The mono
WAV must have the plan's sample rate and contain at least one complete FFT frame.
The current decoder limit is 1,048,576 samples; larger recordings are rejected,
not shortened. Both the noise WAV and calibration resource are hash checked and
protected against report overwrite, including hard-link aliases.
Bytes identical to the imported candidate or baseline IR cannot be relabeled as
noise, even under another filename. This check does not authenticate other
recordings or detect arbitrary edited copies; capture conditions remain declared.

The calibration resource is strict JSON with these fields (values below are a
**synthetic numeric example, not a microphone calibration or recommended gain**):

```json
{
  "calibration_id": "synthetic-pressure-example",
  "pascals_per_sample": 2.0,
  "microphone_id": "synthetic-microphone",
  "orientation": "synthetic-omnidirectional",
  "acquisition_gain_id": "<same gain/path identity as the handoff>",
  "reference_conditions": "synthetic numeric oracle, not hardware calibration",
  "response_freqs_hz": [20.0, 20000.0],
  "response_correction_db": [0.0, 0.0],
  "uncertainty_db": null,
  "self_noise_note": "not characterized; synthetic example"
}
```

For real captures, the numeric Pa/full-scale-sample sensitivity must be supported
by the actual microphone/interface gain and reference calibration. Record the
reference source, level, gain, conditions, and individual microphone orientation.
Frequency-response values are **corrections to add**, applied once by linear
interpolation in log frequency. They must cover the entire declared usable band;
no extrapolation is performed. An all-zero response table explicitly declares
a flat correction and does not establish that a real microphone is flat.
`uncertainty_db: null` means unspecified, not zero uncertainty. The microphone
and interface self-noise remain part of the observed signal; no subtraction is
performed. Numeric values and hashes do not authenticate acquisition/calibration.

The estimator `periodic_hann_welch_pressure_v1` uses power-of-two frames of
256–65536 samples, periodic Hann windows, per-frame mean removal, 50% nominal
overlap, and one-sided linear-power averaging. The last frame is anchored at
the record end if necessary; frame starts are retained. This covers the record
without truncating its tail but is not a uniform-time-weighted Leq measurement.
There is no zero padding. For the unnormalized forward FFT, PSD normalization
is `fs * sum(window²)`, with an interior-bin factor of two and no doubling at
DC/Nyquist. DC is not reported. Bin spacing is `fs/N`; Hann equivalent noise
bandwidth is `1.5 * fs/N`. Neither proves sufficient resolving power or duration.

The bound view retains positive-frequency pressure PSD in Pa²/Hz and nominal
octave bin sums in unweighted dB SPL re 20 µPa, with actual summed-bin centers,
nominal edges, calibration, and observation settings. Only fully supported
octaves with at least three bin centers are shown. These are FFT-bin sums, not
certified octave-filter outputs. Window leakage limits band isolation. Zero
estimated power has no finite SPL; it is not replaced with a floor or proof of
a noiseless room. Missing/unsupported inputs produce explicit unavailable views.
Malformed identity/hash declarations reject the import. No A/C weighting,
NC/NR rating, psychoacoustic loudness, stationarity, or acoustic acceptance is
inferred. Noise reporting is independent of the presence/display limit of a
baseline IR, and synthetic noise remains labeled synthetic.

Render these views at the top of a separate capture diagnostic report:

```bash
venv/bin/python ui/display-roomeq --capture-verification verification-report.json -o captures.html
```

To include the same measured-capture section after the plots in one saved
optimization report, supply both artifacts:

```bash
venv/bin/python ui/display-roomeq room-output.json --capture-verification verification-report.json -o room-report.html
```

The combined renderer recomputes the saved optimization payload binding,
checks referenced FIR resource bytes, and requires the verification candidate
graph ID to equal that bound graph identity. If either check fails, the capture
section states why it is unavailable. The section retains every source/seat
and synthetic/operator declaration and labels the verification report's
recorded accepted/unchanged/rejected/insufficient-evidence status. Displaying
diagnostics does not upgrade a rejected result. It does not turn a reconstructed display
IR into a room measurement or upgrade a playback verdict.

The capture-only destination must be new. The renderer validates the view payload and
graph/source/seat binding before plotting, labels evidence kinds explicitly,
and does not promote the imported comparison into an independent safety or
listening verdict. Comparison mode does not accept a capture-verification
attachment.

The complete IR is evaluated with the existing direct Fourier kernel. Limits
of 1,048,576 samples and 16,777,216 sample-frequency operations bound work;
oversized data is refused, never truncated. Adjacent prediction bins must
satisfy `2 * IR_span_seconds * frequency_gap_hz < 1` to avoid ambiguous
bulk-delay unwrapping. These are implementation limits, not audibility limits.

Reports retain each route's magnitude, useful-output and timing metrics,
explicit budgets, evidence class, capture hash and comparison-plan hash.
A failed or incomplete required route cannot be hidden by a later passing
route. Passing synthetic fixtures stay `insufficient_evidence` (exit 1).
Acoustic classification depends on the operator's declaration; the program
does not independently authenticate acquisition provenance. A passing result
does not establish maximum output, physical safety, preference, or listening benefit.

Without `--verification-prediction-inputs`, bundle generation remains a manifest
and resource snapshot with no numerical comparison. With the supported handoff,
it generates `ir_comparisons` for every required trial/seat; never duplicate one
seat's curve across other seats. General sweep analysis, other capture planes,
larger FFT-based prediction workloads, independent backend replay, and
dynamic/limiter assessment remain separate unfinished roadmap work.

## Choosing a DSP Target

RoomEQ has a canonical JSON output and several external exporters. The
canonical JSON is the most complete description of the result; an external
export is a translation into the target's DSP model, not a guarantee that every
RoomEQ feature remains representable.

| Target | Best use | Preserves | Important limitations |
|--------|----------|-----------|-----------------------|
| **RoomEQ JSON / AudioEngine** | Full-fidelity RoomEQ or an AudioEngine-compatible host | Per-channel gain/EQ/delay, multi-driver crossovers, FIR convolution, mixed phase, global matrices, routed bass management, and route metadata | The consumer must implement the RoomEQ output schema, plugin types, channel ordering, sidecar FIR files, and graph routing. |
| **CamillaDSP (`camilladsp`)** | Multichannel playback with subwoofer/bass-management routing | Routed bass-management graphs, channel mixing, serial filters, and convolution sidecars when the generated paths are available | Requires correct channel numbering, sample rate, sidecar WAV paths, and CamillaDSP configuration. Unsupported RoomEQ plugin features or graph details cannot be carried over automatically. |
| **Equalizer APO / Peace (`apo`)** | Windows playback with serial filters and representable channel routing | APO-compatible gain/EQ stages and routing that fits APO's channel model | It is not a general RoomEQ graph host. Complex fan-out, unsupported plugins, FIR packaging, or unusual channel layouts may be rejected or simplified; verify the generated channel mapping. |
| **EasyEffects (`easyeffects`)** | Linux desktop stereo or simple per-channel correction | Serial single-channel-compatible gain and EQ | No full bass-management graph, channel matrix, crossover topology, arbitrary delay, or general FIR/mixed-phase realization. |
| **Wavelet (`wavelet`)** | Textual magnitude EQ for supported Wavelet workflows | Serial GraphicEQ-style magnitude correction | No routing, delay, crossover, convolution, or phase correction. |
| **PipeWire (`pipewire`)** | PipeWire filter-chain playback, including suitable FIR sidecars | Serial filter chains and supported convolution sidecars | The exporter is not a complete substitute for the canonical routed graph. Confirm channel routing, sidecar paths, and filter-chain support before using it for home-cinema bass management. |
| **Roon (`roon`)** | Roon DSP Engine IIR/FIR playback | Serial Roon-supported IIR/FIR stages within Roon's limits | Roon's supported stage set, channel model, latency, and file-handling limits apply; arbitrary RoomEQ matrices, route graphs, or plugin types are not guaranteed. |
| **REW (`rew`)** | Importing one channel of IIR EQ into REW or another compatible tool | One channel of gain plus supported biquad filters, with an explicit preamp | Exactly one channel. No delay, FIR, crossover, bass routing, matrix, or other graph stage. |
| **Normalized coefficients (`coefficients`)** | Integrating RoomEQ filters into custom DSP | Any number of serial channels, gain, delay, and the 12 canonical RoomEQ biquad types | No FIR, crossover, matrix, bass routing, or plugin graph. The host must apply `preamp_gain_db`, `delay_ms`, section order, and the documented coefficient convention. |

### Practical Target Selection

- For a 5.1/7.1 system with redirected bass, separate main/sub delays, or
  multiple crossover groups, use the canonical JSON or CamillaDSP. These are
  the targets intended to preserve a graph with several source branches
  feeding one physical sub output.
- For ordinary stereo IIR room correction, APO, EasyEffects, PipeWire, Roon,
  or normalized coefficients can be appropriate, depending on the playback
  host.
- For FIR or mixed-phase correction, use the canonical JSON or an exporter
  that explicitly supports the generated convolution sidecars. Keep the WAV
  files with the exported configuration and verify sample rate, channel order,
  latency, and pre-ringing policy.
- Use REW, Wavelet, and normalized coefficients only when you intentionally
  want a reduced per-channel magnitude-EQ representation.

Every external export validates the source graph against the target's known
constraints. A successful export means the artifact is representable by that
target; it does not mean that unsupported RoomEQ routing or temporal behavior
was silently preserved.

## Configuration File Format

### Simple Stereo System

```json
{
  "speakers": {
    "left": "measurements/left_speaker.csv",
    "right": "measurements/right_speaker.csv"
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "nlopt:cobyla",
    "max_iter": 5000,
    "min_freq": 20.0,
    "max_freq": 20000.0,
    "min_q": 0.5,
    "max_q": 10.0,
    "min_db": -12.0,
    "max_db": 12.0,
    "loss_type": "flat"
  }
}
```

### Stereo bass routing with physical outputs (v3)

```json
{
  "version": "3.0.0",
  "system": {
    "model": "stereo",
    "speakers": {
      "L": "left_meas",
      "R": "right_meas"
    },
    "subwoofers": {
      "strategy": "single",
      "routing": "optimize",
      "outputs": [
        { "id": "Sub1", "speaker": "sub_meas" }
      ],
      "crossover": ["bass_xo"]
    }
  },
  "crossovers": {
    "bass_xo": {
      "type": "LR24",
      "frequency": 80.0
    }
  },
  "speakers": {
    "left_meas": "measurements/left.csv",
    "right_meas": "measurements/right.csv",
    "sub_meas": "measurements/sub.csv"
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "cobyla"
  }
}
```

Stereo always exposes exactly two programme inputs, `L` and `R`. The `.1`
or `.2` suffix describes measured physical subwoofer outputs and never creates
an LFE input. A 2.1 route uses the correlated-peak-safe `0.5L + 0.5R` fold.
For 2.2, RoomEQ evaluates direct-pair, crossed-pair, and dual-mono matrices,
running the continuous optimizer independently for each. Every candidate is
round-tripped through the serialized routing graph and must pass the acoustic
splice and electrical-headroom gates before robust-objective comparison;
reports retain each rejection reason and the selected coefficients.

Home cinema uses the same `subwoofers.outputs` shape, but adds the logical
`LFE` programme input implicitly. Do not put `LFE` in `system.speakers`;
that map contains measured main, surround, and height outputs only. LFE gain
and low-pass policy remain under the home-cinema-only `bass_management` object.

The LFE programme cutoff is a separate bass-management control: changing or
optimizing `bass_xo` redirects main-channel bass without narrowing the LFE
programme band. The cinema default is 120 Hz.

For routed home-cinema output, RoomEQ optimizes each logical input against its
own high-passed main plus redirected low-passed sub branch. Crossover type and
frequency are shared within the speaker group; route trim, relative delay, and
polarity are reported per source in `bass_management.optimization.source_results`.
The optimizer never uses a coherent sum of independent programme channels for
tonal calibration. Whole-bus aggregation is reserved for the configured
headroom model, which may add one common down-only input safety trim.

#### Per-sub crossovers (multi-sub low-pass per driver)

A multi-subwoofer system may give each physical sub its own crossover key by
writing `system.subwoofers.crossover` as a positional list instead of a single
string:

```json
"subwoofers": {
  "strategy": "mso",
  "routing": "optimize",
  "outputs": [
    { "id": "Sub1", "speaker": "sub_left" },
    { "id": "Sub2", "speaker": "sub_right" }
  ],
  "crossover": ["bass_xover1", "bass_xover2"]
}
```

Entry `i` applies to physical sub `i` in driver order. The selector evaluates
all mains as separate logical inputs against the complete shared sub array,
including the actual main high-pass and each driver's
low-pass. The selected filters are included before route delay, polarity,
and trim optimization. A crossover list does not create L-to-left-sub or
R-to-right-sub routing. Each selected frequency stays inside its own range.
Every key must exist, and the list must contain exactly one entry per physical
sub output.

Deployment: `LP_i` is a low-pass `crossover` plugin on each physical sub
(`channels.<SUB>.drivers[i].plugins`, staged `post_route`). Redirected-bass
routes omit a second group low-pass when this filter is present. Main high-pass
frequency remains optimized for the shared main group; the LFE programme keeps
its independent low-pass. The optimizer, replay and export use this same graph.
Reports retain `bass_management.groups[].selected_sub_low_pass_hz`,
`bass_management.optimization.sub_output_results[].selected_low_pass_hz`, and
the `per_sub_lp_deployed_to_drivers:N` advisory. Legacy shared-string
crossover configurations are rejected by the v3 loader.


### Multi-driver Speaker (2-way)

```json
{
  "speakers": {
    "left": {
      "name": "Left Speaker (2-way)",
      "measurements": [
        "measurements/left_woofer.csv",
        "measurements/left_tweeter.csv"
      ],
      "crossover": "default_lr24"
    }
  },
  "crossovers": {
    "default_lr24": {
      "type": "LR24"
    }
  },
  "optimizer": {
    "num_filters": 10,
    "algorithm": "nlopt:cobyla",
    "max_iter": 5000,
    "min_freq": 100.0,
    "max_freq": 10000.0,
    "min_q": 0.5,
    "max_q": 10.0,
    "min_db": -12.0,
    "max_db": 12.0,
    "loss_type": "flat"
  }
}
```

### Measurement CSV Format

The four-column microphone phase-calibration library API treats calibration
support as measured evidence: `sample_at` returns `None` outside that support,
and `apply_to_curve` returns an error without modifying the input if any
frequency is unsupported or evidence is malformed. Successful application
invalidates cached phase decomposition. Callers must handle the returned
`Result`; these helper guarantees do not by themselves establish that a
recording/export workflow applied the calibration, or establish absolute SPL
equivalence between CSV, impulse-response and MDAT sources.

Measurement CSV files should have the following columns:
- `freq`: Frequency in Hz
- `spl`: Sound pressure level in dB

Example:
```csv
freq,spl
20,75.0
50,78.0
100,80.0
200,82.0
...
```

## Optimizer Configuration

For single-channel processing, the requested correction band is intersected
with the measurement's native frequency support before preprocessing, scoring,
and PEQ/FIR design. A measurement starting at 100 Hz cannot authorize a 20 Hz
correction band. Disjoint requested and measured bands fail explicitly; the
processor does not invent measurements outside the captured range. This
constrains the design band, not the natural response tails of finite filters.

Progress reporting does not change filter-selection policy. When adaptive
selection is enabled by `min_filter_improvement`, it remains enabled for
interactive and QA runs with progress callbacks, including Hybrid's IIR stage.
A stop request cancels the adaptive run rather than advancing to another pass.

### Tilt stage

`optimizer.tilt_stage` appends an optimizer-driven low-shelf plus high-shelf
pair at the end of the DSP chain to fit broadband bass/treble tilt that peak
filters express poorly. The pair extends `num_filters` by two instead of
replacing peak slots, so enabling tilt never moves the main filters' bounds;
every adaptive pass optimizes its peak count plus the pair. Both shelves use
the existing shelf limits (+/-`max_db` gain, pinned Q) and flow through the
same HF guard, audibility veto, and headroom stages as every other filter.
`ls_band_hz` / `hs_band_hz` optionally pin the hinge bands; each defaults to
its geometric half of the correction band. The stage requires
`peq_model: "pk"`; other bases are rejected during validation.

### Algorithms

Hybrid spatial FIR searches retain the caller's progress/stop callback after
the IIR stage. DE and CMA-ES check it at native generation boundaries as well
as scored FIR basis boundaries. COBYLA and ISRES currently check only at stage
boundaries because the pinned scalar backends lack native stop hooks; their
search can finish before a pending Stop is observed. A result is discarded
once Stop is observed. Completed FIR optimizer evidence records
`callback_cancellation=native_generation_boundaries` or
`callback_cancellation=stage_boundaries_only` when a callback was supplied.
These checks do not interrupt FIR template construction, an initial population,
or an in-flight objective evaluation, and do not promise a maximum stop latency.

- `autoeq:cmaes`: CMA-ES (default global optimizer)
- `autoeq:de`: Differential Evolution
- `autoeq:cobyla`: COBYLA (Constrained Optimization BY Linear Approximations)
- `autoeq:isres`: Improved Stochastic Ranking Evolution Strategy
- Other AutoEQ and metaheuristics algorithms supported by autoeq

### Loss Types

- `flat`: Optimize for flat frequency response
- `score`: Optimize for Harman/Olive score (bass boost + flat PIR)
- `epa`: experimental ERB-rate loss with diagnostic transfer-response EPA descriptors; not a validated programme loudness/roughness or measured-decay model

The neutral flat/asymmetric objective and runtime acceptance use the same
versioned auditory measure, `glasberg-moore-erb-rate-1990-v1`. It integrates
residual energy with discrete ERB-rate cell widths, so changing between linear,
logarithmic, sparse, or dense frequency grids does not silently change the
meaning of the reported RMS. Residual signs are never smoothed before the
nonlinear loss.

For multiple measurements, `spatial_robustness` evaluates every seat directly
with a variance-penalized risk measure. Its spatial-variance correction-depth
mask is applied to each seat; seats are not collapsed to a power-average curve
before optimization. Case-bootstrap uncertainty is explicitly labelled
`spatial_seat_sampling` and assumes independent positions by default. For
correlated nearby seats, configure a reduced `effective_spatial_sample_size`;
the bootstrap then draws fewer cases per resample and produces a wider,
conservative interval. This is not a spatial block bootstrap or covariance
model. Repeat-sweep noise and microphone-calibration uncertainty remain
separate, explicitly supplied nuisance sources.

When `psychoacoustic` is enabled, `psychoacoustic_smoothing` can override the default variable smoothing curve (`1/48` octave below 100 Hz through `1/6` octave above 1 kHz). When `asymmetric_loss` is enabled, `asymmetric_loss_config` can override peak/dip and bass peak/dip weights without changing the default behavior for existing configs.

`perceptual_policy` can fill coherent defaults for `reference`, `music`, `cinema`, `night`, and `speech` use cases. The policy layer maps existing knobs rather than replacing them: target response, EPA/asymmetric weighting, psychoacoustic smoothing, spatial/bootstrap robustness, audibility deadband, high-frequency guardrails, FIR direct/early/late advisories, and validation bundle descriptors remain individually configurable.

The high-frequency guard's `max_q` is a local optimizer constraint, not a
replacement for the global `optimizer.max_q`. Applying defaults (including a
policy override) preserves the global cap so bass filters keep their configured
Q range. Search intervals that reach the guarded band can still receive a
conservative local cap; returned candidates are also checked. A stricter global
cap continues to win. This does not establish direct-sound evidence or complete
the separate evidence-dependent magnitude-target policy.

### Crossover Types

- `LR24` or `LR4`: Linkwitz-Riley 24 dB/oct (4th order)
- `LR48` or `LR8`: Linkwitz-Riley 48 dB/oct (8th order)
- `Butterworth12` or `BW12`: Butterworth 12 dB/oct (2nd order)
- `Butterworth24` or `BW24`: Butterworth 24 dB/oct (4th order)
- `LinearPhase`, `FIR`, or `LPFIR`: complementary FIR crossover with constant group delay and no crossover-point phase rotation

## Advanced Audio Corrections

RoomEQ provides advanced audio correction features for optimizing room acoustics in two scenarios:

- **Scenario A (WITH Subwoofers)**: Phase alignment and multi-seat variance minimization
- **Scenario B (WITHOUT Subwoofers)**: Schroeder split, excursion protection, and target response shaping

### Target Response

Some listeners prefer a gently downward-sloping house curve. RoomEQ's
**-0.8 dB/octave** Harman-style option is a user preference, not a universal
neutral room-correction target. It is emitted in the separately bypassable
preference layer and excluded from neutral correction quality scores. Target
shaping is configured through the unified `target_response` object, together
with optional user-preference shelves and the broadband pre-correction toggle.

```json
{
  "optimizer": {
    "target_response": {
      "shape": "harman",
      "slope_db_per_octave": -0.8,
      "reference_freq": 1000,
      "preference": {
        "bass_shelf_db": 0,
        "bass_shelf_freq": 200,
        "treble_shelf_db": 0,
        "treble_shelf_freq": 8000
      },
      "broadband_precorrection": false
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | string | `"flat"` | Target shape: `"flat"`, `"harman"`, `"custom"`, `"file"`, `"from_measurement"`; `harman` is a house curve realized in the preference layer |
| `slope_db_per_octave` | number | -0.8 | Slope in dB/octave (negative = downward tilt). Used when `shape == "custom"` |
| `reference_freq` | number | 1000 | Frequency where the target slope passes through 0 dB (Hz) |
| `curve_path` | string (path) | - | CSV target file path (used when `shape == "file"`) |
| `preference.bass_shelf_db` | number | 0 | Bass shelf in the separately bypassable post-correction preference layer (dB) |
| `preference.bass_shelf_freq` | number | 200 | Bass shelf transition frequency (Hz) |
| `preference.treble_shelf_db` | number | 0 | Treble shelf in the separately bypassable post-correction preference layer (dB) |
| `preference.treble_shelf_freq` | number | 8000 | Treble shelf transition frequency (Hz) |
| `broadband_precorrection` | boolean | false | Run a preliminary broadband shelf + gain fit before the fine-grained PEQ pass |

The neutral optimizer target is computed without preference shelves:
```
target_db(f) = slope * log2(f / reference_freq)
```

Preference shelves are realized afterward as a separate output IIR layer. They
remain visible in the final DSP/plugin chain, but are excluded from neutral
post-EQ scores and `raw_post_eq_curve`. Output metadata records both
`neutral_target_response` and `preference_layer`, including
`excluded_from_neutral_quality_score: true`.

**Example: Harman with Bass Boost**

```json
{
  "optimizer": {
    "target_response": {
      "shape": "harman",
      "preference": {
        "bass_shelf_db": 3,
        "bass_shelf_freq": 200
      }
    }
  }
}
```

### Excursion Protection

Bookshelf speakers and small drivers have limited bass extension. Attempting to boost bass below the speaker's F3 point (-3dB frequency) can cause excessive driver excursion, increased distortion, and potential damage.

Excursion protection automatically detects the F3 rolloff and generates a highpass filter to prevent dangerous over-boost.

```json
{
  "optimizer": {
    "excursion_protection": {
      "enabled": true,
      "auto_detect_f3": true,
      "f3_reference_min_hz": 100.0,
      "f3_reference_max_hz": 200.0,
      "filter_order": 4,
      "filter_type": "linkwitzriley",
      "margin_octaves": 0.25
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable excursion protection |
| `auto_detect_f3` | boolean | true | Auto-detect F3 from measurement |
| `manual_f3_hz` | number | - | Manual F3 override (Hz) when auto-detect is false |
| `filter_order` | integer | 4 | HPF order: 2=12dB/oct, 4=24dB/oct |
| `filter_type` | string | `"linkwitzriley"` | Filter type: `"linkwitzriley"` or `"butterworth"` |
| `margin_octaves` | number | 0.25 | Safety margin below F3 for HPF placement |

**F3 Detection Algorithm:**
1. Smooth the measurement curve (1/3 octave)
2. Find reference level at 100-200 Hz
3. Search downward for -3dB point
4. Place HPF at `F3 * 2^(-margin_octaves)`

### Schroeder Frequency Split

The **Schroeder frequency** marks the transition between modal (low frequency) and statistical (high frequency) behavior in a room. Below this frequency, room modes dominate and require high-Q narrow filters for correction. Above this frequency, broad tonal adjustments are more appropriate.

Typical Schroeder frequencies:
- Small room (15 m³): ~400 Hz
- Medium room (40 m³): ~250 Hz
- Large room (100 m³): ~160 Hz

```json
{
  "optimizer": {
    "schroeder_split": {
      "enabled": true,
      "schroeder_freq": 300,
      "low_freq_config": {
        "max_q": 5.0,
        "min_q": 0.5,
        "allow_boost": false
      },
      "high_freq_config": {
        "max_q": 1.0,
        "shelving_only": false
      }
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable Schroeder split |
| `schroeder_freq` | number | 300 | Schroeder frequency (Hz) |
| `low_freq_config.max_q` | number | 5.0 | Max Q for low-freq filters |
| `low_freq_config.min_q` | number | 0.5 | Min Q for low-freq filters |
| `low_freq_config.allow_boost` | boolean | false | Allow boosts (not recommended) |
| `high_freq_config.max_q` | number | 1.0 | Max Q for high-freq filters |
| `high_freq_config.shelving_only` | boolean | false | Use only shelving filters |

**Auto-Calculate Schroeder from Room Dimensions:**

Declare dimensions once in `recording_config.room_dimensions`; both Schroeder
split and decomposed correction use that physical room. Optimizer-local copies
are no longer accepted. See [recording configuration](ROOMEQ_INPUT_FORMAT.md#recording-configuration-recording_config).

```json
{
  "recording_config": {
    "room_dimensions": { "length": 5.0, "width": 4.0, "height": 2.5 }
  },
  "optimizer": {
    "schroeder_split": {
      "enabled": true
    }
  }
}
```

When RT60 and room volume are available, RoomEQ calculates the Schroeder
frequency as `f_S ≈ 2000 · √(RT60 / V)`, where RT60 is seconds and V is room
volume in m³.

### Phase Alignment

When integrating a subwoofer with main speakers, proper time/phase alignment in the crossover region is critical. Misalignment causes cancellation dips at crossover, reduced bass output, and poor transient response.

Phase alignment optimizes the delay and polarity to maximize energy sum in the crossover region.

```json
{
  "optimizer": {
    "phase_alignment": {
      "enabled": true,
      "min_freq": 60,
      "max_freq": 100,
      "optimize_polarity": true,
      "max_delay_ms": 30
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable phase alignment |
| `min_freq` | number | 60 | Minimum optimization frequency (Hz) |
| `max_freq` | number | 100 | Maximum optimization frequency (Hz) |
| `optimize_polarity` | boolean | true | Test both normal and inverted polarity |
| `max_delay_ms` | number | 3 | Maximum delay search range (ms); the default is refined by the phase-alignment scan and golden-section search |

**Algorithm:**
1. **Global scan**: Test delays from -max_delay to +max_delay with frequency-adaptive sampling (the default range is ±3 ms)
2. **For each candidate**: Compute combined response `|H_sub + H_speaker * e^(-jωτ) * polarity|`
3. **Integrate energy** in [min_freq, max_freq] band
4. **Fine search**: Refine the best scan interval with a golden-section search
5. **Output**: Optimal delay and polarity for maximum energy sum

**Note:** Both subwoofer and speaker measurements must include phase data (export from REW with phase, or measure with calibrated mic).

### Multi-Seat Optimization

In rooms with multiple listening positions, optimizing for one seat often degrades others. Multi-seat optimization finds subwoofer gain/delay settings that minimize variance across all seats.

```json
{
  "optimizer": {
    "multi_seat": {
      "enabled": true,
      "strategy": "minimize_variance",
      "primary_seat": 0,
      "max_deviation_db": 6
    }
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | false | Enable multi-seat optimization |
| `strategy` | string | `"minimize_variance"` | Optimization strategy |
| `primary_seat` | integer | 0 | Primary seat index (0-based) |
| `max_deviation_db` | number | 6 | Max deviation at secondary seats (dB) |

For bass-managed playback, mains and a single subwoofer use the same primary
seat's measured complex response for crossover alignment. All captured sub
seats remain available for spatial magnitude EQ; their magnitude average is
not a phase measurement. Capture matching seats in the same order and retain a
shared timing reference across speakers.

Before crossover processing, RoomEQ checks raw per-seat coherence and SNR in
the candidate overlap band (half the lowest to twice the highest crossover
frequency). Any declared coherence below `recording_config.coherence_threshold`
(default 0.9), malformed confidence data, or declared SNR below 10 dB rejects
the measurement. Missing coherence or noise-floor metadata produces
`crossover_phase_coherence_unverified` or `crossover_phase_snr_unverified`
advisories, not proof of trustworthy phase. These checks do not certify a
shared timing reference or require subwoofer measurements in the treble.

**Strategies:**

| Strategy | Description |
|----------|-------------|
| `minimize_variance` | Minimize standard deviation of SPL across all seats |
| `primary_with_constraints` | Optimize primary seat, constrain others within max_deviation |
| `average` | Optimize for flattest average response across seats |
| `modal_basis` | Complex modal-basis SFM: extracts dominant seat modes from per-sub/per-seat transfer functions and optimizes sub gain/delay/polarity/all-pass controls |

**Measurement Setup:**

For multi-seat optimization, you need measurements of each subwoofer at each seat position:

```json
{
  "speakers": {
    "subs": {
      "name": "Multi-seat Subwoofers",
      "subwoofers": [
        ["sub1_seat1.csv", "sub1_seat2.csv", "sub1_seat3.csv"],
        ["sub2_seat1.csv", "sub2_seat2.csv", "sub2_seat3.csv"]
      ]
    }
  }
}
```

### Supporting-Source Room Compensation

A supporting-source loudspeaker is a delayed, decorrelated loudspeaker placed in
same room as the primary source. It adds reverberant energy to the primary
source without altering its direct sound, improving apparent source width and
room envelopment while preserving imaging (Brooks-Park et al., JASA 159(4),
2026). RoomEQ computes a minimum-phase FIR for the supporting source that fills
in the primary's room-induced spectral notches, subject to a precedence ceiling
and a configurable compensation band.

```json
{
  "system": {
    "model": "stereo",
    "speakers": {
      "L": "left_pair",
      "R": "right_pair"
    }
  },
  "speakers": {
    "left_pair": {
      "name": "Left Main + Support",
      "primary": "measurements/left_primary.csv",
      "support": "measurements/left_support.csv",
      "supporting_source": {
        "delay_ms": 10.0,
        "allow_unverified_acoustics": true,
        "freq_range_hz": [70.0, 20000.0],
        "decorrelation": "velvet_noise",
        "fir_taps": 8192,
        "velvet_noise_taps": 4096,
        "precedence_limits": [
          { "low_hz": 70.0, "high_hz": 500.0, "limit_db": 10.0 },
          { "low_hz": 500.0, "high_hz": 20000.0, "limit_db": 6.0 }
        ]
      }
    },
    "right_pair": {
      "name": "Right Main + Support",
      "primary": "measurements/right_primary.csv",
      "support": "measurements/right_support.csv",
      "supporting_source": {
        "delay_ms": 10.0,
        "allow_unverified_acoustics": true
      }
    }
  },
  "optimizer": {
    "processing_mode": "phase_linear",
    "loss_type": "flat"
  }
}
```

For each `SupportingSourceGroup`, RoomEQ emits two output channels (e.g.
`L` and `L_support`, or `WideLeft` and `WideLeft_support` in a home-cinema
layout). The primary output intentionally bypasses ordinary RoomEQ EQ, so its
direct sound remains unchanged; the configured optimizer does not apply to
that primary path. The supporting channel contains a `convolution` plugin
loading the generated FIR WAV file. `metadata.supporting_source` records this
as `primary_eq_bypassed_to_preserve_direct_sound`, along with precedence-limit
and spatial-robustness advisories. Absolute DRR summaries are present only
when time-gated impulse-response evidence is available.

The example above is explicitly experimental: magnitude curves alone do not
establish arrival timing or coherent interference. By default supporting-source
processing requires `acoustic_arrival_offset_ms` (unfiltered support arrival minus
primary arrival at the reference seat, measured on a common time reference) and
`shared_phase_reference: true` with measured phase on both transfers. Electrical
delay is `delay_ms - acoustic_arrival_offset_ms`: a support source arriving 2.5 ms
earlier needs 12.5 ms electrical delay to achieve a requested 10 ms propagation-plus-
electrical lag. A required advance, or `optimizer.allow_delay: false` when positive
electrical delay is needed, is rejected. Do not set the shared-phase flag merely
because CSVs contain phase columns; independent sweep time origins are insufficient.

The realized FIR, gain, and delay are replayed for the reported coherent sum.
`max_coherent_cancellation_db` defaults to a 3 dB engineering budget below the
louder branch; exceeding it rejects a verified-mode run. Explicit
`allow_unverified_acoustics: true` permits experimental operation with conspicuous
missing-evidence/over-budget advisories. The report labels the **power-average
design** separately from `coherent_sum`; the latter is omitted without shared-phase
evidence. `propagation_relative_arrival_ms` excludes FIR energy spread and is not a
measured perceptual onset. Reference-seat prediction does not prove fusion,
localization, DRR, or multi-seat benefit; validate those with final-chain measurements
and controlled listening. Required evidence is checked before writing the FIR.

### Listening-stimulus resolution

The quality harness's band-limited noise has nominal -6 dB band edges and a
transition allowance of half the smallest of bandwidth, lower edge, and upper-edge
clearance to Nyquist. The FIR design targets at least 40 dB stopband rejection
outside those transitions. Requests exceeding 4095 taps are errors, not silently
widened bands; for example 90–110 Hz at 48 or 96 kHz is unsupported. Minimum
bandwidth and edge clearance are approximately `8 * sample_rate / 4095` Hz.
Deterministic filter-response checks complement seeded waveform tests; finite
stimulus duration still matters when interpreting measured spectra.

### Complete Configuration Examples

**Scenario A: System with Subwoofers**

```json
{
  "speakers": {
    "left": "measurements/left.csv",
    "right": "measurements/right.csv",
    "sub": "measurements/subwoofer.csv"
  },
  "optimizer": {
    "algorithm": "autoeq:cmaes",
    "num_filters": 10,
    "refine": true,

    "target_response": {
      "shape": "harman",
      "preference": {
        "bass_shelf_db": 2
      }
    },

    "phase_alignment": {
      "enabled": true,
      "min_freq": 60,
      "max_freq": 100,
      "optimize_polarity": true,
      "max_delay_ms": 30
    }
  }
}
```

**Scenario B: Bookshelf Speakers without Subwoofer**

```json
{
  "speakers": {
    "left": "measurements/left_bookshelf.csv",
    "right": "measurements/right_bookshelf.csv"
  },
  "optimizer": {
    "algorithm": "autoeq:cmaes",
    "num_filters": 12,
    "refine": true,

    "target_response": {
      "shape": "harman"
    },

    "excursion_protection": {
      "enabled": true,
      "auto_detect_f3": true,
      "filter_order": 4,
      "margin_octaves": 0.25
    },

    "schroeder_split": {
      "enabled": true,
      "schroeder_freq": 300,
      "low_freq_config": {
        "max_q": 10,
        "allow_boost": false
      },
      "high_freq_config": {
        "max_q": 1.0
      }
    }
  }
}
```

### Optimization Flow

When multiple features are enabled, the optimization follows this order:

```
1. Load measurement(s)
2. Build the neutral target curve from `target_response.shape` (Harman house curve and other preferences stripped)
3. [IF excursion_protection] Detect F3, generate protection HPF
4. [IF has_subwoofer && phase_alignment] Optimize delay/polarity for energy max
5. [IF multi_seat] Optimize sub gains/delays for variance minimization
6. [IF schroeder_split] Two-pass EQ (low-Q high freq, high-Q low freq)
   [ELSE] Standard EQ optimization
7. Append the separately bypassable preference/content layer and combine the DSP chain
```

### API Reference

The features are also available programmatically:

```rust
use autoeq::roomeq::{
    // Target Response
    build_complete_target_curve,
    TargetResponseConfig, TargetShape, UserPreference,

    // Excursion Protection
    detect_f3, generate_excursion_protection,
    ExcursionProtectionConfig, ExcursionProtectionResult,

    // Phase Alignment
    optimize_phase_alignment,
    PhaseAlignmentConfig, PhaseAlignmentResult,

    // Multi-Seat
    optimize_multiseat,
    MultiSeatMeasurements, MultiSeatConfig, MultiSeatOptimizationResult,
};
```

## Output Format

The output is a JSON file containing DSP chains for each channel:

```json
{
  "channels": {
    "left": {
      "channel": "left",
      "plugins": [
        {
          "plugin_type": "gain",
          "parameters": {
            "gain_db": -2.5
          }
        },
        {
          "plugin_type": "eq",
          "parameters": {
            "filters": [
              {
                "filter_type": "peak",
                "freq": 1000.0,
                "q": 1.5,
                "db_gain": 3.0
              }
            ]
          }
        }
      ]
    }
  },
  "metadata": {
    "pre_score": 0.0,
    "post_score": 0.0,
    "algorithm": "nlopt:cobyla",
    "iterations": 5000,
    "timestamp": "2025-01-15T12:00:00Z"
  }
}
```

This output can be loaded directly into the AudioEngine plugin system.

## Examples

See the `tests/data/roomeq/` directory for example configurations:
- `test_config_stereo.json`: Simple stereo system
- `test_config_multidriver.json`: Multi-driver speaker with crossover

## Documentation

Detailed format documentation with examples:
- [`ROOMEQ_INPUT_FORMAT.md`](ROOMEQ_INPUT_FORMAT.md): Complete input configuration format
- [`ROOMEQ_OUTPUT_FORMAT.md`](ROOMEQ_OUTPUT_FORMAT.md): Complete DSP chain output format

JSON Schemas for validation:
- [`input_schema.json`](../src/bin/roomeq/input_schema.json): Input configuration schema
- [`output_schema.json`](../src/bin/roomeq/output_schema.json): Output DSP chain schema

Configuration validation is also exposed as a versioned five-stage runtime
report: `schema_version`, `structural`, `resolved_resource`, `acoustic`, and
`export_target`. A structural-only report is intentionally not
`production_ready`; the CLI loader returns the staged report and the production
optimizer reruns the required resource/acoustic gates after path resolution.

## Testing

Run the integration tests:
```bash
cargo test -p autoeq --test roomeq_integration_test
```

Run the unit tests:
```bash
cargo test -p autoeq --bin roomeq
```

## Architecture

Production RoomEQ code is partitioned by responsibility:

- `roomeq-model`: configuration, validation, and output contracts.
- `roomeq-analysis`: measurement, phase, spatial, and acoustic analysis.
- `roomeq-quality`: perceptual metrics and acoustic-corpus acceptance.
- `roomeq-engine`: DSP, filter design, optimization, routing, and safety gates.
- `roomeq-workflow`: configuration loading and complete run orchestration.
- `roomeq-export`: export-target conformance and artifact generation.
- `roomeq-cli`: command-line parsing, schemas, and result serialization.

The historical root `src/roomeq/` tree is a compatibility facade, not the
production implementation boundary.

### QA tiers and scenario registry

Escaped-defect ownership checking now requires a regression identity object with
`package`, library `target`, and fully qualified `name`, resolved against nextest
JSON discovery. Ignored or zero-selected inventories cannot establish ownership.
Recipes are resolved from Just's JSON dump and literal dependency/run commands
reachable from `.github/workflows/ci.yml`; dynamic shell invocations are not
guessed. A mutant fixture needs a manifest and runner, not only a README.
`just qa-roomeq-escaped-defects` resolves and executes the registered exact tests;
`just qa-roomeq-escaped-defect-mutants` separately executes the registered
semantic faults. Their evidence records distinguish a completed run from a
running or failed one and retain per-defect artifact hashes. These recipes are
wired into the blocking RoomEQ CI workflow; local results are not evidence of
a hosted CI run. Ownership discovery alone never establishes killed mutants.
Checker regressions run with
`python3 -m unittest scripts.test_escaped_defect_ownership`.

The opt-in `--parameter-matrix` runner executes two separate contracts:

- `--parameter-matrix-refusals` retains the six full-scale 5.1 cases. Each
  must refuse all five seeds specifically because structural attenuation
  exceeds the unchanged 12 dB limit. A different error or successful output
  fails the test. Results go to `target/qa/roomeq-parameter-refusals.json`;
  each bundle has `refusal.json`, not a selected processing output.
- `--parameter-matrix-reduced-level` runs the full processing matrix with
  explicitly declared logical-input peak amplitude 0.1 (−20 dBFS). The
  measurements, routing, 0 dBFS output ceiling and 12 dB attenuation limit
  are unchanged. The scenario and exact finalization settings are retained
  in requests/results. These are lower-level processing tests, not evidence
  that the same processing is safe for full-scale correlated inputs.

The combined command fails if either contract fails. Processing cases still
require finite output and artifact checks; unrelated crossover/timing errors
are not accepted as headroom refusals. This establishes finite-output smoke
coverage, not complete pairwise execution or useful correction. Each reduced
row also records whether the selected correction was accepted or safely fell
back to identity, with refusal reasons. The smoke check requires actual FIR
tap retention for accepted PhaseLinear and Hybrid cases; a requested FIR in a
row without usable phase can safely fall back. The current matrix delivers
20 ms FIR examples. Separate fast engine tests exercise 5 and 10 ms
PhaseLinear FIR generation, sidecar emission, and attenuation of a broad
300 Hz peak through direct coefficient evaluation. Those lower-level tests
do not establish final workflow acceptance for the shorter lengths.
Each parameter-matrix row retains a unique directory under
`target/qa/roomeq-parameter-bundles/`. `request.json` stores the exact single-speaker
measurements separately from `configuration_without_speakers`, since internal
in-memory sources are not serializable CLI measurement references. Phase-bearing
synthetic responses additionally carry a declared common stationary timing
reference in `declared_measurement_sources`; replay must restore those sources
to exercise automatic crossover admission. This declaration describes the
analytic QA fixture, not an acoustic recording or authenticated capture.
`selected-output.json` and its convolution sidecars retain the selected rerun's
DSP for independent replay after execution. The matrix's `replay_bundle` links
these artifacts; earlier bundles are not overwritten by subsequent runs.
This artifact-retention contract does not establish backend or acoustic success.

Processing failures do not stop later rows. A refused or invalid row retains
`failure.json` in its bundle, including its row, stage, request axes, and error.
Running checkpoints retain both `completed_rows` and `failed_rows`. If any row
fails, the final matrix artifact is an object with `status: "failed"`, both row
lists, `expected_rows`, and `attempted_rows`; the command exits nonzero. An
all-successful run retains the existing array format. Missing or zero executed
rows cannot pass. Artifact-write errors remain fatal. Continuing after a safety
refusal does not authorize that output or change its safety budget.

`target/qa/roomeq-parameter-matrix.json` records requested axes, effective optimizer
and routing settings, measurement descriptors, selected DSP rate, delivered FIR
lengths, stage outcomes, and seed reliability. The selected rate is passed to
optimization; stereo, redirected 2.1, and 5.1 configurations use their actual
topology builders. Crossover values request fixed 160 Hz LR24, automatic
120–220 Hz LR24, and fixed 160 Hz LR48 respectively. The shell contract checks
each reported physical main/sub branch against its selected crossover group;
automatic selection without measured phase must remain explicitly skipped,
not counted as successful search. These route checks are not an independent
render of every matrix row. Full execution and usefulness/mutation coverage
remain unfinished. CLI success/failure checks are in
`scripts/test_roomeq_synthetic_exit.py`; run after building the release QA binary.

The public acoustic quality evaluator now records `useful_output` separately
from normalized shape RMS, for every training and held-out seat. Its default
authorized broadband gain is 0 dB. Library callers can use
`evaluate_acoustic_quality_with_permitted_gain` to declare an intended trim or
headroom attenuation; this gain is not fitted from the candidate. The unexplained
loss metric is log-frequency-weighted RMS below the lesser of baseline and target
(or baseline when no target exists), after the declared gain. This allows removing
above-target peaks without requiring inversion of existing nulls. The public gate
uses a configurable 3 dB engineering budget, not a listening-calibrated threshold.
A separate loss RMS over measured support at or below 200 Hz prevents the main
band from diluting bass loss; its actual evaluated band is reported, and fewer
than two supported bass samples leave that evidence absent.
Absolute target-shortfall RMS remains visible even for authorized attenuation;
exceeding its separate 3 dB advisory budget reports `best_effort_target_shortfall`.
Final multi-seat replay now retains this evidence across logical inputs and both
partitions and enforces the 3 dB loss budget. Explicit per-logical-input correction
gain allowances use `optimizer.permitted_output_gain_db`; see the input format.
Structural routing gain remains in both pre/post baselines. Final cumulative selection also evaluates single-seat paths. It enforces
`optimizer.finalization` electrical limits on the assembled graph and tries
reduced correction strengths with bounded headroom attenuation. Main and sub
strengths can vary separately to preserve useful bass correction. Configured
channel-level alignment is reapplied before final electrical and acoustic
checks. Every candidate is evaluated against the same structural baseline;
correction-owned headroom attenuation remains part of the useful-output test.
Rejected search alternatives are diagnostics, with separate enforced safety
evidence for the selected graph.

The default input assumption is independently phased unit-peak sinusoids on
every logical input, with a sampled output ceiling of 0 dBFS. The default 12 dB
attenuation search limit does not grant an acoustic gain allowance. See
`INPUT_FORMAT.md` for input assumptions and gain policy. The electrical verdict
covers the reported frequency grid; continuous-frequency, transient, true-peak
and physical-device certification remain separate work.

Final-seat replay preserves the full-range main assessment without requiring
subwoofer measurements through the treble. A routed subwoofer measured through
twice its deployed low-pass frequency uses an automatic stopband assumption:
a falling measured tail continues at its fitted rolloff (`measured_subwoofer_stopband_rolloff`),
while a rising tail keeps the flat peak-hold form (`assumed_subwoofer_stopband_below_measured_tail`).
Actual branch DSP is applied to that envelope, and aggregate
omission must stay within 0.1 dB magnitude uncertainty. No subwoofer phase
is extrapolated. Mains and missing crossover-band measurements are not exempt.

Explicit calibrated `optimizer.upper_band_acoustic_bounds` declarations take
precedence for their physical output, partition, and seat. Reports retain
magnitude and phase uncertainty and use a conservative improvement lower bound.
If neither measured support nor an adequate bound is available, replay retains
the insufficient-evidence outcome. See `src/bin/roomeq/INPUT_FORMAT.md` for the
configuration and calibration contract.

`optimizer.correction_band` is an optional explicit active-correction range.
It can leave a source's natural extension (for example 20–40 Hz) untouched
while the fixed `optimizer.min_freq..=max_freq` observation band remains in
the before/after score. A narrowed range must set
`allow_natural_rolloff: true`; no high-pass or low-pass is installed by this
policy. Final-seat reports expose the requested correction band separately
from the evaluated band.

Unequal lower endpoints do not silently shorten the assessment band either.
Within the requested band, if one driver or routed output has measured bass
that another physical branch does not cover, replay reports insufficient
summation evidence. Upper-band bounds do not certify missing lower-band
response. Supply the missing capture or explicitly request only the common
supported band; replay does not infer a tweeter's acoustic stopband from its
electrical crossover. ULP-scale endpoint rounding is tolerated, not meaningful
frequency extrapolation.

Routed correction scores use the same measured passband
and correction-only basis for pre and post, removing routing transfer from the
post response. These shape scores do not replace coherent routed-splice,
useful-level, electrical-headroom, or held-out-seat evidence.

Phase-linear and single-measurement Hybrid FIR outputs publish the calibrated design target,
including flat targets. Final correction acceptance preserves explicit target
levels when cropping to a routed passband; only inferred legacy targets are
re-leveled within that band. A selected crossover must not redefine the target
after optimization. Native target grids are aligned explicitly to the measured
response for acceptance. A target that does not cover the evaluated passband is
not extrapolated or accepted on a silently narrower band.

Hybrid residual FIR design keeps the target level prepared from the original
optimization input. The IIR stage's residual does not establish a new target
level; existing FIR boost limits and final acceptance still apply.

For multi-measurement Hybrid with a linear-phase FIR, the residual stage now
scores the complete IIR+FIR response against the same prepared per-measurement
objectives as the IIR stage, including weights, spatial masks and bootstrap
risk. It searches convex combinations of equal-length per-objective,
representative and neutral FIR designs; this is a finite candidate-basis
optimization, not an unrestricted FIR optimum. The displayed channel target
remains a representative target, not a replacement for the per-seat objective
bank. Optimizer evidence identifies this search and its selected candidate.
The bounded scalar backends currently supported by this stage are DE, CMA-ES,
COBYLA and ISRES; another requested backend fails explicitly rather than being
silently substituted. Multi-measurement minimum-phase Hybrid uses an aligned
per-objective dB-correction basis instead: each trial is realized through the
minimum-phase generator before its actual finite-tap response is scored.
It does not mix minimum-phase coefficients. This costs more per evaluation
than the cached linear-phase bank. It currently requires aligned objective
grids and uses the same bounded scalar backends. Kirkeby now uses the same
realized dB-basis search for multi-seat magnitude selection, while retaining
the representative residual as its acoustic phase reference. Requested
excess-phase correction requires actual reference acoustic phase; electrical
IIR phase cannot replace missing measurement evidence.

Kirkeby FIR generation compares the actual windowed coefficients with a
magnitude-only design using the same tap count, target and correction band.
If excess-phase inversion worsens target error by more than 0.05 dB, RoomEQ
keeps magnitude-only correction and logs the fallback. This also applies to
standalone FIR and mixed-band FIR generation. Final electrical, acoustic-output
and temporal acceptance limits remain unchanged; the fallback does not imply
successful phase correction.

Standalone phase correction also retains refusal decisions in workflow metadata
and the CLI's finalized correction ledger. These distinguish missing evidence,
missing phase, failed assessment, magnitude/latency limits, and FIR artifact-write
failure. Reported bands describe evaluated evidence scope, not an applied FIR's
affected interval. A failed FIR write leaves the original channel processing and
predicted response unchanged. Identity fallback means the candidate was reverted;
it does not establish that the uncorrected baseline is already acceptable.

Direct-sound claims now require per-source `provenance.direct_sound` capture facts
and explicit quasi-anechoic policy. Intake checks positive gate/geometry, stationary
averaging, distinct angular coverage, and acquisition bandwidth, then intersects
the supported band with the measured grid and declared limits. A legacy angular
boolean cannot authorize detailed correction. Short-gate support cannot authorize
coherent crossover/joint-sub work below its valid band. Missing facts refuse those
claims while preserving independently supported magnitude processing; ordinary
stationary room IR timing does not require a quasi-anechoic declaration. See
`src/bin/roomeq/INPUT_FORMAT.md` for the schema and a synthetic example. Policy
values are explicit engineering choices, not universal psychoacoustic thresholds;
consistent declarations are not independently authenticated captures.

Successfully realized standalone phase FIRs also produce provisional phase
decisions carrying target-policy reasons, evidence references, requested band
endpoints, measured FIR magnitude deviation, and causal centering delay with
the applicable limits. Requested endpoints are not reported as the exact
affected interval of a finite FIR. Finalization checks the emitted resource
reference: a removed convolution becomes a bound phase reversion with the
original attempt preserved as history; a replacement resource requires
reassessment. This includes removals inside safety processing before the final
processing snapshot. Applied processing does not establish audible benefit.

Public `RoomPipeline`/`optimize_room` results now carry a workflow-finalized
decision snapshot through `to_dsp_chain_output`; consumers need not manually
attach a ledger. Ordinary selected PEQ stages report their objective scores and
requested frequency limits. These limits do not assert an observed affected
interval, and selected processing does not prove acoustic or perceptual benefit.
Processing changes during finalization leave candidate claims unresolved until
reassessed; a later serialized-payload change invalidates the stored bindings.
The CLI re-finalizes after adding effective configuration metadata.

Final headroom and channel-level alignment gains also produce bound
`gain_adjust` decisions. Each row identifies the logical input and physical
output using the same branch expansion as electrical replay, and reports the
signed `delivered_gain_db` from the serialized gain stage. A common pre-route
gain appears on each affected branch; a shared post-route gain is shown for
each contributing input. Do not sum rows across parallel paths to infer net
output gain. These are broadband scalar DSP observations, not measured seat
responses, acoustic-benefit claims, or calibrated physical-output margins.
The report displays these explanations alongside other correction decisions.
Repeated finalization recomputes the records and refuses stale or altered
generated claims; unchanged records are not duplicated.

New finalized ledgers also contain `payload_binding` with the versioned
`sha256-json-typed-v1` digest. The Python report independently recomputes it
over the delivered JSON (excluding the ledger), checks final decision identities,
and checks actual FIR bytes against `metadata.final_convolution_sha256`.
Use the normal result loader so relative sidecars resolve beside the saved JSON.
Missing legacy bindings, unavailable resources, changed payloads, and changed
FIR bytes are explicitly unverified; they cannot produce applied-delivery counts
or a green playback-approval box. The verdict renders as three boxes — recorded
eligibility (what RoomEQ accepted), delivered-payload binding (whether the
bytes still match the decision ledger), and playback approval (the final
go/no-go) — so a changed payload shows as a yellow binding box plus a red
playback box even when the recorded eligibility stays green. A changed payload
means the recorded decisions describe different bytes; re-finalize before
playback. Plot-only FIR caches do not modify the
serialized graph. Packaging verifies the source ledger before rebinding after
resource-reference rewriting. The digest is an artifact-consistency check, not
a signature, acoustic playback verification, physical-safety proof, or listening result.

The cross-language production-report regression runs with
`cargo test -p autoeq --test roomeq_admission_correction report_binds`.
It uses `ROOMEQ_REPORT_PYTHON` when set, otherwise `venv/bin/python` when
available, then `python3`; install the normal report dependencies for that
interpreter. Its temporary files are explicitly under `/Volumes/home_tmp/tmp`.

The scratch-buffer and
calibration-phase defects are repaired in integrated math-iir-fir 0.5.23.
The 2026-09-20 focused Kirkeby regressions pass reference-phase seat weighting,
missing acoustic phase rejection, and causal-delay export. These software
checks do not establish complete temporal or spatial outcome acceptance.
A separate primitive diagnostic finds arrival-delay attenuation
from finite-tap windowing. Do not interpret magnitude improvement as phase
benefit. Causal-support and phase realization fixes, broader
minimum-phase outcome checks, full temporal/headroom validation and
backend-rendered spatial benefit remain open outcome-audit work.
When a required post-workflow FIR cannot be generated or its WAV cannot be
written, the workflow fails instead of installing a nonexistent sidecar or
silently omitting the requested stage.

Fractional FIR group-delay alignment uses a 129-tap windowed-sinc delay kernel
with a declared usable band up to 0.46 times sample rate and a 0.01 dB magnitude
tolerance. It allocates enough common integer padding to keep the entire delay
kernel causal, including requested negative advances; all playback channels
receive that common latency. For a half-sample delay starting at tap zero,
the padding is 64 samples (1.333 ms at 48 kHz). The new convolution stage is
placed before routing, and retains `gd_requested_delay_ms`,
`gd_effective_delay_ms`, `gd_common_padding_samples`, and
`gd_usable_band_max_hz`. The group-delay summary reports effective delays,
and temporal evidence is recomputed from the resulting coefficients. Existing
FIR sidecars are preserved. Nyquist is outside the fractional-delay guarantee;
an unsupported measured playback band is reported as an unapplied GD stage.

The required `just qa-roomeq-camilladsp-backend` gate also renders packaged
fractional-delay exports with CamillaDSP at 44.1, 48, and 96 kHz. It checks
-0.5, +0.25, and +0.5 sample offsets with original FIR impulses at taps 0 and
17, after removing the source sidecar directory. Each of these 18 renders is
checked at 1,025 frequencies from 20 Hz through 0.46 times sample rate against
an independent pure-delay reference and the reported complex response. This
is sampled-band evidence, not continuous-frequency or Nyquist certification.

The machine-readable QA inventory is
`crates/roomeq-qa/src/registry.json`. Registered
scenarios declare their configuration, solver, processing modes, execution
tier, claimed features, and quantitative expectations. The `pr`, `nightly`,
and `weekly` recipes select cumulative cost tiers and do not maintain separate
case inventories. Stochastic QA runs five deterministic seeds and selects the
median explicitly accepted result (or the median of all runs if none is accepted).
Every seed's acceptance decision, reversion/degradation reasons, optimizer
termination/budget evidence and final scores are retained in
`metadata.qa_seed_distribution`, acoustic current/candidate reports, and the
append-only `target/qa/roomeq-seed-distributions.jsonl` artifact. The
`accepted_useful_rate` counts explicitly accepted runs with improved delivered
scores; `safe_output_rate` conservatively counts finite outputs with explicit
final acceptance and no failed stage. Reverted or missing acceptance does not
establish safety, even when the pipeline returns successfully. This rate is not
independent backend certification. Neither rate is replaced by the selected
median or score spread. Missing acceptance evidence does not count as useful acceptance.
Completed JSONL records retain the optimizer configuration, requested seeds,
sample rate, and a separate `final_artifact` verdict with the rerun's scores,
acceptance, optimizer evidence, and stage outcomes. Population rates describe
the five selection runs, not the final rerun; `final_artifact_delivered` means
the pipeline returned successfully, not that the correction was accepted.
Seed execution errors and non-finite scores fail the QA run after all five
selection seeds have been attempted. The JSONL artifact retains the completed
outcomes and identified errors with `status: failed` and
`phase: seed_selection`; rates retain the five-seed denominator, and failed
seeds count as neither useful nor verified safe. An error while rerunning the
selected seed for final artifacts is recorded separately as
`phase: selected_artifact_run`, preserving the selection population and marking
`final_artifact_delivered: false`. Its population rates do not certify the
failed final rerun. Failure to write evidence also fails QA.
Runtime safety
fallback is reported as the distinct `REVERTED` outcome and passes only when
explicitly permitted by the registry. The full registry-driven FEM matrix and
the retained generated-data integration matrix also run on the scheduled
weekly workflow.

Acoustic corpus current/candidate scoring and robustness rescoring use the
runtime physical-seat replay contract. Native training captures are snapshotted
before optimization; held-out CSVs are not reduced to a display grid. Each
logical source is reconstructed from its contributing physical outputs, with
structural routing retained in the correction-disabled baseline. Missing physical
branches, required phase, or sidecars fail replay. Reports retain
`playback_evidence` with source/seat/output identity, baseline and delivered
curves, and summation support evidence. Robustness perturbations operate on
physical captures before summation; seat dropout removes the same seat across
all outputs. Declared upper-band bounds are conservatively increased by the
perturbation's SPL limit and that derivation is retained in their evidence IDs.
These software replay checks do not supply missing corpus captures or establish
unmeasured spatial performance. Existing acoustic baselines are not automatically
recalibrated when scoring changes.

Held-out corpus descriptors accept `seat_id`, an explicit shared listening
position ID, alongside the physical-output `channel` and capture `path`.
Coherent multi-output evaluation requires it. Named capture sets are sorted by
seat ID before replay; every contributing output must have the same ID set.
Duplicate output/seat pairs, mixed named/unnamed captures, and mismatched sets
are errors. Runtime seat indices (including support-bound declarations) follow
this sorted order; reports retain the ID as `seat_label`, including after
robustness dropout. IDs must come from acquisition provenance, not filename
guessing. Legacy independent-channel descriptors may remain unnamed and do not
thereby establish cross-output position identity. A physical sub capture need
not itself be selected as a logical source for scoring.

Acoustic quality shape normalization fits a constant gain using the same
log-frequency trapezoid weights as its residual RMS and mean seat spread.
Adding redundant bass samples therefore does not move the broadband reference.
Pooled bin percentiles remain explicitly distinct from frequency-integrated
statistics. Shape scores alone do not establish preserved output level.
Quality alignment retains the union of supplied pre/post/target grid points,
so a candidate-only native cancellation bin is not lost by baseline-grid sampling.
Useful-output evidence includes the worst sampled unexplained loss and runs of
consecutive samples exceeding the recorded 3 dB diagnostic threshold. These
`loss_bands` use the fixed evaluation support, calibrated target, and permitted
gain; they do not classify authorized attenuation as loss. Band endpoints are
sample locations, not a continuous-frequency guarantee. The diagnostic threshold
does not replace the separate runtime RMS-loss policy.
QA peak/dip metrics use a fixed band derived from the uncorrected measurement;
outer roll-offs can narrow that band, but internal holes and new candidate
cancellations remain eligible even below 20 dB relative to the response peak.

Gate purpose is part of the acceptance contract. A `safety` case may accept a
runtime `REVERTED` result only when it explicitly enables safe reversion.
`functional` and `quality` cases must retain the requested correction; a safe
fallback is not evidence that the processing mode or quality objective works.
Those tiers therefore require zero unexpected reversion.

`just qa-roomeq-contract-pr` runs deterministic realization, CTC replay,
main/sub role, registry-semantics, and measured Genelec 5.1.4 cross-mode
contracts with an explicit optimizer budget. Nightly and weekly quality runs
use the larger convergence budget, five-seed median, and broader matrices; the
PR contract uses one fixed seed. A reachability test
also fails if a runner declared by the registry disappears from CI and
scheduled workflows.

The blocking `qa-roomeq-ci` companion recipe runs quick safety coverage,
multi-seat guards, and perceptual contracts. The randomized five-seed quality
fuzzer remains scheduled nightly/weekly so its convergence runtime and random
case difficulty cannot stall deterministic PR feedback.

Optimizer, routing, crossover, and realization changes should observe a
48-hour pre-release stabilization window with the deterministic contract on
every change and completed scheduled quality runs. Release reports should
classify escaped defects as optimizer/objective, role/routing, DSP realization,
acceptance/reporting, or automation reachability, and record unexpected
reverts, cross-mode drift, mutation survivors, and suite runtime.

### Audibility acceptance contract (2026-09-20)

This contract records the Stage 0 decisions and explicit deferrals for the
2026-09-05 and 2026-09-16 reviews. A deferred validation item does not authorize
enforcement or establish inaudibility. Stage 2 must supply the evidence below
before a perceptual claim can be promoted. These decisions add no configuration
fields and change no defaults or numerical acceptance limits.

**References and comparison tasks.** Pruning compares each proposed kept chain
against the same frozen, ordered full chain `F0`, before any removal. Record its
identity and the removed filters' original indices and coefficients so rollback
restores the chain. Check both the incremental removal and the accumulated change
from `F0`; reoptimization must not reset the reference. A harmful-filter removal
that intentionally changes quality needs a separate quality acceptance decision.
Static PEQ equivalence alone does not establish equivalence of the routed export
or of a transition between live configurations.

Correction quality uses a separately declared desired reference. Below the room
transition, its construction must specify the target and seat aggregation; above
the transition, it must specify which direct-sound response, target tilt, timing,
and spatial relationships are preserved. A universal construction when direct or
anechoic data are absent is **deferred**. The selected target curve remains an
engineering objective, not evidence of a preferred or perceptually transparent
reference. No default reference change follows from this contract.

**Model and supported claims.** The current `heuristic-erb-proxy` evaluates a
filter-composed magnitude shape and nominal-level loudness differences. Its
`sones-experimental-proxy` values are not reference ISO loudness, a validated
signal-pair difference measure, or portable detection thresholds. Its acceptance
records remain `Low` confidence, including explicitly enabled experimental
removals. Equal total loudness does not establish equal timbre. Selection and
version pinning of a validated auditory comparison model, its reference
implementation, supported attributes, and presentation domain are **deferred**.
PEMO-Q is a candidate to evaluate, not a selected room-EQ preference model.

**Conditions, calibration, and measurements.** The acceptance condition set must
explicitly identify seats/ears, programmes, playback levels, sample rate, and
measurement versions. Record trustworthy frequency support, grid and smoothing,
repeatability, channel identity, phase/time reference, and uncertainty. Unknown
conditions cannot count as passing conditions. Magnitude-only data support
magnitude diagnostics; phase, temporal, and spatial claims require appropriate
complex responses or time-referenced RIRs. Noisy, sparse, or phase-uncertain data
must restrict the supported claim or yield insufficient evidence.

`listening_level_phon`, including its nominal 75-phon fallback, is an assumption,
not a physical SPL calibration. Acoustic calibration must bind digital playback
level to measured SPL at the declared seat, with measurement method and
uncertainty. Assumed levels may support advisory sensitivity analysis; they do
not establish calibrated masking or authorize a validated removal. Selection of
the physical calibration procedure, programme corpus and hashes, playback-level
range, and minimum input-quality tolerances is **deferred**. The existing
experimental opt-in retains its explicitly unvalidated status.

**Cumulative policy and uncertainty.** Accept a simplification only when every
declared condition satisfies its incremental and cumulative limits against
`F0`; do not let an average hide a failing seat, programme, or level. Retain the
filter when a condition is missing, unsupported, non-finite, or uncertain. Keep
identity/zero-filter output reachable when all checks pass. The existing scalar
proxy quantum, optional cumulative cap, and local-bin cap are implementation
guards, not scientifically validated all-condition budgets. Their numerical
values are unchanged. Validated perceptual budgets, confidence margins,
worst-seat tradeoff allowances for quality-changing correction, and independent
justification of engineering limits are **deferred**.

The engine's `adjudicate_veto_removals_for_conditions` accepts explicit condition
spectra on a shared grid and freezes each full-chain level anchor. Every removal
checks the worst incremental condition and the configured sum or maximum of
cumulative differences. Missing, duplicate, non-finite, or misaligned evidence
retains filters. Reference identifiers bind the frequency grid, ordered filter
responses, condition identities, background spectra, and nominal levels. This
implements experimental condition evaluation, not validated auditory equivalence.

Native single- and multi-measurement EQ workflows accept the optional
`optimizer.pruning_budget.evaluation` declaration, version `spectral-v1`.
It names every supplied measurement in input order and supplies programme spectra
and nominal listening levels. Each condition combines the original measured
response with the programme spectrum, interpolated in log frequency within their
shared support. The complete measurement × programme × level product is checked,
including measurements assigned zero optimization weight. Missing measurements,
unsupported frequency ranges, or invalid evidence retain filters.

With `evaluation`, an empty `pruning_budget.conditions` uses that complete product;
a nonempty list must match it exactly. IDs have the form `seat-0/music/75phon`.
Without `evaluation`, the legacy `flat-background` proxy remains available;
other declared IDs remain unresolved and retain filters. This declaration does
not supply physical calibration or establish perceptual validity.

Routed systems and workflows with held-out validation captures defer removal
until final graph selection. The final pass replays
each declared condition through the complete exported topology, checks individual
and correlated logical inputs, and reruns electrical and acoustic acceptance.
It requires an explicit evaluation declaration and complete measured phase for
correlated playback. Held-out captures are evaluated as an additional partition;
an incomplete partition retains F0. Its Low-confidence proxy
assessment is recorded under `final_routed_graph`; advisory mode preserves all
processing while recording the same hypothetical removal walk.

The export matrix covers native single/multiple measurements, adaptive/single-pass
selection, local refinement, scalar DE/Pareto NSGA-II selection, and
advisory/enforced modes. Serial and frequency-split hybrid rows additionally
verify the exported FIR samples and IIR removal/preservation. Native crossover
IIR optimization retains every original measurement. Final-seat phase availability
comes from all actual captures and replays, not the phase-free power average; a
missing seat phase still prevents FIR/hybrid acceptance. Native JSON export and
delivered-response replay also cover stereo bass management with one or two
physical sub outputs, in advisory and enforced modes. These static rows do not
establish continuous-refinement transition safety.

Adaptive veto paths now preserve the full pre-pruning chain until cumulative
adjudication, including report-only runs. The explicit raw-loss fallback retains
its legacy engineering semantics and does not establish audibility acceptance.
`just qa-roomeq-pruning-conditions` exercises a two-seat, two-programme, two-level
engine matrix, frozen level anchoring, cumulative overlap, cancelling filters,
narrow peaks, identity output, rollback, missing conditions, and the adaptive
advisory path. The same recipe now includes native workflow/export rows for
single/multiple measurements, adaptive selection, Pareto selection, local
refinement, all-channel multi-seat, routed single/two-sub playback, and hybrid
IIR/FIR. These static software checks do not establish continuous-transition
safety or replace Stage 2 listening validation.

Engineering gate results and unresolved defects are tracked in
[the 2026-09-16 review evidence log](ROOMEQ_REVIEW_20260916.md). The all-channel
multi-seat export rows pass, including zero-weight seats. Kautz/warped reports
now use the serialized topology, including the Kautz playback consumer's
unity dry path (`x + bank(x)`). Kautz section weights are linear coefficients,
not PEQ dB gains; a zero-weight bank therefore passes the dry input unchanged.
The evaluator also preserves legacy single-section settings and rejects malformed
or duplicate section declarations. The in-repository fixed-pole fitter now
searches linear coefficients against the actual dry-plus-bank dB magnitude,
uses prepared targets, and rejects candidates violating sampled composite gain
bounds. It uses deterministic coordinate sweeps bounded by `max_iter`, not the
selected PEQ optimizer backend. Unity is retained when no feasible improving
move is found; gain bounds must contain unity. The unchanged multirate budget
canary now passes, but the unshifted analytic peak's RMS remains slightly worse.
This is not a continuous-frequency bound, optimizer-quality guarantee, or
evidence of perceptual benefit. Kautz no longer populates legacy Biquad summary
fields with linear weights disguised as dB gains. Inspect its serialized bank
for section counts and parameters, and use the serialized response for replay.
Python filter decomposition/text paths still require topology-aware handling;
the absence of a PEQ cache must not be interpreted as absence of correction.

Final correction-strength selection scales Kautz's nested linear weights,
not its unused top-level `db_gain` placeholder. Strength `s` gives
`H_s = 1 + s * (H_full - 1)`: zero is unity, while intermediate strength is
complex-transfer interpolation rather than dB interpolation. Poles and the
allpass chain are preserved, and every candidate still requires final replay.

Kautz pole selection now respects the requested section count, observation and
optional narrower correction band, Nyquist, and supported Q bounds. Excess modes
are selected by prominence and then ordered by frequency before constructing the
basis. Low-weight sections are retained: their allpass stage still affects later
sections. These are pole-inventory constraints, not proof of out-of-band neutrality.

**Listening study and Stage 2 acceptance.** Listener population, recruitment,
sample size, and the powered equivalence/detection bound are **deferred**; no
completed listening study is claimed. Before collecting outcomes, preregister
those choices together with calibrated levels, task-specific level matching,
randomization/blinding, repetitions, exclusions, and uncertainty analysis. Keep
pruning detectability/equivalence separate from correction preference. A
nonsignificant ABX result alone does not prove equivalence. Independently inspect
actual playback headroom, channel balance, and timing rather than hiding changes
through listening-test normalization.

Stage 2 must pin the model and validate published reference cases, render actual
comparison audio, and include speech, music, sustained tones, exposed transients,
equal-loudness/different-timbre cases, narrow resonances, overlapping/cancelling
filters, and accumulated small removals. Separate development from held-out
seats, programmes, levels, and rooms. Verify measurement repeatability, no-EQ
cases, main/sub interactions, and the final routed/exported chain. Stimulus
descriptors, synthetic tests, measured transfer curves, and export byte identity
each prove their own software contract; none substitutes for listener evidence.

**Reporting, cost, and rollout.** Use the existing independent outcome,
enforcement, and confidence fields: `keep`, `candidate_removal`,
`accepted_removal`, `risk_limited_correction`, or `insufficient_evidence`;
`not_evaluated`, `advisory`, or `enforced`; and the stated evidence confidence.
Record model/version, calibration or assumption, reference identity, thresholds,
condition coverage, and reasons. An experimental `accepted_removal` records an
applied change, not proof of perceptual equivalence. Do not turn a skipped or
unsupported assessment into a successful one.

The validated evaluator's runtime budget and versioned policy/configuration
design are **deferred**. Cache fixed reference/probe transforms and evaluate
candidates in stages when that evaluator is introduced; a timeout must retain
the filter and report incomplete evaluation. The auditory reranker remains
blocked on Stage 2 evidence and is not a prerequisite for independent physical
safety fixes. No new backend or latency rewrite is selected. `flat` remains the
default loss; absent veto configuration remains disabled, and configured veto
remains advisory by default. A future preset recommendation needs the claimed
release-gate evidence, held-out validation, and recorded listening outcomes for
any listener-benefit claim. Native/SOTF playback equivalence remains an external
proof obligation; in-tree export checks do not discharge it.

### Staged rollout and release gates

New audibility and acceptance policies roll out in three behaviors:
`Legacy` (policy disabled — the output it replaced, always available),
`Advisory` (report-only: evaluates and records reason-coded verdicts but
never changes output — the default whenever a new policy is selected), and `Enforcing`
(explicit opt-in that changes emitted output). Deleting a policy selection
restores legacy behavior; enforcement is never the default.

Promotion is evidence-gated by four independent release gates
(`crates/roomeq-qa/src/release_gates.rs`): implementation correctness,
physical safety, perceptual-model validation, and demonstrated listening
benefit. Passing one gate never implies the others. Advisory releases and
elapsed warning cycles carry no evidence and never promote. Physical
safeguards promote on correctness plus physical safety alone — the
optional rerank objective (Stage 4) never blocks them. Perceptual claims
additionally need perceptual validation; listening-benefit claims need
recorded listening outcomes.

Every staged artifact carries provenance for its stage: stimulus
manifests record renderer, version, platform, SPL mapping, and file
hashes; validation results record the preregistration hash. Trial imports
additionally bind the frozen listening setup hash, covering
the exact chain/stimulus binding, population, absolute playback level,
matching method, programme classes, and holdouts. Protocol-only imports
remain software evidence; a self-declared nonsynthetic table does not prove
that listeners ran the trial. Rerank reports record evaluator and loss
pins; export round trips
(`roomeq-export/src/roundtrip.rs`) verify biquad coefficients,
routing, preamp normalization, delay, and convolution bytes against the
canonical graph. Demo the gates and round trips without listening
evidence via `roomeq-qa-synthetic --release-gates`.

### Multi-sub measurement resolution and output preservation

Bass measurements ending at or below 500 Hz retain every native sample and
measured phase. Full-range measurements retain their native bass samples while
using the configured reduced grid above 500 Hz. MSO, DBA, and spatial MSO retain
all measured frequency points in their common supported band. Interpolating a
summed magnitude/phase curve is not equivalent to summing individual responses;
optimization and routed reconstruction sum the drivers on the receiving grid.
The measurement resolution remains the limit on knowledge of unmeasured nulls.

A sub measurement ending at 200 Hz does not truncate full-range main analysis.
For prediction above a tail at least 24 dB below its measured peak and falling
at least 12 dB/octave, the model continues a falling envelope (capped at
48 dB/octave). An energetic endpoint is held conservatively. Sub alignment and
EQ remain inside the measured, useful bass band; this prediction extension is
not new measurement evidence and is never used to extend the sub EQ band.

The useful upper bound is the last measured sample within 20 dB of the in-band
peak. An internal room null therefore does not truncate later useful bass.
Crossover evaluation includes the native sub samples even when the receiving
main measurement has a coarser grid.

Independent subs driven by the same input are summed coherently when measured
phase is available. Without phase, preprocessing explicitly reports a power-sum
approximation; phase-critical route verification requires suitable phase data.
All-pass and multi-seat routed processing use the dedicated sub engine, preserve
per-driver PEQ/all-pass, polarity and primary-seat measurements, and retain the
configured global-EQ policy instead of adding another single-seat sub EQ pass.

DBA, ordinary/all-pass MSO and continuous-area scalarizations penalize loss of
useful output, deep new nulls, low-band loss and unnecessary gain. DBA here
optimizes measured magnitude with an inverted rear array; room geometry and
late-energy measurements are still needed to establish rear-wall absorption.
The cardioid path is a fixed geometric delay/inversion recipe; matching and
directional measurements are needed to establish rear rejection. Its legitimate
low-frequency efficiency loss is not treated as an MSO defect.
`primary_with_constraints.max_deviation_db` is a soft penalty threshold, not a
hard guarantee at every seat and frequency.
Multi-seat cardioid inputs retain the measured phase of each front/rear pair.
Both branches must supply the same ordered seats; missing phase or unmatched
seat counts reject preprocessing. The combined responses remain separate for
shared EQ, while the configured primary seat supplies the routing reference.
Both physical outputs must be declared. This does not establish measured
directional rejection or perceptual validation.

Non-routed groups may expose several physical outputs under one logical DSP
chain. Final-seat replay resolves each declared output to that chain while
retaining every driver's seat evidence. Missing or ambiguous ownership rejects
the result; legacy multi-seat driver groups require explicit stable driver IDs.

Final-seat scorecards assess each logical input over its measured band. Independent
mains and subwoofers may occupy disjoint bands; in that case the aggregate omits
`measurement_overlap_hz`, while each `final_seats` entry retains its actual band.
Seats of the same input must still share supported frequencies. The aggregate's
`evaluated_band_hz` encloses its individual bands and does not claim measured
support in gaps between them.

## Prepared FIR target grid contract

The internal prepared-FIR API requires target frequencies to match the
measurement frequency grid exactly, with one level per frequency in each curve.
It rejects mismatches before pointwise boost capping or filter design. Callers
with independent target grids must align them during target preparation; changing
array order or silently pairing samples by index is not a resampling operation.

## Final-graph electrical headroom assessment

The output metadata includes a `final_graph_sampled_electrical_headroom` stage
after final level alignment, CTC and convolution artifact binding. This is
separate from acoustic/correction acceptance: an accepted correction can still
need an explicit playback gain budget.

The assessment assumes independently phased sinusoidal logical inputs, each
with peak amplitude 1.0. It evaluates complete serialized input/route/output
transfers at 8193 points from DC to Nyquist, plus serialized EQ and crossover
centers. Per-output checks report the sampled amplitude against full scale;
their diagnostics include peak frequency, required attenuation, grid size and
input limits. A `degraded` stage identifies sampled overload or unavailable
electrical evidence. Unsupported global processing or unresolved sidecars is
not silently omitted and does not receive a passing electrical check.

This report does not apply attenuation, alter the configured correction
acceptance policy, or certify frequencies between samples, transient/true-peak
headroom, native-backend agreement, acoustic benefit or device assignment.
Required attenuation must be reconciled with calibrated useful-output goals;
it is not a recommendation to attenuate blindly.

## Per-driver FIR placement (input schema 2.2.0)

Set `optimizer.fir.placement` to `per_driver` to try separate FIRs for the
physical speakers in each independent main/sub group. Shared routed sub outputs
deploy the common post-route kernel on each physical driver, preserving every
logical-input transfer. This constrained shared-kernel design is not independent
multi-input matrix inversion. Pre-route source filters remain on their owner.
The default, `shared`,
is unchanged. Phase mode is independent of placement. See
[the input contract](../src/bin/roomeq/INPUT_FORMAT.md#configuration-schema-version)
for the JSON fragment, supported group types, safeguards and limitations.

The engine retains crossover, gain, delay and intentional IIR stages, then
evaluates complete sets of per-driver FIRs against the calibrated acoustic sum.
It does not flatten every physical speaker towards the full-range target.
Each exported driver's plugin list references its own WAV; load all of them on
their respective physical outputs. Mixed-phase FIRs remain phase-only, and all
branches have a common causal support budget. Minimum-phase FIRs remain causal
without imposing a linear-phase centering delay.

Conservative null masks prevent FIR boost into deep local dips, low-coherence
bins and low signal/noise regions. A remaining target deficit at a protected
null is intentional, not an invitation to increase boost. Single-position
measurements cannot definitively distinguish all SBIR/room nulls. Relative
main/sub phase correction can help interference, but cannot guarantee room-wide
improvement. Validate other seats and listen at matched levels.

Serialized capture curves preserve optional `noise_floor_db` and `coherence`
arrays on the capture frequency grid. These now survive the physical-driver
handoff to FIR design; old files without them retain unknown quality, not an
assumed low noise floor or perfect coherence. Curve conversion does not
normalize levels. Any level-reference change must shift the noise floor with
SPL. These fields describe capture evidence, not a new post-correction noise
measurement or, by themselves, authorization for excess-phase correction.

Joint per-driver `mixed_phase` and Kirkeby with `correct_excess_phase: true`
now require a supported excess-phase assessment for every capture, using
`optimizer.mixed_phase.assessment` (or its documented defaults). The design
uses bounded strengths of the assessment's tapered correction phase, not a
separate unassessed decomposition. Missing/insufficient measured noise-floor
SNR, measured poor coherence, or excessive narrow/wide smoothing sensitivity
rejects the explicit request before artifacts are written. Smoothing sensitivity
is a numerical diagnostic, not a substitute for repeated windowed recordings.
Magnitude-only modes remain available separately; no implicit mode downgrade
is performed. Assessment success alone does not establish direct-sound target
eligibility, measured playback improvement, or complete temporal-budget support.

For physical-driver placement, the explicit opt-in
`optimizer.mixed_phase.assessment.retain_magnitude_on_refusal: true` retains
the existing magnitude-processing chain when the phase assessment or source-bound
direct-sound target policy refuses the phase action. Default `false` preserves
abort-on-refusal. No substitute magnitude FIR is generated, no phase sidecars
are written, and the final ledger records insufficient evidence rather than
applied phase correction. The retained graph must still pass normal final safety
checks. This does not waive coherent capture phase/timing, malformed-data or
configuration errors, or artifact failures; shared-FIR behavior is unchanged.

Before joint excess-phase design, the workflow also applies the target-chain
direct-sound guard to the active correction band for every contributing source.
Upper-band detail requires assessed direct-sound and angular evidence covering
that band; a room curve and good SNR alone do not authorize it. DBA aggregates
check every cabinet, not just one representative per array. The configured
Schroeder transition is used, with the target model's documented 300 Hz fallback
when unavailable. Refusal leaves the requested target and chain unchanged and
writes no new FIR sidecars. These checks assess declared capture facts; they do
not authenticate recordings or establish measured playback benefit.

Successful joint designs carry provisional source/target decisions into final
reconciliation, including selected phase strength, requested design band,
joint target error, causal delay, and the full set of branch FIR references.
The design band is not labeled as exact finite-FIR support. A zero phase
selection is reported as unresolved, not as applied, already acceptable, or
physically impossible to improve. Removing a referenced FIR or swapping its
branch prevents the original joint phase claim from remaining applied;
equivalent replacements need reassessment. Explicit admission failures still
return an error without a successful output ledger. Complete-chain temporal
budget coverage remains a separate requirement.

Sidecar packaging updates final phase-decision file references and driver/file
bindings to the delivered package names, including collision-driven renaming.
Original provisional records and unrelated capture references remain unchanged.
Renaming preserves resource bytes; it does not reassess or authorize a changed
filter. Stale source graph or bound resource content is still rejected.

The measured `2.2_sigberg2/optimiser-fir.json` opts in. Use `placement: shared`
for an A/B run in a separate output directory. Compare the complete replayed
response, not an individual driver's SPL against the whole-system target.
Automatic shared/per-driver selection is not implemented. Routed systems with
shared physical subs are explicitly rejected by this initial option.

## Verification and diagnostic modules

The following library modules implement the verification roadmap
(`reviews/next-20260921.md`). They are diagnostic and objective building
blocks; none of them changes the correction pipeline on its own.

| Module | Purpose |
|---|---|
| `crates/autoeq-measurements/src/direct_sound.rs` | Capture facts for direct-sound evidence: gate interval from `(rR−rD)/c`, valid-band bound, angular coverage, averaging; moving-microphone averages refused as phase sources, unknown fails closed. |
| `crates/roomeq-analysis/src/quasi_anechoic.rs` | Quasi-anechoic validator (`DetailEligible`/`TonalOnly`/`Unsupported`) with phase-source verdicts and gate labeling mapped onto the K2 eligibility lane. |
| `crates/roomeq-analysis/src/excess_phase.rs` | Excess-phase decomposition: log-frequency-smoothed minimum-phase estimate versus measured phase, coherence- and SNR-weighted, with a coarse-to-fine bulk-delay fit. |
| `roomeq-engine/src/summation_search.rs` | Exhaustive per-sub delay/polarity search over the complex transfer matrix (`Hsum` band search, delay ledger, per-seat combined re-verification), with the workflow-side MLP search and per-seat reconciliation in `roomeq-workflow/src/crossover_summation.rs`. |
| `autoeq-optim/src/loss/joint_multisub.rs` + `roomeq-engine/src/multisub/joint_objective.rs` | Joint multi-sub objective over the transfer matrix: seat variance, acoustic output-loss proxy (not calibrated physical drive), and target error. Array acceptance also checks each retained seat against its baseline. A shared-EQ residual ledger refuses double application of LFE vs redirected gains; per-seat scorecard in `roomeq-quality/src/joint_sub_scorecard.rs`. |
| `roomeq-quality/src/promotion.rs` | Candidate promotion policy: programme holdout discipline, four-family control summary, `EnforcementReadiness::check_ready` (fail-closed `blocked_external` without real-trial evidence), and intent-level equal-loudness guard; perceptual signal-pair scaffolding in `autoeq-optim/src/perceptual_promotion.rs`. |
| `roomeq-quality/src/battery.rs` | Listening battery protocol: mono/spatial/L+R-sum presentations, resonance/transient/programme material, per-cell level match, concealed randomization, preregistered alpha/power/sizing; nonsignificant ABX never proves equivalence and synthetic verdicts never promote. Rendered-byte binding in `roomeq-workflow/src/listening_stimuli.rs`. |
| `roomeq-model/src/target_transition.rs` | Smooth logistic target/transition architecture (separable calibration/tilt/level stages; damage guard). Chain validation requires calibration before optional tilt/level stages. `roomeq-engine/src/target_enforcement.rs` and `roomeq-workflow/src/target_enforcement.rs` currently connect this policy to phase assessment; ordinary magnitude-proposal integration remains incomplete. The logistic regime weight alone is not a smoothly tapered realized correction. |
| `roomeq-quality/src/acceptance_bundle.rs` | Multi-view acceptance bundle: magnitude, common-reference IR/step, octave-band ETC, frequency-resolved decay, calibrated ambient noise, and routed-chain headroom under one matched `ViewSettings` hash. Partial bundles fail the gate. |
| `roomeq-export/src/acceptance_views.rs` | Verifies actual view payloads using `sha256-json-typed-v1`, matching settings/graph provenance, and unique names. Production export packages bind carried views to the rendered main artifact's SHA-256 in an acceptance-view sidecar. |
| `roomeq-engine/src/dsp_conventions.rs`, `autoeq-core/src/dsp_conventions.rs` | Executable DSP convention checks: FFT normalization roundtrip, window scaling, linear-vs-circular padding, group-delay sign, FIR latency, multi-rate delay preservation, biquad/SOS stability, PEQ sign and units. |

Acceptance-bundle gates validate internal trace/settings/provenance consistency
and supplied headroom evidence, not capture authenticity or correction benefit.
IR/step sample spacing must match the declared rate; ETC traces must cover every
declared band and time window. Headroom requires exact coverage of an explicit
physical-output plan, calibrated comparable demand/limits, and finite computed
margins. Blank or `uncalibrated` calibration identities cannot pass, including
whitespace/case variants of that sentinel. Two polarity inversions cancel in
derived DSP dispositions. Legacy missing-output plans and hash-only view inputs
remain readable but cannot establish complete or verified evidence.

Finalized workflow output carries `correction_decisions.acceptance_evidence`,
a versioned, separately hashed quality payload outside the graph's own identity
boundary. It computes matched 1/12-octave general and 1/24-octave fine magnitude
views from retained pre/post channel response pairs, and derives serialized DSP
gain/delay/polarity dispositions. These are **retained predictions**, not raw
acquisition, independently replayed routed outputs, or absolute calibrated SPL.
Existing conditioning remains in the source data; additional smoothing does not
restore lost detail. Mismatched grids or normalization metadata are not silently
zipped or relabeled. Final response rebuilding saves the exact input used for
that realization rather than pairing an extended display curve with a narrower
recomputed output. Views are limited to that retained support.

Reports place these diagnostics after the correction explanation and before the
optimization summary, separately for each comparison mode. They independently
verify graph/resource and view-payload bindings. Missing common-reference raw
IRs, ETC/decay fit evidence, calibrated silent-path noise, physical demand/limits,
and complete routed true-peak trials remain explicit unavailable states. The
serialized partial bundle does not pass the full quality gate or approve playback.

Export packages carrying these diagnostics include
`<main filename>.acceptance-views.json`, bound to the actual main artifact bytes.
Convolution reference rewriting regenerates dispositions and graph identities;
stale source evidence and mismatched analysis/export sample rates are refused.
This association is not proof of independent backend response agreement. Complete
source/seat acquisition, supported raw time-domain views, physical headroom, and
final routed-chain replay remain required for full acceptance.

Playback-verified tiers (before/after correction playback captures with bound
graph identities) remain operator-supplied inputs: no post-correction
re-captures exist in-tree, so those tiers report `blocked_external` with the
exact missing inputs instead of inventing them.

### Required software-backend PCM contracts

The seven CamillaDSP PCM contracts are explicitly ignored in ordinary Rust test
runs because they require an external executable. An ignored test is not backend
verification. Run `scripts/run_camilladsp_backend_contracts.py` with
`ROOMEQ_CAMILLADSP_BIN` pointing to the intended executable (or `camilladsp` on
PATH). The launcher opts these tests in and requires every named contract to
execute successfully. Explicitly selecting them without the environment variable
fails rather than returning a successful no-op.
The launcher bounds the contract run at 180 seconds and, on POSIX, terminates
its launched process group on timeout or interruption. Partial timeout output
is retained in the QA log and the artifact records failure, not a stale pass.

These contracts use stdin/stdout PCM, not audio hardware. Their scope is software
backend response agreement; they do not verify a DAC, loudspeaker, room, dynamic
protection behavior, or perceptual benefit.
