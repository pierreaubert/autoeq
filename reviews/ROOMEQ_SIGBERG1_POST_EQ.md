# Sigberg 2.2: Post-EQ acceptance comparison and implementation review

Measured case: `data_tests/roomeq/measured/2.2_sigberg1`, IIR override,
seed 42, 48 kHz, default 200 frequency samples. Evaluated on 2026-09-28.
The rerun reproduced the target-shortfall and cancellation values from the
original 11:06:20–11:06:21 UTC log. The initial diagnostic repair preserved
thresholds; the implementation below adds the subsequently approved policy.

## Immediate with/without comparison

Both columns use the same optimized routing, preceding correction, measurements,
target, and crossover. Only the candidate common pre-route Post-EQ changes.
The configured cancellation baseline is a separate, earlier reference captured
before array optimization and EQ.

| Metric | L without | L with | R without | R with |
|---|---:|---:|---:|---:|
| Worst target shortfall, dB | 10.160899 | 3.950030 | 20.113734 | 6.765109 |
| Main/sub cancellation, dB | 6.367721 | 6.367721 | 10.176027 | 10.176027 |
| Combined crossover objective, lower is better | 24.137654 | 11.995450 | 46.835973 | 18.336512 |
| Main-only score, lower is better | 2.576080 | 3.364361 | 2.823535 | 3.689003 |
| Peak gain from this Post-EQ pass, dB | 0 | 12.599675 | 0 | 16.452214 |

Target-shortfall reductions are 6.210869 dB (61.13%) for L and 13.348625 dB
(66.37%) for R. These percentages divide the reduction in dB shortfall by the
original dB shortfall; they are not pressure, power, or loudness percentages.
The shortfall is referenced to the response above the crossover, as in the
existing objective. It is not an absolute calibrated sound-pressure prediction.

Useful-output losses are 0.110674 dB for L and 0.032734 dB for R, both below
the 3 dB budget. This loss check does not establish available boost headroom.
Peak gain is the maximum filter magnitude sampled on the main measurement grid;
it is not the electrical demand of the complete routed graph.

## Reasons for rejection under the original policy

The original active failures for both channels were:

1. Target shortfall remains above the absolute 3 dB limit plus 0.05 dB tolerance.
2. The main-only score regresses, even though the combined main-plus-sub objective
   improves substantially.

Cancellation screening is **disabled in this run**. The saved acquisition
metadata declares shared stationary timing for the subs, but not the mains.
The workflow records `crossover_timing_reference_refused` and
`source_route_optimizer_skipped_unverified_timing`. The calculated cancellation
numbers are therefore not validated coherent predictions for the complete setup.
The old warning incorrectly selected cancellation as the reason without checking
whether that gate was active. The new message lists only active failed gates.

Even on a synchronized grid, common EQ cannot change relative main/sub
cancellation: applying H to both branches gives H(M + B). The measured rerun
shows the same cancellation before and after this pass. Removing this pass does
not repair cancellation inherited from earlier processing.

## Implemented policy and final comparison

The target gate now accepts:

```text
target shortfall <= 3.05 dB
OR
(shortfall reduction >= 20% AND shortfall reduction >= 1 dB)
```

Both stage candidates pass this rule. The percentage and minimum dB change are
product choices, not established audibility thresholds. The absolute route
handles already-good responses and avoids ratios near zero.

The combined objective evaluates the splice region. A separate main-response
check protects frequencies above twice the crossover (144.22 Hz in this run).
It measures error against the requested target, falling back to flatness when
no target is supplied. Active cancellation screening checks the immediate
with/without change; final routed validation retains the configured baseline.

| Main target error above twice crossover | Without Post-EQ | With Post-EQ | Stage decision |
|---|---:|---:|---|
| L | 1.765992 | 1.736634 | Accept |
| R | 1.346301 | 1.856213 | Reject: protected main response regresses |

Final selection explicitly includes alternatives without the optional main
Post-EQ. These retain preceding correction, routing, and physical-sub EQ. The
full L candidate requires 12.428 dB of attenuation, exceeding the 12 dB budget;
its scaled alternatives also fail existing output/boost checks. Final selection
therefore chooses `correction_strength_0.12500_sub_0.12500_output_without_post_eq`.

The published result is accepted, with aggregate score 2.8381773205974468, equal
to the original accepted score. Retained-seat post RMS is 4.981989 dB for L and
4.402420 dB for R. Derived L input trim changes from -0.29 to -0.45 dB, so this
policy run is not a byte-identical DSP export. No optional main Post-EQ is
exported; the physical-sub Post-EQ remains. These comparisons do not establish
audible improvement, particularly with unverified main/sub timing.

Local evaluation artifacts are under
`/Volumes/home_tmp/tmp/autoeq-issue-70/final-selection/` (`dsp.json`, `run.log`).

## Waveform repair and reproduction

The original IIR run logged the identical Sub1 mapping failure 81 times.
The initial diagnostic-only rerun logged it zero times and produced both Sub1 waveform views.
Every exported channel plugin and driver chain remained identical to the
original run. The final aggregate score remained 2.8381773205974468, with the
same accepted result. The mapping fix changes evidence availability without
changing the correction in this case.

```bash
cargo build --release --features cli --bin roomeq
RUST_LOG=warn,roomeq_workflow::bass_management=info target/release/roomeq \
  --config data_tests/roomeq/measured/2.2_sigberg1/recordings.json \
  --override-config data_tests/roomeq/measured/2.2_sigberg1/optimiser-iir.json \
  --output /path/to/separate/output/dsp-iir.json
```

Regression coverage includes hybrid-threshold boundaries, protection outside
the splice window, removal of only optional main Post-EQ, complete physical-sub mapping, incorrect names and
indices, missing drivers and captures, duplicate output IDs, mismatched timing,
warning recurrence after recovery, and common-EQ cancellation invariance.
