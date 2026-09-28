# Sigberg 2.2: Post-EQ acceptance comparison

Measured case: `data_tests/roomeq/measured/2.2_sigberg1`, IIR override,
seed 42, 48 kHz, default 200 frequency samples. Evaluated on 2026-09-28.
The rerun reproduced the target-shortfall and cancellation values from the
original 11:06:20–11:06:21 UTC log. Acceptance thresholds were not changed.

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

## Actual reasons for rejection

The active failures for both channels are:

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

## Relative-threshold proposal for discussion

Retain an absolute good-enough target, but allow a second route for material
improvement against the immediate input. An illustrative product rule is:

```text
target shortfall <= 3.05 dB
OR
(shortfall reduction >= 20% AND shortfall reduction >= 1 dB)
```

Both candidates pass that illustrative relative rule. The percentages and
minimum dB change are product choices for discussion, not established audibility
thresholds. Keeping an absolute route handles already-good responses and avoids
ratios near zero. A relative rule alone can accept a very large remaining defect.

Changing that rule alone would still reject both candidates on the main-only
score. For bass management, the useful next policy comparison should score the
delivered main-plus-sub response in the crossover region and separately protect
the main response outside that region. It should retain final electrical drive,
headroom, temporal, and per-seat checks. Validated cancellation should be checked
for regression relative to the immediate input as well as reported against the
original configured baseline; an inherited defect should not be attributed to
the added common EQ.

The current candidate peaks of +12.6 and +16.45 dB make a complete routed
with/without finalization comparison necessary before choosing either candidate
for output. This evaluation measured the stage candidates; it did not force
them through the downstream acceptance gates or export them for playback.
The predicted improvement motivates further evaluation, but does not establish
audible benefit, especially with the missing main/sub timing declarations.

## Waveform repair and reproduction

The original IIR run logged the identical Sub1 mapping failure 81 times.
The fixed rerun logged it zero times and produced both Sub1 waveform views.
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

Regression coverage includes complete physical-sub mapping, incorrect names and
indices, missing drivers and captures, duplicate output IDs, mismatched timing,
warning recurrence after recovery, and common-EQ cancellation invariance.
