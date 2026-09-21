# roomeq-quality — Architecture

Acoustic quality metrics, acceptance policies, and QA fixtures for RoomEQ.
Answers "how good is this correction?" independently of the optimizer that
produced it: candidate DSP is scored from its complex transfer function, not
from plugin presence.

## Layer

Independent scoring boundary over `autoeq-core` + `roomeq-model`. Both the
production pipeline (gates) and `roomeq-qa` (regression) consume it.

## Key modules

| Module | Owns |
|---|---|
| `quality` | `evaluate_acoustic_quality`: pre/post scorecards over an evaluation band (Schroeder-aware), level-normalized |
| `metrics` | Seat aggregation, worst-seat rules, seeded bootstrap intervals |
| `robustness` | Rescoring under noise seeds, SPL-calibration offsets, seat dropout, head-position perturbation |
| `corpus` | `AcousticCorpusManifest` / baseline: repository-backed scenarios (FEM, real captures, synthetic) with tiers (`pr`/`nightly`/…), provenances, held-out seats. Enforced gates need ≥ 2 held-out measurements covering every scored channel; single-position captures stay `report_only` |
| `seeded`, `fixtures`, `stimuli` | Deterministic resampling/noise corpora, analytic ground-truth fixtures, test signals |
| `acceptance`, `final_check` | Correction-acceptance policy and final invariant checks |
| `band_policy`, `chain_constraints`, `inversion_support` | Evaluation-band rules, DSP-chain constraints, excess-phase inversion support analysis |
| `scenario`, `protocol`, `types` | QA scenario descriptors and shared types |
| `capture` | Pure prediction-vs-capture comparison over K5 bindings: identity compatibility first, matched bands only, declared gain/delay in the ledger, per-metric tolerances; simulated/backend vs acoustic and small-signal vs max-output stay distinct |
| `listening` | Frozen listening setups: baseline-vs-corrected and pruned-vs-full arms, mono vs L+R-sum vs spatial presentations, immutable chain/stimulus binding, holdout programmes; intent/claim separation, no outcomes |
| `trial_import` | Real trial import validation (ids, counts, randomization, protocol hash, binding) with Wilson effects, independent exact-binomial cross-check, per-claim success/failure/inconclusive verdicts; synthetic tables stay labelled and never qualify |
| `validation_corpus` | Staged validation corpora with deterministic hold-out splits |
| `electrical_headroom` | Quality-side headroom metrics (pipeline enforcement lives in `roomeq-workflow`) |

## Core abstractions

- **Scorecard in, scorecard out.** `evaluate_acoustic_quality(training_pre,
  training_post, held_out_pre, held_out_post, target, config, temporal)`
  compares pre/post `Curve` slices (never plugin lists) and returns an
  `AcousticQualityScorecard`. `QualityEvaluationConfig { min/max_freq_hz,
  schroeder_hz, normalize_level }` fixes the band and the level convention,
  so scores are comparable across runs.
- **Gates, then baselines.** `evaluate_quality_gate` turns a scorecard into a
  pass/advisory/fail `QualityGateReport`; `compare_quality_to_baseline`
  diffs against the platform-scoped `AcousticCorpusBaseline` (OS + arch key).
- **Manifest with teeth.** `AcousticCorpusManifest::load` canonicalizes the
  manifest path (relative CLI paths resolve correctly), absolutizes every
  scenario path, validates existence, tiers, and the enforced-gate held-out
  rule, and rejects duplicate IDs. `scenarios_for(tier)` selects the
  tier-closed subset (nightly includes pr).
- **Deterministic splits.** `validation_corpus::deterministic_split` hashes
  item IDs with a seed for train/held-out partitions that never degenerate
  (≥ 2 items always split 1-and-1-or-more).

## API

```rust
// Score a correction from curves alone:
let scorecard: AcousticQualityScorecard = evaluate_acoustic_quality(
    &pre, &post, &held_pre, &held_post, target.as_ref(), config, temporal,
)?;
// Gate it (enforce=false keeps failures advisory, e.g. report-only scenarios),
// then compare against the calibrated baseline metrics:
let gate: QualityGateReport = evaluate_quality_gate(&scorecard, policy, enforce);
let delta: QualityBaselineComparison =
    compare_quality_to_baseline(&scorecard, &baseline_metrics, regression_policy)?;

// Load the repository corpus (validates everything on the way in):
let manifest = AcousticCorpusManifest::load(Path::new("data_tests/roomeq/acoustic_corpus/manifest.json"))?;
for scenario in manifest.scenarios_for(QaTier::Nightly) {
    println!("{} [{}] held_out={}", scenario.id, scenario.provenance.as_str(), scenario.held_out.len());
}
```

## Consumers

`roomeq-engine` (in-loop scoring), `roomeq-workflow` (gates), `roomeq-qa`
(regression matrices).
