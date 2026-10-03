# Perceptual metric status

Source review: AutoEQ backend revision `e5463bd913bc6098796a4d55212b51a0415ea57c`.
The programme holdout safeguard is verified separately in `e6df215`.

These classifications describe the inspected implementation and its available
validation. A numerical regression verifies a software calculation; independent
model agreement and relevant listening evidence are additional requirements.

| Metric or path | Classification | Implemented input and meaning | Current limit |
| --- | --- | --- | --- |
| Active EPA optimizer objective | Heuristic spectral objective | Transfer-response residual on the configured frequency band, ERB/band weights, optional deadband and smoothness | The active objective uses spectral flatness. Reported loudness, roughness and modal descriptors do not steer this objective or establish listener preference. |
| EPA report dimensions | Heuristic diagnostics | Transfer-response predictions with an assumed listening level and target sharpness | Evaluation, potency, activity and composite preference are model descriptors. Programme waveform, listener population and independent domain validation are absent from this transfer-only input. |
| EPA 24-Bark loudness | Heuristic diagnostic | Simplified auditory-band calculation, with the response anchored at an assumed 1 kHz listening level and scaled to a defining 40-phon test | Passing the defining reference and monotonicity checks does not establish ISO 532 conformance, calibrated programme loudness, or a validated free-field/headphone domain. |
| Per-filter ERB audibility veto | Experimental heuristic | With/without filter magnitude difference, affected ERB width and a simplified masked-loudness delta on a nominal phon background | Report-only by default. Removal requires the explicit experimental acknowledgment and cumulative full-chain checks. The proxy and thresholds have no demonstrated listener validation; a keep/remove nomination is not a measured audibility result. |
| Modal temporal penalty | Experimental heuristic | Detected mode prominence/severity and the candidate's magnitude change, with configured material-profile weights | Available for offline/report diagnostics; excluded from active EPA fitness. Magnitude-derived modal properties cannot establish measured decay or programme masking. |
| FIR peak and unmasked pre-main energy | Physical numerical descriptors | Actual FIR coefficients, sample rate, and the selected maximum-amplitude main sample | They describe the supplied impulse and reference convention. They do not establish acoustic room decay or audibility. |
| FIR masked pre/post energy and penalty | Experimental heuristic | FIR energy weighted by configured masking windows, material profile and threshold, treating the selected main peak as a transient masker | The windows and material labels are assumptions. The values are model predictions, with no independently validated programme/listener domain supplied here. |
| PEMO-Q fidelity promotion | Blocked pending independent inputs | Contracts require a pinned implementation/edition/license/vector set, calibrated input, approved numerical agreement, domain and controls | The default approved-reference registry is empty. Descriptions and synthetic protocol fixtures cannot authorize a validated model or listening claim. |

## Implementation anchors

- Active objective: [EPA strategy](../crates/autoeq-optim/src/optim/loss/strategies/epa.rs)
  and [spectral loss](../crates/autoeq-optim/src/loss/epa/score/epa.rs).
- Diagnostic loudness: [24-Bark wrapper](../crates/autoeq-optim/src/loss/epa/loudness.rs).
- Report meaning and assumed level: [report contracts](../crates/roomeq-model/src/report_contracts.rs).
- Experimental filter removal: [veto configuration](../crates/roomeq-model/src/config/filter_audibility_config.rs)
  and [engine evaluation/adjudication](../crates/roomeq-engine/src/eq/audibility_veto.rs).
- Modal and FIR temporal predictions: [temporal diagnostics](../crates/autoeq-optim/src/loss/epa/score/temporal.rs)
  and [configuration](../crates/roomeq-model/src/optimizer_settings.rs).
- Fidelity input and model pin: [promotion contracts](../crates/autoeq-optim/src/perceptual_promotion.rs).
- Independent agreement: [approved-reference registry](../crates/roomeq-model/src/reference_registry.rs)
  and [readiness/control gate](../crates/roomeq-quality/src/promotion.rs).

## Programme holdout safeguard

The programme split now refuses both a shared programme identity and an exact
rendered-stimulus hash shared with tuning. A renamed tuning stimulus reproduced
the leak before the fix and is refused afterward. The full quality library passed
238 tests with one existing ignored test; scoped production Clippy, formatting
and diff checks passed. The recorded source/dependency inventories match across
these checks.

This validates declared identity disjointness. Different rendered hashes can
still come from related source material, excerpts or transformations. Study
curation must retain source-programme lineage and disjoint rooms/programmes;
this safeguard does not certify those relationships or replace raw trial data.

## Evidence needed for promotion

Record the exact implementation and vector identities, model edition, auditory
scale, units, calibration and presentation transfer, supported level/band/time
domain, predeclared tolerances and observed agreement. Keep physical output,
headroom, seat and routing constraints authoritative during experimental ranking.

Listening claims additionally need preregistered concealed comparisons with
repeatable absolute level, a stated matching method, randomized conditions,
held-out rooms/programmes, retained raw trials, listener/stimulus variation and
uncertainty. Detection, attribute ratings, preference and equivalence require
their respective study criteria. Equal total loudness alone does not establish
any timbral equivalence or preference outcome.
