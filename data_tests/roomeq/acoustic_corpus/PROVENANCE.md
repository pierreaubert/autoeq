# RoomEQ acoustic corpus provenance

The corpus is stored in the repository so every quality decision can be
reproduced offline. A scenario must use an opaque identifier and must not add a
person's name, postal address, room photograph, device serial number, or
embedded location metadata.

## Evidence scope

The related FEM `small_stereo_2_2_group` declaration now uses explicit main/sub
driver IDs in low-to-high crossover order. Its original five-seat captures,
crossover frequency/type, optimizer controls, and acceptance limits are retained.
The minimax, weighted-sum, and variance-penalized invocations pass after this
declaration repair; it introduces no new measured or listening evidence.

The related FEM `small_stereo_2_2_cardioid` declaration was repaired on
2026-09-20 to name both physical outputs and all existing `sub_top_lp0..4.csv`
and `sub_bottom_lp0..4.csv` captures. No measurement samples, phase values,
acceptance limits, corpus membership, or rights classification changed.
Cardioid processing now combines synchronous driver pairs per seat before
spatial aggregation. This is software/fixture evidence, not listening evidence.

This is a measurement/engineering QA corpus, not a calibrated programme-audio
listening study. Its rights and held-out-seat records do not establish listener
population, playback SPL calibration, perceptual thresholds, or pruning
equivalence. Programme selection, playback-level range, and listening validation
remain explicit deferrals in the
[Stage 0 audibility acceptance contract](../../../docs/ROOMEQ_MANUAL.md#audibility-acceptance-contract-2026-09-20).
No fixture, corpus membership, rights classification, baseline, or enforcement
limit changes as part of that documentation update.

## Independent perceptual reference vectors

No independent perceptual reference vectors are registered in this corpus.
Before a G7 submission can be considered, record each vector's source,
set identity, model edition, license identifier, local artifact path, and
SHA-256 digest.
The QA intake verifies the artifact bytes against that digest and rejects
missing edition/license metadata or a vector identical to a generated
control. Passing intake only establishes traceability; model agreement still
needs approved numeric reference evidence, and real listening trials remain
a separate gate.

## Sources and redistribution

The 2026-09-20 pruning implementation adds synthetic software regression rows
for declared spectra/levels, correlated playback, native export, and exact
rollback. The synthetic QA builder now emits schema-v3 physical-output and
crossover declarations. These changes do not add measured corpus material,
recalibrate acceptance limits, or establish listening-study evidence.

| Scenario family | Source | Rights classification | Privacy review |
|---|---|---|---|
| fem_* | Deterministic finite-element fixtures under ../generate/fem/ | Generated project test data; covered by the repository license | Contains numeric curves/configuration only; no personal data |
| measured_stereo_8361a | Contributor-supplied stereo room capture under ../measured/2.0_8361a/ | LicenseRef-SOTF-Project-Test-Data | Opaque room ID; CSV/WAV and channel names only |
| measured_stereo_d3v | Contributor-supplied stereo room capture under ../measured/2.0_d3v/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; CSV/WAV and channel names only |
| measured_stereo_t7v | Contributor-supplied stereo room capture under ../measured/2.0_t7v_2024/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; CSV/WAV and channel names only |
| measured_stereo_fidelia | Public third-party stereo room capture under ../measured/2.0_fidelia/ (see README.md) | LicenseRef-SOTF-Project-Test-Data | Opaque scenario ID; CSV and channel names only |
| measured_stereo_ascilab1 | Contributor-supplied stereo room capture under ../measured/2.0_ascilab1/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; REW TXT and channel names only |
| measured_stereo_ascilab2 | Contributor-supplied stereo room capture under ../measured/2.0_ascilab2/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; REW TXT and channel names only |
| measured_stereo_ascilab3 | Contributor-supplied stereo room capture under ../measured/2.0_ascilab3/ | LicenseRef-SOTF-Project-Test-Data | Opaque device-family ID; REW TXT and channel names only |

FEM held-out entries use `synthetic-heldout-1` and
`synthetic-heldout-2` seat IDs. The generator applies the same indexed
interpolation pattern to each channel at a given synthetic position, so
coherent replay can join channels by an explicit generated identity. These IDs
are virtual positions, not surveyed physical seats or real playback captures.
The same generator now emits held-out curves for every declared FEM physical
sub output; coherent replay requires the sub response at each synthetic seat
and never broadcasts a single sub curve across seats.

LicenseRef-SOTF-Project-Test-Data means the files are retained and exercised
as part of this repository's test suite. It is not a grant to extract and
redistribute the recordings as a separate dataset. A maintainer must confirm
the contributor's rights before adding another real capture.

## Intake checklist

Before committing a real measurement:

1. Confirm the contributor created the recording or has permission to share it.
2. Strip names, addresses, geolocation, free-form notes, and device serials.
3. Use opaque scenario and directory identifiers.
4. Record the source family, rights classification, and privacy result here.
5. Add at least one held-out position where the capture contains multiple seats.
   A single-position capture may join as `report_only` (never `enforce`); the
   manifest validator rejects enforced scenarios without two held-out
   measurements covering every scored channel. `measured_stereo_fidelia` is
   the current example: full-range measured timbre with level/dropout
   robustness, awaiting a second seat before it can enforce.
6. Run the PR corpus twice and confirm byte-identical JSON output.

The unused ../measured/5_1_kef/*.mdat capture is not in the acoustic corpus.
The privacy-safe converter can recover its seven channel labels and numeric
SPL/phase curves without exporting embedded note bodies, but contributor rights
remain unverified and the file contains only one listening position. It cannot
serve as held-out or multi-seat evidence until a maintainer completes the
rights review and obtains additional positions.
