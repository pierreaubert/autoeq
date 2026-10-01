# Measured IR import (2026-09-30)

Test cases are under `data_tests/roomeq/measured/`.

Native IRs were extracted with `utils/mdat2csv.py`'s `parse_mdat` and
`export_ir_csv`, without normalization, recentering, windowing, or resampling.
The original seven cases were matched numerically to the configured response CSV: SPL RMS
difference below 0.00001 dB on that CSV's frequency grid, with a unique match.
Exported amplitudes and time coordinates were checked against the parsed MDAT.
Existing frequency-response CSVs and generated `recordings.json` files were not
changed. New IR files are in each case's `measured_ir/` directory.

| Case | MDAT | Captures selected | Parent config attachment |
| --- | --- | --- | --- |
| 2.0_t7v_2026 | 2.0_t7v_20260912.mdat | L Sep 12; R Sep 12 | Existing L/R declarations and files verified, unchanged |
| 2.0_ascilab1 | ascila1.mdat | L/R C8C-BX8C-P1 | Native 48 kHz L/R declarations added |
| 2.0_ascilab2 | ascilab2.mdat | L/R C8C-BX8C-P2 | Native 48 kHz L/R declarations added |
| 2.0_ascilab3 | ascilab3.mdat | L/R C8C-BX8C-P3 | Native 48 kHz L/R declarations added |
| 2.0_ascilab_mmm | ascilab_mmm.mdat | L/R magnitude-only RTA curves, no stored IRs | Local L/R config and IIR/FIR/hybrid presets added; no IR declarations |
| 2.2_genelec | originals/MLP_NoGLM-No_XO.mdat | L/R MAIN MLP; L/R SUB MLP | L/R and explicit Sub1/Sub2 driver declarations added |
| 2.2_sigberg1 | pierre-new-1-dualsubs.mdat | Left/Right speaker; left/right sub | All four added; sub timing IDs preserved, unknown main timing omitted |
| 2.2_sigberg2 | pierre-new-1-dualsubs.mdat | Left/Right speaker; left/right sub **better placement** | All four added; unknown timing omitted |
| 2.2_sigberg3 | pierre-new-2-dualsubs.mdat | Left SBS. P2 → left_main; Right SBS P2 → right_main; Sub Gulv → left_sub; Sub Vegg → right_sub | All four added with exact driver targets; unknown timing omitted |
| 2.2_unknown | 2.2.mdat | L/R No EQ Sep 1; L/R Sub No EQ Sep 1 | L/R and native 3 kHz Sub1/Sub2 driver declarations added |
| 5.1_kef | 260706a_1800_cdsp_straightthru_baseline.mdat | All seven takes | Six physical outputs added (L/R/C/BL/BR/Sub1); measured L+R retained separately, not a playback output |

Genelec's `originals/5_pos_sweep_subs_only_290626.mdat` contains different
captures/positions; none matches a response used by the parent config, so it
was not substituted for the MLP July 2 captures.

The measured-IR validator accepts native 1–192 kHz captures. The sub IRs in the
Sigberg and unknown MDATs remain natively 3 kHz, without resampling. Complete
octave bands above Nyquist and the 1–8 kHz reflection analysis are unavailable.
Per-driver diagnostics and native waveforms are saved in acoustic sidecars and
displayed on the corresponding report tab, not on the parent sum.

The Ascilab position captures were matched uniquely by measurement-name prefix
and exact REW capture date/time. Their existing response exports use variable
smoothing, so raw MDAT SPL arrays were not treated as identical to those exports.
All six IRs contain 131072 samples at 48 kHz; every exported amplitude and time
coordinate was checked for exact round-trip equality. No timing-reference ID
was inferred from the acoustic-reference label. The MMM takes have null `ir`
and `irData` fields, so there is no measured room IR to extract.

There are 34 attached physical captures across ten cases. A filename or shared
MDAT container does not establish a shared clock, so missing timing IDs were not
fabricated. Independent acoustic diagnostics allow unknown timing; these IR
declarations are not inputs to coherent summation or channel-alignment analysis.
Those operations still require their own shared-clock provenance gates.

Config keys for routed KEF outputs follow the delivered graph (`L`, `R`, `C`,
`BL`, `BR`, `Sub1`), not the long measurement names. Existing timing IDs were
copied from the corresponding measurement provenance.
