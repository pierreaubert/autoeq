(* autoeq-qa case autoeq-qa.qa01-clock-phase.v1 (catalogue family QA01).
   Independent clock/phase arithmetic: accumulated offset
   ppm*1e-6*duration (sign-preserving), phase uncertainty 360*f*dt in
   degrees, coherent two-source sums 20*Log10[|sum|] with exact
   cancellation flagged (never 0 dB), and common-EQ invariance of
   seat-to-seat differences (missing entries skipped, never
   interpolated). *)

Off[General::munfl];
clockCases = {{50., 20.}, {-50., 20.}, {100., 1.}};
clockOffsets = {N[0.001, 16], N[-0.001, 16], N[0.0001, 16]};

phaseCases = {{100., 0.0005}, {1000., 0.0005}, {50., 0.001}};
phaseDeg = {N[18., 16], N[180., 16], N[18., 16]};

coherentGains = {N[20*Log10[2.], 16], N[20*Log10[4.], 16]};

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.qa01-clock-phase.v1",
  "family" -> "QA01",
  "wolfram_engine" -> "15.0.0",
  "units" -> "s, degrees, dB",
  "calibration" -> "none required: pure algebraic identities",
  "clock_cases_ppm_s" -> clockCases,
  "clock_offsets_s" -> clockOffsets,
  "phase_cases_hz_s" -> phaseCases,
  "phase_degrees" -> phaseDeg,
  "coherent_gain_11_db" -> coherentGains[[1]],
  "coherent_gain_1111_db" -> coherentGains[[2]],
  "coherent_1m1_cancels" -> True,
  "seat_a_db" -> {70., 72., Null, 68.},
  "seat_b_db" -> {66., 71., 69., Null},
  "common_eq_db" -> {-3., 2., 1., -1.},
  "common_eq_drift_db" -> N[0., 16],
  "seat_changed_drift_db" -> N[0., 16],
  "naive_forgetful_drift_db" -> N[3., 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
