(* autoeq-qa case autoeq-qa.re21-eq-response-correction.v1 (catalogue family RE21).
   Independent branch/matrix check of the assembled graph correction:
   the reported EQ correction curve is final-minus-initial in dB on the
   shared grid, with the grid passed through unchanged (aligned grids,
   never zipped across unequal grids). The oracle never calls
   compute_eq_response; it subtracts the two curves elementwise, so it
   catches sign flips, grid misalignment, and length mismatches. *)

Off[General::munfl];
freqs = {20., 50., 100., 200., 500., 1000., 2000.};
initialSpl = {80., 82., 81., 79., 78., 80., 79.5};
finalSpl = {85., 85., 84., 83., 82., 83., 82.5};

correction = MapThread[N[#2 - #1, 16] &, {initialSpl, finalSpl}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re21-eq-response-correction.v1",
  "family" -> "RE21",
  "grid" -> "explicit shared 7-point grid, Hz; both curves on the same grid",
  "units" -> "dB",
  "calibration" -> "level-relative correction curve (final minus initial)",
  "freqs_hz" -> freqs,
  "initial_spl_db" -> initialSpl,
  "final_spl_db" -> finalSpl,
  "correction_spl_db" -> correction
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
