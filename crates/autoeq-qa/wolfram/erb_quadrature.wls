(* autoeq-qa case autoeq-qa.erb-quadrature.v1 (catalogue family C06).
   Glasberg-Moore ERB-rate numbers rate(f) = 21.4*Log10[1+0.00437 f],
   trapezoidal cell widths on the ERB-rate axis (half cells at the ends,
   centered differences inside), the weighted RMS Sqrt[Sum v^2 w/Sum w],
   and the timing-to-phase conversion phase = 360 f dt in degrees.
   Compared against autoeq_core::auditory_frequency and
   autoeq_core::alignment::timing_uncertainty_to_phase_deg. *)

Off[General::munfl];
grid = {20., 100., 500., 1000., 5000., 10000., 20000.};
vals = {1., -2., 0.5, 3., -1.5, 2., 0.};

rate[f_] := N[21.4*Log10[1 + 0.00437*f], 16];
rates = rate /@ grid;
n = Length[grid];
widths = Table[
  Which[
    i == 1, (rates[[2]] - rates[[1]])/2,
    i == n, (rates[[n]] - rates[[n - 1]])/2,
    True, (rates[[i + 1]] - rates[[i - 1]])/2
  ], {i, 1, n}];
rms = N[Sqrt[Total[vals^2*widths]/Total[widths]], 16];

timingCases = {
  <|"freq_hz" -> 1000., "dt_s" -> 1.*^-5, "phase_deg" -> N[360.*1000.*1.*^-5, 16]|>,
  <|"freq_hz" -> 20000., "dt_s" -> 2.5*^-6, "phase_deg" -> N[360.*20000.*2.5*^-6, 16]|>
};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.erb-quadrature.v1",
  "family" -> "C06",
  "measure" -> "glasberg-moore-erb-rate-1990-v1",
  "grid_hz" -> grid,
  "values" -> vals,
  "erb_rates" -> rates,
  "cell_widths" -> widths,
  "weighted_rms" -> rms,
  "timing_cases" -> timingCases
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
