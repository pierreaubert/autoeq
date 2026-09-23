(* autoeq-qa case autoeq-qa.ra01-log-slope.v1 (catalogue family RA01).
   Independent least-squares slope of SPL vs log2(frequency): the oracle
   plants SPL = refSpl + trueSlope*Log2(f/refF) on a 60-point log grid
   plus two out-of-window spikes, then fits the ordinary least-squares
   line (covariance/variance closed form) inside the declared windows.
   Catches log-base mixups, window-selection slips and sign errors. *)

Off[General::munfl];
ns = 60;
fLo = 200.;
fHi = 10000.;
refF = 200.;
refSpl = 80.;
slopeTrue = -1.5;
grid = Table[N[fLo*((fHi/fLo)^((i - 1)/(ns - 1))), 16], {i, 1, ns}];
splGrid = N[refSpl + slopeTrue*Log[2, grid/refF], 16];
freqs = Join[{150.}, grid, {15000.}];
spl = Join[{120.}, splGrid, {40.}];

olsSlope[lo_, hi_] := Module[{sel, x, y, n, sx, sy, sxx, sxy, den},
  sel = Select[Transpose[{freqs, spl}], (#[[1]] >= lo && #[[1]] <= hi) &];
  x = N[Log[2, #[[1]]] & /@ sel, 16];
  y = #[[2]] & /@ sel;
  n = Length[sel];
  sx = Total[x]; sy = Total[y];
  sxx = Total[x^2]; sxy = Total[x*y];
  den = n*sxx - sx^2;
  {n, N[(n*sxy - sx*sy)/den, 16]}
];

full = olsSlope[200., 10000.];
sub = olsSlope[500., 4000.];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra01-log-slope.v1",
  "family" -> "RA01",
  "sample_rate_hz" -> "not-applicable",
  "grid_hz" -> N[freqs, 16],
  "units" -> "dB-per-octave (SPL vs log2 Hz)",
  "calibration" -> "none (synthetic SPL, 80 dB reference)",
  "freqs_hz" -> N[freqs, 16],
  "spl_db" -> N[spl, 16],
  "true_slope_db_per_oct" -> slopeTrue,
  "full_window_hz" -> {200., 10000.},
  "full_n" -> full[[1]],
  "full_slope_db_per_oct" -> full[[2]],
  "sub_window_hz" -> {500., 4000.},
  "sub_n" -> sub[[1]],
  "sub_slope_db_per_oct" -> sub[[2]]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
