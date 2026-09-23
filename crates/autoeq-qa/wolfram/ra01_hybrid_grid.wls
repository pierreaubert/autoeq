(* autoeq-qa case autoeq-qa.ra01-hybrid-grid.v1 (catalogue family RA01).
   Independent construction of the canonical RoomEQ hybrid optimizer grid
   from the declared constants: linear 4 Hz bass spacing from 20 Hz to
   1000 Hz, then base-10 logarithmic spacing to 20000 Hz whose density
   derives from frequency_samples over the full 20..20000 span. The
   oracle re-derives the interval counts from those constants (it does
   not copy the Rust loop) and emits the full grid as the fixture of
   record. Catches junction/endpoint slips and linear-vs-log mixups. *)

Off[General::munfl];
ns = 200;
fMin = 20.;
fJunction = 1000.;
fMax = 20000.;
bassStep = 4.;
logStep = N[Log[fMax/fMin]/(ns - 1), 16];
nInt = Ceiling[Log[fMax/fJunction]/logStep];
nLow = Round[(fJunction - fMin)/bassStep] + 1;
low = Table[N[fMin + bassStep*k, 16], {k, 0, nLow - 1}];
hi = Table[N[10^(Log10[fJunction] + j*(Log10[fMax] - Log10[fJunction])/nInt), 16], {j, 1, nInt}];
freqs = Join[low, hi];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra01-hybrid-grid.v1",
  "family" -> "RA01",
  "sample_rate_hz" -> 48000.,
  "grid_hz" -> N[freqs, 16],
  "units" -> "Hz",
  "calibration" -> "none (geometric grid)",
  "frequency_samples" -> ns,
  "bass_step_hz" -> bassStep,
  "junction_hz" -> fJunction,
  "n_low" -> nLow,
  "n_high_intervals" -> nInt,
  "freqs_hz" -> N[freqs, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
