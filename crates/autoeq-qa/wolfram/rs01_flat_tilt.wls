(* autoeq-qa case autoeq-qa.rs01-flat-tilt.v1 (catalogue family RS01).
   Independent magnitude/phase plants: a flat 0 dB curve on a log-spaced
   grid, a Harman-style tilt of -0.8 dB/octave about 200 Hz, and a
   speaker highpass rolloff (0 dB at/above 80 Hz, -12 dB/octave below).
   Degenerate grid parameters must yield an empty curve, never NaNs. *)

Off[General::munfl];
n = 8;
fmin = 20.;
fmax = 20000.;
grid = Table[N[10^(Log10[fmin] + (Log10[fmax] - Log10[fmin])*i/(n - 1)), 16], {i, 0, n - 1}];
flat = Table[N[0., 16], {i, 1, n}];
tilt = Table[N[-0.8*Log[grid[[i]]/200.]/Log[2.], 16], {i, 1, n}];
xo = 80.;
slope = -12.;
rolloff = Table[
  If[grid[[i]] >= xo, N[0., 16], N[slope*Log[xo/grid[[i]]]/Log[2.], 16]],
  {i, 1, n}];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rs01-flat-tilt.v1",
  "family" -> "RS01",
  "wolfram_engine" -> "15.0.0",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> grid,
  "units" -> "dB SPL, Hz",
  "calibration" -> "0 dB reference plants",
  "flat_db" -> flat,
  "tilt_db_per_oct" -> -0.8,
  "tilt_ref_hz" -> 200.,
  "tilt_db" -> tilt,
  "crossover_hz" -> xo,
  "slope_db_per_oct" -> slope,
  "rolloff_db" -> rolloff,
  "degenerate_is_empty" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
