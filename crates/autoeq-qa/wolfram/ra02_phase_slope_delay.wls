(* autoeq-qa case autoeq-qa.ra02-phase-slope-delay.v1 (catalogue family RA02).
   Independent phase-slope delay oracle: plants a pure-delay phase law
   phi(f) = -360*f*tau (unwrapped degrees, so no unwrap policy is under
   test) with a fractional-sample delay tau = 2.38125 ms at 48 kHz
   (= 114.3 samples), then fits the documented trapezoidal-weighted
   least-squares slope of phase (radians) vs frequency: delay = -slope /
   (2 Pi), seconds converted to ms. Catches delay-sign flips,
   seconds/ms slips, and sample/ms conversion errors. *)

Off[General::munfl];
sr = 48000.;
tauMs = 2.38125;
minF = 200.;
maxF = 2000.;
freqs = Table[N[minF + 10.*k, 16], {k, 0, 180}];
phase = N[-360.*freqs*(tauMs/1000.), 16];

n = Length[freqs];
w = Table[
  If[i == 1 || i == n, (freqs[[2]] - freqs[[1]])/2., (freqs[[i + 1]] - freqs[[i - 1]])/2.],
  {i, 1, n}];
rad = N[phase*Pi/180., 16];
sw = Total[w];
sf = Total[w*freqs];
sp = Total[w*rad];
sf2 = Total[w*freqs^2];
sfp = Total[w*freqs*rad];
den = sw*sf2 - sf^2;
slope = (sw*sfp - sf*sp)/den;
delayMs = N[-slope/(2 Pi)*1000., 16];
delaySamples = N[tauMs/1000.*sr, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra02-phase-slope-delay.v1",
  "family" -> "RA02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> N[freqs, 16],
  "units" -> "degrees; ms; samples",
  "calibration" -> "none (synthetic pure-delay phase)",
  "timing_origin" -> "zero-phase reference plane",
  "freqs_hz" -> N[freqs, 16],
  "phase_deg" -> N[phase, 16],
  "band_hz" -> {minF, maxF},
  "planted_delay_ms" -> tauMs,
  "fitted_delay_ms" -> delayMs,
  "planted_delay_samples" -> delaySamples
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
