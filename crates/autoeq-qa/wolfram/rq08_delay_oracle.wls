(* autoeq-qa case autoeq-qa.rq08-delay-oracle.v1 (catalogue family RQ08).
   Independent oracle transfer-equation plants: a pure-delay plant
   H(f) = Exp[-I 2 Pi f tau] (positive delay is a negative phase ramp)
   on a log-frequency grid, and a polarity plant H = -1 (inverted) with
   unit magnitude everywhere. Both are analytic ground truth for generated
   candidate/input/output data, not fitted values. *)

Off[General::munfl];
n = 8;
fmin = 80.;
fmax = 16000.;
grid = Table[N[10^(Log10[fmin] + (Log10[fmax] - Log10[fmin])*i/(n - 1)), 16], {i, 0, n - 1}];
delayMs = 1.75;
tau = N[delayMs/1000., 16];

delayH = Table[N[Exp[-I*2*Pi*grid[[i]]*tau], 16], {i, 1, n}];
complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};
magOf[v_] := N[Abs[v], 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rq08-delay-oracle.v1",
  "family" -> "RQ08",
  "wolfram_engine" -> "15.0.0",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> grid,
  "units" -> "linear complex transfer, Hz, ms",
  "calibration" -> "unity reference transfer",
  "delay_ms" -> delayMs,
  "delay_re_im" -> (complexPair[#] & /@ delayH),
  "delay_mag" -> (magOf[#] & /@ delayH),
  "polarity_re_im" -> Table[{N[-1., 16], N[0., 16]}, {i, 1, n}],
  "valid_region_hz" -> {N[grid[[1]], 16], N[grid[[n]], 16]}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
