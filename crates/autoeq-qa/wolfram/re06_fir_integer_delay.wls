(* autoeq-qa case autoeq-qa.re06-fir-integer-delay.v1 (catalogue family RE06).
   Independent integer FIR delay realization: the input taps are padded
   with exactly `shiftSamples` leading zeros (no wraparound, no
   resampling), and the delayed transfer is the direct DTFT sum
   H(f) = Sum_n h[n] Exp[-I 2 Pi f n/sr] over the padded taps.
   The oracle never calls realize_gd_fir_delay; it constructs the
   zero-padded vector and the DTFT from the delay definition, so it
   catches double-delay application, truncation, and off-by-one shifts. *)

Off[General::munfl];
sr = 48000.;
taps = {1., 0.5, -0.25};
delayMs = 1.;
shiftSamples = 48;
freqs = {0., 100., 1000., 5000., 10000., 20000.};

padded = Join[ConstantArray[0., shiftSamples], taps];

response[f_] := Sum[padded[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, Length[padded] - 1}];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re06-fir-integer-delay.v1",
  "family" -> "RE06",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 6-point grid including DC, Hz",
  "units" -> "linear magnitude, radians; taps dimensionless",
  "calibration" -> "tap zero at time zero; delay adds causal leading zeros",
  "input_taps" -> taps,
  "delay_ms" -> delayMs,
  "shift_samples" -> shiftSamples,
  "stored_taps" -> padded,
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
