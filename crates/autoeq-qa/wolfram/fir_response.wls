(* autoeq-qa case autoeq-qa.fir-response.v1 (catalogue family C01).
   Direct finite DTFT sum H(f) = Sum_n h[n] Exp[-I 2 Pi f n/sr] with tap
   zero at time zero. Compared against
   autoeq_core::response::compute_fir_complex_response at exact
   frequencies — not against the padded-FFT nearest-bin approximation
   used by some FIR test helpers. *)

Off[General::munfl];
sr = 48000.;
taps = {0.25, 0.5, 1.0, 0.5, 0.25};
freqs = {0., 100., 1000., 5000., 10000., 20000.};

response[f_] := Sum[taps[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, Length[taps] - 1}];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.fir-response.v1",
  "family" -> "C01",
  "sample_rate_hz" -> sr,
  "taps" -> taps,
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
