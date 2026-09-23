(* autoeq-qa case autoeq-qa.h02-scorer-sign.v1 (catalogue family H02).
   Independent RBJ cookbook peaking coefficients and direct SOS transfer
   for the Python QA-helper scorer path (scripts/src/dsp.py
   biquad_coefficients, Rust-canonical order a1,a2,b0,b1,b2): catches a
   coefficient-sign transcription slip in the helper, not the cookbook. *)

Off[General::munfl];
sr = 48000.; fc = 1000.; q = 1.; gainDb = 6.;
freqs = {100., 1000., 5000.};
a = N[10^(gainDb/40), 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0]; cw = Cos[w0]; alpha = sn/(2 q);
b0 = 1 + alpha*a; b1 = -2*cw; b2 = 1 - alpha*a;
a0 = 1 + alpha/a; a1 = -2*cw; a2 = 1 - alpha/a;
coeffs = N[{a1/a0, a2/a0, b0/a0, b1/a0, b2/a0}, 16];
resp[f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  ((b0/a0) + (b1/a0) z + (b2/a0) z^2)/(1 + (a1/a0) z + (a2/a0) z^2)];
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.h02-scorer-sign.v1",
  "family" -> "H02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> freqs,
  "units" -> "normalized RBJ coefficients; complex transfer (re, im)",
  "calibration" -> "peak fc=1000Hz Q=1 +6dB; direct SOS evaluation",
  "center_hz" -> fc,
  "q" -> q,
  "gain_db" -> gainDb,
  "coeffs_a1_a2_b0_b1_b2" -> coeffs,
  "response_re_im" -> (pair[resp[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
