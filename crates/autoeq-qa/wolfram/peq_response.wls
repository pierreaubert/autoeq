(* autoeq-qa case autoeq-qa.peq-response.v1 (catalogue family C01).
   Independent evaluation of the RBJ cookbook peaking design equations
   (the same published formulas the Rust Biquad constructor implements)
   at the exact QA parameters, followed by direct SOS transfer
   evaluation (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2) with
   z^-1 = Exp[-I 2 Pi f/sr]. Catches coefficient-transcription and
   response-evaluation slips, not the cookbook itself. The oracle keeps
   the raw a0 form; the ratio is normalization-invariant, so it also
   checks the Rust a0 normalization. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
q = 1.;
gainDb = 6.;
freqs = {20., 100., 500., 1000., 2000., 5000., 10000., 20000.};

a = N[10^(gainDb/40), 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0];
cw = Cos[w0];
alpha = sn/(2 q);
b0 = 1 + alpha*a;
b1 = -2*cw;
b2 = 1 - alpha*a;
a0 = 1 + alpha/a;
a1 = -2*cw;
a2 = 1 - alpha/a;

response[f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.peq-response.v1",
  "family" -> "C01",
  "sample_rate_hz" -> sr,
  "center_hz" -> fc,
  "q" -> q,
  "gain_db" -> gainDb,
  "design" -> "RBJ-cookbook-peak",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
