(* autoeq-qa case autoeq-qa.root01-facade-passthrough.v1 (catalogue family ROOT01).
   Independent RBJ cookbook peaking evaluation (published design
   equations, direct SOS transfer with z^-1 = Exp[-I 2 Pi f/sr]) at a
   C01 fixture with a non-default sample rate (44100 Hz) and cut gains.
   The Rust comparison routes this fixture through the engine root
   facade (roomeq_engine::response) and separately through the direct
   kernel path; both must reproduce this oracle with numeric arguments
   unchanged (sample-rate and unit forwarding). Catches facade
   argument reordering, rate/unit mangling, and removed-alias drift. *)

Off[General::munfl];
sr = 44100.;
fc = 2000.;
q = 0.5;
gainDb = -4.;
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
  "case" -> "autoeq-qa.root01-facade-passthrough.v1",
  "family" -> "ROOT01",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 8-point log-spanning grid, Hz",
  "units" -> "linear magnitude, radians",
  "calibration" -> "relative digital gain, unity = 0 dBFS",
  "center_hz" -> fc,
  "q" -> q,
  "gain_db" -> gainDb,
  "design" -> "RBJ-cookbook-peak via root facade",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
