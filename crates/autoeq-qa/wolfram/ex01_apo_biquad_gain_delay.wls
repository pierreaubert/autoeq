(* autoeq-qa case autoeq-qa.ex01-apo-biquad-gain-delay.v1 (catalogue family EX01).
   Independent evaluation of one exported peak section with preamp gain
   and an integer-sample delay: H(f) = 10^(preampDb/20) *
   Exp[-I 2 Pi f delaySamples/sr] * Hpeak(f), with Hpeak from the
   published RBJ cookbook peaking equations in raw-a0 form
   (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2). The Rust test parses the
   Equalizer APO text and the normalized-biquad JSON renders
   independently and evaluates the documented a0 = 1 convention
   H = (b0+b1 z^-1+b2 z^-2)/(1+a1 z^-1+a2 z^-2); agreement checks the
   export coefficient normalization, the text/JSON parameter passes,
   the preamp dB factor, and the delay factorization. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
q = 1.;
peakDb = 6.;
preampDb = -3.;
delaySamples = 5;
delayMs = N[delaySamples*1000/sr, 16];
freqs = {20., 100., 500., 1000., 2000., 5000., 10000., 20000.};

a = N[10^(peakDb/40), 16];
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

peak[f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];
response[f_] := N[10^(preampDb/20), 16]*
  Exp[-I N[2 Pi f delaySamples/sr, 16]]*peak[f];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ex01-apo-biquad-gain-delay.v1",
  "family" -> "EX01",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 8-point log-spanning grid, Hz",
  "units" -> <|"frequency" -> "Hz", "gain" -> "dB", "delay" -> "samples and ms",
    "response" -> "complex-linear"|>,
  "calibration" -> "relative digital gain, unity = 0 dBFS; delay is exact samples",
  "peak" -> <|"filter_type" -> "peak", "freq_hz" -> fc, "q" -> q, "db_gain" -> peakDb|>,
  "preamp_gain_db" -> preampDb,
  "delay_samples" -> delaySamples,
  "delay_ms" -> delayMs,
  "rbj_raw" -> <|"b0" -> N[b0, 16], "b1" -> N[b1, 16], "b2" -> N[b2, 16],
    "a0" -> N[a0, 16], "a1" -> N[a1, 16], "a2" -> N[a2, 16]|>,
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
