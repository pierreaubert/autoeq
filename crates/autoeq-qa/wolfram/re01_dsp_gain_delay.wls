(* autoeq-qa case autoeq-qa.re01-dsp-gain-delay.v1 (catalogue family RE01).
   Independent evaluation of the canonical RoomEQ replay for a serial
   gain + delay + single-peak-EQ chain: H(f) = 10^(gainDb/20) *
   Exp[-I 2 Pi f delayMs/1000] * Hpeak(f), with Hpeak from the published
   RBJ cookbook peaking equations evaluated directly as
   (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2), z^-1 = Exp[-I 2 Pi f/sr].
   The oracle never calls the Rust replay (RealizedDsp); it re-derives
   the transfer from the serialized plugin parameters. Catches
   gain/dB-factor, polarity, delay-sign, and SOS-transcription slips. *)

Off[General::munfl];
sr = 48000.;
gainDb = 6.;
delayMs = 1.;
fc = 1000.;
q = 1.;
peakDb = 6.;
freqs = {20., 100., 500., 1000., 2000., 5000., 10000., 20000.};

(* RBJ peaking section, independent of the Rust Biquad constructor. *)
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

gainLin = N[10^(gainDb/20), 16];
response[f_] := gainLin*Exp[-I N[2 Pi f delayMs/1000, 16]]*peak[f];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re01-dsp-gain-delay.v1",
  "family" -> "RE01",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 8-point log-spanning grid, Hz",
  "units" -> "linear magnitude, radians",
  "calibration" -> "relative digital gain, unity = 0 dBFS",
  "gain_db" -> gainDb,
  "delay_ms" -> delayMs,
  "peak" -> <|"filter_type" -> "peak", "freq_hz" -> fc, "q" -> q, "db_gain" -> peakDb|>,
  "design" -> "RBJ-cookbook-peak with serial gain and delay",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
