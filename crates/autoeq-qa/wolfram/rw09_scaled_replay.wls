(* autoeq-qa case autoeq-qa.rw09-scaled-replay.v1 (catalogue family RW09).
   Independent replay of a post-finalization candidate after gain
   scaling/attenuation: H(f) = 10^(gainDb/20) *
   Exp[-I 2 Pi f delayMs/1000] * Hpeak1(f) * Hpeak2(f), with each peak
   from the published RBJ cookbook peaking equations evaluated directly
   as (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2),
   z^-1 = Exp[-I 2 Pi f/sr]. The attenuated (-3 dB) two-peak chain
   stands in for the delivered transfer measured after finalization
   (late trim + repeated EQ sections), never for pre-finalization
   scores. The oracle never calls the Rust replay (RealizedDsp).
   Catches dB-factor, delay-sign, and second-section slips. *)

Off[General::munfl];
sr = 48000.;
gainDb = -3.;
delayMs = 0.5;
fc1 = 2000.; q1 = 0.7; peakDb1 = 4.5;
fc2 = 150.; q2 = 1.2; peakDb2 = -2.5;
freqs = {20., 100., 500., 1000., 2000., 5000., 10000., 20000.};

peakCoeffs[fc_, q_, pdb_] := Module[{a, w0, sn, cw, al},
  a = N[10^(pdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 q);
  {1 + al*a, -2*cw, 1 - al*a, 1 + al/a, -2*cw, 1 - al/a}
];
{b01, b11, b21, a01, a11, a21} = peakCoeffs[fc1, q1, peakDb1];
{b02, b12, b22, a02, a12, a22} = peakCoeffs[fc2, q2, peakDb2];

section[f_, b0_, b1_, b2_, a0_, a1_, a2_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];

gainLin = N[10^(gainDb/20), 16];
response[f_] := gainLin*Exp[-I N[2 Pi f delayMs/1000, 16]] *
  section[f, b01, b11, b21, a01, a11, a21] *
  section[f, b02, b12, b22, a02, a12, a22];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw09-scaled-replay.v1",
  "family" -> "RW09",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 8-point log-spanning grid, Hz",
  "units" -> "linear magnitude, radians",
  "calibration" -> "relative digital gain after -3 dB final trim, unity = 0 dBFS",
  "gain_db" -> gainDb,
  "delay_ms" -> delayMs,
  "peak1" -> <|"filter_type" -> "peak", "freq_hz" -> fc1, "q" -> q1, "db_gain" -> peakDb1|>,
  "peak2" -> <|"filter_type" -> "peak", "freq_hz" -> fc2, "q" -> q2, "db_gain" -> peakDb2|>,
  "design" -> "RBJ-cookbook peaks with serial attenuation and delay",
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
