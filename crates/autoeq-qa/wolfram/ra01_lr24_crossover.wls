(* autoeq-qa case autoeq-qa.ra01-lr24-crossover.v1 (catalogue family RA01).
   Independent LR24 crossover branches from the published RBJ cookbook:
   two cascaded 2nd-order Butterworth sections (Q = 1/Sqrt[2]) for the
   lowpass and highpass, evaluated as squared SOS transfers
   (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2) with z^-1 = Exp[-I w].
   The oracle keeps the raw a0 form (ratio is normalization-invariant).
   Also emits the crossover-sum magnitude |LP+HP|, which is unity by the
   Linkwitz-Riley complementarity identity. Catches coefficient slips,
   missing cascade squaring, and conjugation errors. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
q = N[1/Sqrt[2], 16];
freqs = {20., 100., 250., 500., 1000., 2000., 4000., 8000., 16000., 20000.};

w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0];
cw = Cos[w0];
alpha = sn/(2 q);
a0 = 1 + alpha;
a1 = -2*cw;
a2 = 1 - alpha;

lpB0 = (1 - cw)/2; lpB1 = 1 - cw; lpB2 = (1 - cw)/2;
hpB0 = (1 + cw)/2; hpB1 = -(1 + cw); hpB2 = (1 + cw)/2;

branch[b0_, b1_, b2_, f_] := Module[{w = N[2 Pi f/sr, 16], z, h},
  z = Exp[-I w];
  h = (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2);
  h^2
];

lpVals = branch[lpB0, lpB1, lpB2, #] & /@ freqs;
hpVals = branch[hpB0, hpB1, hpB2, #] & /@ freqs;
sumMag = N[Abs[#] & /@ (lpVals + hpVals), 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra01-lr24-crossover.v1",
  "family" -> "RA01",
  "sample_rate_hz" -> sr,
  "grid_hz" -> freqs,
  "units" -> "complex transfer (dimensionless); sum magnitude dimensionless",
  "calibration" -> "none (unity-gain crossover)",
  "crossover_hz" -> fc,
  "q" -> q,
  "design" -> "RBJ-cookbook-Butterworth-cascaded-x2 (LR24)",
  "freqs_hz" -> freqs,
  "lp_re_im" -> (complexPair[#] & /@ lpVals),
  "hp_re_im" -> (complexPair[#] & /@ hpVals),
  "sum_magnitude" -> sumMag
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
