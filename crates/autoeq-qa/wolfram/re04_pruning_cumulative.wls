(* autoeq-qa case autoeq-qa.re04-pruning-cumulative.v1 (catalogue family RE04).
   Frozen full-chain (F0) reference for a two-biquad chain plus every
   cumulative removal subset, evaluated INDEPENDENTLY from the RBJ cookbook
   peak equations and direct SOS transfer products. With/without-filter dB
   differences and ERB-rate-weighted RMS differences replicate the audit's
   per-condition impact measure. Catches coefficient, conjugation, dB-factor,
   and subset-enumeration slips. *)

Off[General::munfl];
sr = 48000.;
freqs = {50., 100., 200., 500., 1000., 2000., 5000., 10000.};

peakCoeffs[fc_, q_, gainDb_] := Module[{a, w0, sn, cw, alpha},
  a = N[10^(gainDb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; alpha = sn/(2 q);
  N[{1 + alpha*a, -2*cw, 1 - alpha*a, 1 + alpha/a, -2*cw, 1 - alpha/a}, 16]
];
sosResp[coeffs_, f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (coeffs[[1]] + coeffs[[2]]*z + coeffs[[3]]*z^2)/(coeffs[[4]] + coeffs[[5]]*z + coeffs[[6]]*z^2)
];
cA = peakCoeffs[1000., 1.0, 6.];
cB = peakCoeffs[100., 2.0, -4.];
hA[f_] := sosResp[cA, f];
hB[f_] := sosResp[cB, f];
hF0[f_] := hA[f]*hB[f];

complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};
magDb[v_] := N[20*Log10[Abs[v]], 16];
f0 = hF0 /@ freqs;
onlyB = hB /@ freqs;
onlyA = hA /@ freqs;
dry = Table[N[1. + 0. I, 16], {Length[freqs]}];
magF0 = magDb /@ f0;
diffRemoveA = N[magF0 - (magDb /@ onlyB), 16];
diffRemoveB = N[magF0 - (magDb /@ onlyA), 16];

erbRate[f_] := N[21.4*Log10[1. + 0.00437*f], 16];
rates = erbRate /@ freqs;
n = Length[freqs];
erbW = N[Table[If[i == 1, 0.5*(rates[[2]] - rates[[1]]), If[i == n, 0.5*(rates[[n]] - rates[[n - 1]]), 0.5*(rates[[i + 1]] - rates[[i - 1]])]], {i, n}], 16];
erbRms[d_] := N[Sqrt[Total[d^2*erbW]/Total[erbW]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re04-pruning-cumulative.v1",
  "family" -> "RE04",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "units" -> "dB",
  "calibration" -> "relative filter transfer only; no SPL calibration",
  "design" -> "RBJ-cookbook-peak x2 cascade",
  "filter_a" -> <|"center_hz" -> 1000., "q" -> 1.0, "gain_db" -> 6.|>,
  "filter_b" -> <|"center_hz" -> 100., "q" -> 2.0, "gain_db" -> -4.|>,
  "f0_re_im" -> (complexPair /@ f0),
  "minus_a_re_im" -> (complexPair /@ onlyB),
  "minus_b_re_im" -> (complexPair /@ onlyA),
  "f0_mag_db" -> magF0,
  "diff_remove_a_db" -> diffRemoveA,
  "diff_remove_b_db" -> diffRemoveB,
  "erb_rms_remove_a_db" -> erbRms[diffRemoveA],
  "erb_rms_remove_b_db" -> erbRms[diffRemoveB]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
