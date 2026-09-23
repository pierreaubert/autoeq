(* autoeq-qa case autoeq-qa.re09-kautz-bank.v1 (catalogue family RE09).
   Fixed-pole Kautz basis derivation from the published second-order
   definitions (pole radius from Q, allpass branches, chained basis
   functions), dry-plus-bank composite 1 + g1*B1 + g2*B2, and a tiny
   complex least-squares dry-plus-bank fit against a prescribed correction
   target solved through the normal equations. The oracle prescribes the
   fitted gains; the Rust side realizes the delivered plugin chain. *)

Off[General::munfl];
sr = 48000.;
freqs = {20., 40., 55., 70., 120., 200., 500., 1000.};
poles = {55., 120.};
qs = {8., 10.};
gPrescribed = {0.35, -0.25};

radius[f0_, q_] := N[Exp[-Pi*f0/(Max[q, 0.1]*sr)], 16];
theta[f0_] := N[2 Pi*f0/sr, 16];
a1c[f0_, q_] := N[-2*radius[f0, q]*Cos[theta[f0]], 16];
a2c[f0_, q_] := N[radius[f0, q]^2, 16];
allpass[f0_, q_, f_] := Module[{w = N[2 Pi*f/sr, 16], z, a1 = a1c[f0, q], a2 = a2c[f0, q]},
  z = Exp[-I w]; (a2 + a1*z + z^2)/(1 + a1*z + a2*z^2)];
basis[f0_, q_, chain_, f_] := Module[{w = N[2 Pi*f/sr, 16], z, a1 = a1c[f0, q], a2 = a2c[f0, q], r = radius[f0, q]},
  z = Exp[-I w]; (N[Sqrt[1 - r^2]*(1 - a2), 16]/(1 + a1*z + a2*z^2))*chain];
basisPair[f_] := Module[{b1 = basis[poles[[1]], qs[[1]], 1., f], a1 = allpass[poles[[1]], qs[[1]], f]},
  {b1, basis[poles[[2]], qs[[2]], a1, f]}];
bank[bps_, gains_] := gains[[1]]*bps[[1]] + gains[[2]]*bps[[2]];

complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};
basisGrid = basisPair /@ freqs;
prescribed = N[Table[1. + bank[basisGrid[[i]], gPrescribed], {i, Length[freqs]}], 16];

corrTarget = N[0.3*Exp[-((Log[2, #/55.])/0.25)^2] - 0.2*Exp[-((Log[2, #/120.])/0.25)^2] & /@ freqs, 16];
b1v = basisGrid[[All, 1]];
b2v = basisGrid[[All, 2]];
m11 = Total[Conjugate[b1v]*b1v]; m12 = Total[Conjugate[b1v]*b2v];
m21 = Total[Conjugate[b2v]*b1v]; m22 = Total[Conjugate[b2v]*b2v];
v1 = Total[Conjugate[b1v]*corrTarget]; v2 = Total[Conjugate[b2v]*corrTarget];
gFit = N[LinearSolve[N[Re[{{m11, m12}, {m21, m22}}], 16], N[Re[{v1, v2}], 16]], 16];
fitted = N[Table[1. + bank[basisGrid[[i]], gFit], {i, Length[freqs]}], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re09-kautz-bank.v1",
  "family" -> "RE09",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "units" -> "linear transfer (dimensionless); section gains are linear basis weights",
  "calibration" -> "relative transfer only; no SPL calibration",
  "poles_hz" -> poles,
  "q" -> qs,
  "pole_radius" -> N[radius @@@ Transpose[{poles, qs}], 16],
  "prescribed_gains" -> gPrescribed,
  "basis1_re_im" -> (complexPair /@ b1v),
  "basis2_re_im" -> (complexPair /@ b2v),
  "prescribed_re_im" -> (complexPair /@ prescribed),
  "correction_target" -> corrTarget,
  "fitted_gains" -> N[gFit, 16],
  "fitted_re_im" -> (complexPair /@ fitted)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
