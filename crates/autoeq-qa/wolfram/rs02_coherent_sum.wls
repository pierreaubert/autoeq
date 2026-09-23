(* autoeq-qa case autoeq-qa.rs02-coherent-sum.v1 (catalogue family RS02).
   Independent coherent pressure sums: magnitudes in dB re unit pressure,
   B-phase relative to A. Equal in-phase sources gain 20*Log10[2];
   opposite polarity cancels (flagged, never a finite dB score); a 90-degree
   offset gains 20*Log10[Sqrt[2]]; a shared bass route sums or cancels by
   polarity. Non-finite inputs must be refused. *)

Off[General::munfl];
sumDb[aDb_, bDb_, deg_] := Module[{aa = 10^(aDb/20), bb = 10^(bDb/20),
   phi = deg*Pi/180, re, im, p},
  re = aa + bb*Cos[phi];
  im = bb*Sin[phi];
  p = Sqrt[re^2 + im^2];
  If[p <= 10^-12, Null, 20*Log10[p]]];
sumP[aDb_, bDb_, deg_] := Module[{aa = 10^(aDb/20), bb = 10^(bDb/20),
   phi = deg*Pi/180},
  Sqrt[(aa + bb*Cos[phi])^2 + (bb*Sin[phi])^2]];

cases = {{0., 0., 0.}, {0., 0., 180.}, {0., 0., 90.}, {-6., 0., 0.}, {0., 0., 60.}};
dbs = N[sumDb @@@ cases, 16];
pressures = N[sumP @@@ cases, 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rs02-coherent-sum.v1",
  "family" -> "RS02",
  "wolfram_engine" -> "15.0.0",
  "units" -> "dB re unit pressure, degrees",
  "calibration" -> "0 dB equals unit pressure",
  "cases_magA_magB_phaseDeg" -> cases,
  "sum_db" -> dbs,
  "sum_pressure" -> pressures,
  "inphase_combined_db" -> N[20*Log10[2.], 16],
  "opposite_cancels" -> True,
  "nonfinite_is_error" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
