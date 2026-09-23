(* autoeq-qa case autoeq-qa.m02-avg-coherent.v1 (catalogue family M02).
   Independent evaluation of the three take-average kinds from direct
   complex sums: power mean 10log10(mean(10^(L/10))), arithmetic dB mean,
   and weighted complex-pressure mean (magnitude AND angle from the mean
   phasor). Fixture: two takes at equal level, 90 degrees apart, so power
   and dB means stay at the take level while the coherent mean sits
   10log10(2) below it; the coherent SUM documents the +6.0206 dB factor.
   Spatial magnitude averages must never carry an averaged angle. *)

Off[General::munfl];
freqs = {100., 1000., 10000.};
splA = {80., 84., 78.};
phA = {0., 0., 0.};
splB = {80., 84., 78.};
phB = {90., 90., 90.};

amp[s_] := N[10^(s/20), 16];
powerMean[a_, b_] := N[10*Log[10, (10^(a/10) + 10^(b/10))/2], 16];
dbMean[a_, b_] := N[(a + b)/2, 16];

coh[a_, pa_, b_, pb_] := Module[{re, im, mag},
  re = (amp[a]*Cos[N[pa*Pi/180, 16]] + amp[b]*Cos[N[pb*Pi/180, 16]])/2;
  im = (amp[a]*Sin[N[pa*Pi/180, 16]] + amp[b]*Sin[N[pb*Pi/180, 16]])/2;
  mag = Sqrt[re^2 + im^2];
  {N[20*Log[10, mag], 16], N[ArcTan[re, im]*180/Pi, 16], N[20*Log[10, 2*mag], 16]}];

n = Length[freqs];
power = Table[powerMean[splA[[i]], splB[[i]]], {i, n}];
dbm = Table[dbMean[splA[[i]], splB[[i]]], {i, n}];
triples = Table[coh[splA[[i]], phA[[i]], splB[[i]], phB[[i]]], {i, n}];
cohSpl = triples[[All, 1]];
cohPhase = triples[[All, 2]];
cohSumSpl = triples[[All, 3]];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.m02-avg-coherent.v1",
  "family" -> "M02",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> freqs,
  "units" -> "SPL dB (power mean uses 10log10, coherent uses 20log10); phase degrees",
  "calibration" -> "common timing/gain reference ref-a on both takes",
  "grid_note" -> "shared 3-point grid on every take; no resampling (never zip unequal grids)",
  "take_a_spl_db" -> splA, "take_a_phase_deg" -> phA,
  "take_b_spl_db" -> splB, "take_b_phase_deg" -> phB,
  "power_mean_db" -> power,
  "db_mean_db" -> dbm,
  "coherent_mean_db" -> cohSpl,
  "coherent_phase_deg" -> cohPhase,
  "coherent_sum_db" -> cohSumSpl,
  "coherent_sum_minus_mean_db" -> N[20*Log[10, 2], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
