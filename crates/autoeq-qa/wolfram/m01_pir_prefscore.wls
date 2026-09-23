(* autoeq-qa case autoeq-qa.m01-pir-prefscore.v1 (catalogue family M01).
   Independent evaluation of the CEA-2034 PIR mix, pressure<->SPL
   conversions, NBD/MAD, LFX, R^2 smoothness, and the preference-score
   formula (the same published definitions the Rust cea2034 module
   implements), plus the approximate-PEQ rescore path (dB-domain addition
   of the PEQ trace before scoring). Direct sums only; never a
   transliteration of the Rust loops. *)

Off[General::munfl];
freqs = {100., 200., 400., 800., 1600., 3200., 6400., 12800., 16000.};
on = {80., 82., 81., 79., 78., 80., 82., 81., 79.};
lw = {81., 81., 80., 79., 78., 79., 81., 80., 78.};
er = {79., 80., 80., 78., 77., 78., 80., 79., 77.};
sp = {70., 71., 79., 77., 76., 77., 79., 78., 76.};
peq = {0., 0.5, -0.5, 1., -1., 0.5, 0., -0.5, 0.};

(* Rust intervals [(1,3),(3,5),(5,7)] (0-based, end-exclusive) as 1-based spans. *)
bands = {{2, 3}, {4, 5}, {6, 7}};

spl2p[s_] := N[10^((s - 105)/20), 16];
p2spl[p_] := N[20*Log[10, p] + 105, 16];

pirOf[lwV_, erV_, spV_] := Table[
  p2spl[Sqrt[0.12*spl2p[lwV[[i]]]^2 + 0.44*spl2p[erV[[i]]]^2 + 0.44*spl2p[spV[[i]]]^2]],
  {i, Length[lwV]}];

madOf[v_, {a_, b_}] := Module[{seg = Take[v, {a, b}], m},
  m = Mean[seg];
  Mean[Abs[# - m] & /@ seg]];
nbdOf[v_] := Mean[madOf[v, #] & /@ bands];

lfxOf[f_, lwV_, spV_] := Module[{iMin, iMax, ref, cand, last},
  iMin = First[Flatten[Position[f, x_ /; x >= 300.]]];
  iMax = First[Flatten[Position[f, x_ /; x > 10000.]], {Length[f] + 1}];
  ref = Mean[Take[lwV, {iMin, iMax - 1}]] - 6.;
  cand = Flatten[Position[Table[If[f[[i]] <= 300. && spV[[i]] <= ref, 1, 0], {i, Length[f]}], 1]];
  If[Length[cand] == 0, Return[N[Log[10, f[[1]]], 16]]];
  last = cand[[1]];
  Do[If[cand[[k]] == last + 1, last = cand[[k]], Break[]], {k, 2, Length[cand]}];
  N[Log[10, f[[last + 1]]], 16]];

smOf[f_, v_] := Module[{iMin, iMax, x, y, n, sx, sy, sxx, syy, sxy, num, den},
  iMin = First[Flatten[Position[f, x_ /; x > 100.]]];
  iMax = First[Flatten[Position[f, x_ /; x >= 16000.]], {Length[f] + 1}];
  x = N[Log[10, #] & /@ Take[f, {iMin, iMax - 1}], 16];
  y = Take[v, {iMin, iMax - 1}];
  n = Length[x];
  sx = Total[x]; sy = Total[y];
  sxx = Total[#^2 & /@ x]; syy = Total[#^2 & /@ y];
  sxy = Total[Table[x[[i]]*y[[i]], {i, n}]];
  num = (n*sxy - sx*sy)^2;
  den = (n*sxx - sx^2)*(n*syy - sy^2);
  N[num/den, 16]];

prefOf[nbdOn_, nbdPir_, lfxV_, smV_] :=
  N[12.69 - 2.49*nbdOn - 2.99*nbdPir - 4.31*lfxV + 2.32*smV, 16];

pir = pirOf[lw, er, sp];
nbdOn = nbdOf[on]; nbdPir = nbdOf[pir];
lfxV = lfxOf[freqs, lw, sp]; smV = smOf[freqs, pir];
pref = prefOf[nbdOn, nbdPir, lfxV, smV];

on2 = on + peq; lw2 = lw + peq; sp2 = sp + peq; pir2 = pir + peq;
nbdOn2 = nbdOf[on2]; nbdPir2 = nbdOf[pir2];
lfxV2 = lfxOf[freqs, lw2, sp2]; smV2 = smOf[freqs, pir2];
pref2 = prefOf[nbdOn2, nbdPir2, lfxV2, smV2];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.m01-pir-prefscore.v1",
  "family" -> "M01",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> freqs,
  "units" -> "dB SPL (ref 20uPa via 20log10(p)+105); frequencies Hz",
  "calibration" -> "none (absolute SPL fixture)",
  "grid_note" -> "shared 9-point grid on every curve; no resampling (never zip unequal grids)",
  "on_db" -> on, "lw_db" -> lw, "er_db" -> er, "sp_db" -> sp, "peq_db" -> peq,
  "intervals_0based_exclusive" -> {{1, 3}, {3, 5}, {5, 7}},
  "pir_db" -> pir,
  "nbd_on" -> nbdOn, "nbd_pir" -> nbdPir,
  "lfx_log10_hz" -> lfxV, "sm_r2" -> smV, "pref_score" -> pref,
  "nbd_on_peq" -> nbdOn2, "nbd_pir_peq" -> nbdPir2,
  "lfx_log10_hz_peq" -> lfxV2, "sm_r2_peq" -> smV2, "pref_score_peq" -> pref2
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
