(* autoeq-qa case autoeq-qa.rw04-crossover-bounded-sum.v1 (catalogue family RW04).
   Independent input-route-output replay and bounded sum: each
   candidate (polarity, delay_s, gain_db) forms
   Hsum = Hmain + s * 10^(g/20) * Hsub * Exp[-I 2 Pi f tau]
   (s = -1 inverted, +1 otherwise) from polar dB/degree transfers,
   with the unknown-phase magnitude bound ideal = |Hmain| + |g||Hsub|
   (triangle inequality: |Hsum| <= ideal, equality only in phase).
   Band error = Mean[(ideal - |Hsum|)^2]/Mean[ideal^2] over in-band
   bins; the winner is the first strict minimum in
   polarity {False, True} x delay x gain order. The oracle never
   calls search_summation or verify_crossover_alignment; it restates
   the published error, so it catches sign flips, dB-factor slips,
   conjugation, and bound violations. *)

Off[General::munfl];
freqs = {40., 50., 60., 70., 80., 90., 100., 110., 120.};
mainMag = Table[0., {Length[freqs]}];
mainPh = Table[0., {Length[freqs]}];
subMag = Table[-6., {Length[freqs]}];
subPh = Table[30., {Length[freqs]}];
band = {60., 100.};
delays = {0., 0.0005, 0.001};
gains = {-6., 0.};

toC[magDb_, phDeg_] := N[10^(magDb/20), 16]*{Cos[N[phDeg*Pi/180, 16]], Sin[N[phDeg*Pi/180, 16]]};
cadd[{a_, b_}, {c_, d_}] := {a + c, b + d};
cmul[{a_, b_}, {c_, d_}] := {a*c - b*d, a*d + b*c};
cnorm[{a_, b_}] := Sqrt[a^2 + b^2];
shift[f_, tau_] := {Cos[N[-2 Pi f tau, 16]], Sin[N[-2 Pi f tau, 16]]};

inBand = Table[band[[1]] <= freqs[[i]] <= band[[2]], {i, Length[freqs]}];
mainC = Table[toC[mainMag[[i]], mainPh[[i]]], {i, Length[freqs]}];
subC = Table[toC[subMag[[i]], subPh[[i]]], {i, Length[freqs]}];

candError[pol_, tau_, g_] := Module[{glin, hsum, ideal, num = 0., den = 0.},
  glin = N[10^(g/20), 16]*If[pol, -1., 1.];
  Do[
   If[inBand[[i]],
    hsum = cadd[mainC[[i]], cmul[subC[[i]], {glin*shift[freqs[[i]], tau][[1]], glin*shift[freqs[[i]], tau][[2]]}]];
    ideal = cnorm[mainC[[i]]] + Abs[glin]*cnorm[subC[[i]]];
    num += (ideal - cnorm[hsum])^2;
    den += ideal^2;
    ],
   {i, Length[freqs]}];
  N[num, 16]/N[den, 16]
];

cands = Flatten[Table[<|
   "polarity_inverted" -> pol, "delay_s" -> tau, "gain_db" -> g,
   "band_error" -> N[candError[pol, tau, g], 16]|>,
  {pol, {False, True}}, {tau, delays}, {g, gains}], 2];
best = cands[[1]];
Do[If[c["band_error"] < best["band_error"], best = c], {c, cands}];

winGain = N[10^(best["gain_db"]/20), 16]*If[best["polarity_inverted"], -1., 1.];
winSum = Table[
  cadd[mainC[[i]], cmul[subC[[i]],
    {winGain*shift[freqs[[i]], best["delay_s"]][[1]], winGain*shift[freqs[[i]], best["delay_s"]][[2]]}]],
  {i, Length[freqs]}];
winMag = N[cnorm[#], 16] & /@ winSum;
winIdeal = Table[N[cnorm[mainC[[i]]] + Abs[winGain]*cnorm[subC[[i]]], 16], {i, Length[freqs]}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw04-crossover-bounded-sum.v1",
  "family" -> "RW04",
  "sample_rate_hz" -> 48000.,
  "grid" -> "explicit 9-point grid 40..120 Hz; band [60, 100] Hz",
  "units" -> "dB magnitudes, degrees phase; delays in seconds",
  "calibration" -> "flat 0 dB/0 deg main, flat -6 dB/30 deg sub, common timing scope",
  "freqs_hz" -> freqs,
  "main_mag_db" -> mainMag,
  "main_phase_deg" -> mainPh,
  "sub_mag_db" -> subMag,
  "sub_phase_deg" -> subPh,
  "band_hz" -> band,
  "delays_s" -> delays,
  "gains_db" -> gains,
  "candidates" -> cands,
  "evaluated" -> Length[cands],
  "winner" -> best,
  "winner_combined_mag" -> winMag,
  "winner_ideal_mag" -> winIdeal
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
