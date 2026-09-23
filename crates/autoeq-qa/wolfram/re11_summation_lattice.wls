(* autoeq-qa case autoeq-qa.re11-summation-lattice.v1 (catalogue family RE11).
   Independent evaluation of Hmain + g Hsub Exp[-j 2 Pi f tau] direct
   summation, the normalized band summation error, and a bounded
   delay/polarity/gain lattice enumeration with refinement, from closed
   forms only (never a transliteration of the Rust search loops).
   Fixture: flat main 80 dB with a pure 3 ms propagation lag, flat sub
   76 dB with zero phase, band 40..120 Hz. The band error is the band
   mean of (ideal - |Hsum|)^2 normalized by the band mean of ideal^2
   with ideal = |Hmain| + |g| |Hsub| (polarity folds into g). The coarse
   lattice (delays {0,1,2,3,4,6} ms, 0 dB gain, both polarities, 12
   candidates) must select (non-inverted, 3 ms, 0 dB) with zero error;
   the refinement lattice around 3 ms confirms it. Gain and polarity
   variants are checked as individual direct evaluations. A second seat with no relative delay replays the accepted
   candidate with a nonzero but bounded error (tolerance 2.0). The delay
   ledger advance outcomes (reduce-then-common-latency) and the sample
   count of 3 ms at 48 kHz are declared from their documented rules. No
   weighting-curve or optimizer-convergence claim is made. *)

Off[General::munfl];
freqs = N[{40., 50., 60., 80., 100., 120.}, 16];
band = {40., 120.};
mainDb = 80.;
subDb = 76.;
trueLag = 0.003;

hMain[f_] := N[10^(mainDb/20), 16] * Exp[-I N[2 Pi f, 16] trueLag];
hSub[f_] := N[10^(subDb/20), 16];
bandIdx = Flatten[Position[freqs, f_ /; band[[1]] <= f <= band[[2]]]];
bandFreqs = freqs[[bandIdx]];

bandError[polInv_, delay_, gainDb_] := Module[{g, hs, id, er},
  g = N[10^(gainDb/20), 16] * If[polInv == 1, -1., 1.];
  hs = Table[
    hMain[f] + hSub[f] * g * Exp[-I N[2 Pi f, 16] delay],
    {f, bandFreqs}];
  id = Table[
    Abs[hMain[f]] + Abs[g] * Abs[hSub[f]],
    {f, bandFreqs}];
  er = Table[(id[[i]] - Abs[hs[[i]]])^2, {i, 1, Length[hs]}];
  N[Mean[er]/Mean[id^2], 16]];

coarseDelays = {0., 0.001, 0.002, 0.003, 0.004, 0.006};
coarseGains = {0.};
coarseLat = Flatten[Table[{p, t, g},
  {p, {0, 1}}, {t, coarseDelays}, {g, coarseGains}], 2];
coarseErrs = bandError[#[[1]], #[[2]], #[[3]]] & /@ coarseLat;
coarseBestPos = First[Ordering[coarseErrs, 1]];
coarseBest = coarseLat[[coarseBestPos]];

fineDelays = {0.0025, 0.00275, 0.003, 0.00325, 0.0035};
fineLat = Table[{0, t, 0.}, {t, fineDelays}];
fineErrs = bandError[#[[1]], #[[2]], #[[3]]] & /@ fineLat;
fineBestPos = First[Ordering[fineErrs, 1]];
fineBest = fineLat[[fineBestPos]];

offLatticeErr = bandError[0, 0.0025, 0.];
invertedErr = bandError[1, 0.003, 0.];

(* Second seat: no relative delay, different levels; replay accepted. *)
seat2MainDb = 78.;
seat2SubDb = 74.;
seat2Err = Module[{g, hs, id, er, hM, hS},
  g = N[10^(coarseBest[[3]]/20), 16];
  hM[f_] := N[10^(seat2MainDb/20), 16];
  hS[f_] := N[10^(seat2SubDb/20), 16];
  hs = Table[
    hM[f] + hS[f] * g * Exp[-I N[2 Pi f, 16] coarseBest[[2]]],
    {f, bandFreqs}];
  id = Table[Abs[hM[f]] + Abs[g] * Abs[hS[f]], {f, bandFreqs}];
  er = Table[(id[[i]] - Abs[hs[[i]]])^2, {i, 1, Length[hs]}];
  N[Mean[er]/Mean[id^2], 16]];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re11-summation-lattice.v1",
  "family" -> "RE11",
  "sample_rate_hz" -> 48000.,
  "grid" -> "explicit 6-point grid 40..120 Hz shared by main and sub",
  "units" -> "Hz, dB SPL, seconds, dB gain",
  "calibration" -> "absolute branch levels; flat 80 dB main with 3 ms lag, flat 76 dB sub",
  "formula" -> "Hsum = Hmain + g Hsub Exp[-j 2 Pi f tau]; err = mean((ideal-|Hsum|)^2)/mean(ideal^2)",
  "band_hz" -> band,
  "freqs_hz" -> freqs,
  "main_db" -> mainDb,
  "sub_db" -> subDb,
  "true_lag_s" -> trueLag,
  "coarse_delays_s" -> coarseDelays,
  "coarse_gains_db" -> coarseGains,
  "coarse_errors" -> coarseErrs,
  "coarse_evaluated" -> Length[coarseErrs],
  "coarse_best_polarity_delay_gain" -> coarseBest,
  "coarse_best_error" -> coarseErrs[[coarseBestPos]],
  "fine_delays_s" -> fineDelays,
  "fine_errors" -> fineErrs,
  "fine_best_polarity_delay_gain" -> fineBest,
  "fine_best_error" -> fineErrs[[fineBestPos]],
  "off_lattice_error" -> offLatticeErr,
  "off_lattice_point" -> {0, 0.0025, 0.},
  "inverted_error" -> invertedErr,
  "inverted_point" -> {1, 0.003, 0.},
  "seat2_main_db" -> seat2MainDb,
  "seat2_sub_db" -> seat2SubDb,
  "seat2_tolerance" -> 2.,
  "seat2_error" -> seat2Err,
  "ledger_label" -> "sub-1",
  "ledger_delay_s" -> 0.003,
  "ledger_rate_hz" -> 48000.,
  "ledger_delay_samples" -> 144,
  "advance_first_s" -> 0.001,
  "advance_first_outcome" -> "reduced",
  "advance_first_from_s" -> 0.003,
  "advance_first_to_s" -> 0.002,
  "advance_second_s" -> 0.005,
  "advance_second_outcome" -> "common",
  "advance_second_added_s" -> 0.003
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
