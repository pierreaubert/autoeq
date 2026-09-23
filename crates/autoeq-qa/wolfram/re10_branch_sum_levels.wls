(* autoeq-qa case autoeq-qa.re10-branch-sum-levels.v1 (catalogue family RE10).
   Independent evaluation of a gain/delay/polarity-weighted complex branch
   sum with no crossover filters (CrossoverType::None path) plus the
   two-band target-level adjustment, from closed forms only. The sub
   branch (flat 84 dB, inverted polarity = +180 deg) spans 20..300 Hz;
   the main branch (flat 80 dB, missing phase = 0 deg convention) spans
   20..20000 Hz. Reference bands [40, 80] Hz (sub) and [150, 250] Hz
   (main) have band means 84 and 80 dB, so the declared level adjustment
   is deltaSub = -4 dB applied to the sub gain; the combined response is
   the complex pressure sum with 10^(g/20) gains and Exp[-I 2 Pi f tau]
   delays (1 ms sub bulk delay), reported with 20 Log10 magnitudes
   floored at 1e-12. No target-shape claim beyond the declared band
   means is made. *)

Off[General::munfl];
sr = 48000.;
n = 100;
freqs = N[10^Range[Log10[20.], Log10[300.], (Log10[300.] - Log10[20.])/(n - 1)], 16];

subSpl = 84.;
mainSpl = 80.;
subBand = {40., 80.};
mainBand = {150., 250.};
subMean = subSpl;
mainMean = mainSpl;
deltaSub = N[mainMean - subMean, 16];

gSub = N[10^(deltaSub/20), 16];
gMain = N[10^(0./20), 16];
tauSub = 0.001;
resp[f_] := Module[{zSub, zMain},
  zSub = N[10^(subSpl/20), 16] * Exp[I Pi] * gSub * Exp[-I N[2 Pi f, 16] tauSub];
  zMain = N[10^(mainSpl/20), 16] * gMain;
  zSub + zMain];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
comb = resp /@ freqs;
mags = (N[20 Log10[Max[Abs[#], 10^-12]], 16] & /@ comb);
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re10-branch-sum-levels.v1",
  "family" -> "RE10",
  "sample_rate_hz" -> sr,
  "grid" -> "logspace(100, 20 Hz, 300 Hz); driver native spans sub 20..300, main 20..20000",
  "units" -> "Hz, dB SPL, degrees, ms",
  "calibration" -> "absolute calibrated branch SPL; flat branches; no output normalization",
  "crossover" -> "None",
  "sub_band_hz" -> subBand,
  "main_band_hz" -> mainBand,
  "sub_band_mean_db" -> subMean,
  "main_band_mean_db" -> mainMean,
  "delta_sub_db" -> deltaSub,
  "gains_db" -> {deltaSub, 0.},
  "delays_ms" -> {1., 0.},
  "sub_inverted" -> True,
  "main_phase_missing" -> True,
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair /@ comb),
  "combined_db" -> mags
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
