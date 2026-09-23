(* autoeq-qa case autoeq-qa.o01-asymmetric-weights.v1 (catalogue family O01).
   Independent evaluation of the asymmetric peak/dip loss: per-sample weight
   w(f, sign) from a sigmoid crossfade in log frequency around the transition
   frequency (blend = 1/(1+Exp[-(ln f - ln ft)*k]) with k = 2*ln9/ln2, so ~90%
   of the crossover completes within +-0.5 octaves), peak branch
   bassPeak + blend*(peak - bassPeak), dip branch
   (bassDip + blend*(dip - bassDip))*mask, error rescaled to e*Sqrt[w] and
   scored by the same 0.7/0.3 ERB + band blend as the scalar case.
   Compared against autoeq_optim::loss::weighted_mse_asymmetric
   (custom config so every branch is exercised) and the cached
   PreparedAsymmetricLoss path. Units are dB on a declared grid. *)

Off[General::munfl];
grid = {60., 150., 300., 600., 1500., 6000.};
err = {4., -3., 2., -2., 5., -1.};
mask = {0.5, 0.5, 1., 0., 0.25, 0.25};
peakW = 3.; dipW = 0.5; bassPeakW = 6.; bassDipW = 0.25; transF = 300.;
erbMix = 0.7; bandMix = 0.3;
bassW = 2.; midW = 1.; trebleW = 0.8;

k = N[2*Log[9]/Log[2], 16];
blend[f_] := N[1/(1 + Exp[-(Log[f] - Log[transF])*k]), 16];
peakWt[f_] := N[bassPeakW + blend[f]*(peakW - bassPeakW), 16];
dipWt[f_] := N[bassDipW + blend[f]*(dipW - bassDipW), 16];
n = Length[grid];
w = Table[N[If[err[[i]] > 0., peakWt[grid[[i]]], dipWt[grid[[i]]]*mask[[i]]], 16], {i, 1, n}];
ewtd = N[err*Sqrt[w], 16];

rate[f_] := N[21.4*Log10[1 + 0.00437*f], 16];
rates = rate /@ grid;
trap[axis_] := Table[
  Which[
    i == 1, (axis[[2]] - axis[[1]])/2,
    i == n, (axis[[n]] - axis[[n - 1]])/2,
    True, (axis[[i + 1]] - axis[[i - 1]])/2
  ], {i, 1, n}];
qw = trap[rates];
lw = trap[N[Log[#], 16] & /@ grid];
erbLoss = N[Sqrt[Total[ewtd^2*qw]/Total[qw]], 16];
bandOf[f_] := Which[f <= 200., 1, f <= 4000., 2, True, 3];
bands = bandOf /@ grid;
bandRMS[b_] := Module[{sel = Pick[Range[n], bands, b]},
  N[Sqrt[Total[(ewtd[[sel]]^2)*(lw[[sel]])]/Total[lw[[sel]]]], 16]];
rb = bandRMS[1]; rm = bandRMS[2]; rt = bandRMS[3];
bandLoss = N[(bassW*rb + midW*rm + trebleW*rt)/(bassW + midW + trebleW), 16];
combined = N[erbMix*erbLoss + bandMix*bandLoss, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o01-asymmetric-weights.v1",
  "family" -> "O01",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "dB-domain loss on a declared grid; no sampling involved",
  "grid_hz" -> grid,
  "units" -> "dB",
  "calibration" -> "absolute deviation-from-target dB; no SPL calibration",
  "min_freq_hz" -> 20.,
  "max_freq_hz" -> 20000.,
  "config" -> <|
    "peak_weight" -> peakW, "dip_weight" -> dipW,
    "bass_peak_weight" -> bassPeakW, "bass_dip_weight" -> bassDipW,
    "transition_freq_hz" -> transF|>,
  "errors_db" -> err,
  "null_mask" -> mask,
  "sigmoid_k" -> k,
  "blend" -> (blend /@ grid),
  "sample_weights" -> w,
  "weighted_errors_db" -> ewtd,
  "erb_loss" -> erbLoss,
  "band_loss" -> bandLoss,
  "combined_loss" -> combined
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
