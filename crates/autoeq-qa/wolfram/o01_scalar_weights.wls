(* autoeq-qa case autoeq-qa.o01-scalar-weights.v1 (catalogue family O01).
   Independent evaluation of the flat scalar-weight loss blend:
   Glasberg-Moore ERB-rate R(f) = 21.4*Log10[1+0.00437 f] with trapezoidal
   cell widths on the rate axis, natural-log trapezoidal quadrature weights,
   per-band RMS on bass [20,200] / mid (200,4000] / treble (4000,20000] Hz
   blended as (2*Rb + 1*Rm + 0.8*Rt)/3.8, combined as 0.7*ERB + 0.3*band.
   The masked variant scales only the dip branch (err < 0) by the null
   suppression mask before squaring. Compared against
   autoeq_optim::loss::{flat_loss, erb_weighted_loss, band_weighted_loss}
   and PreparedFlatLoss::new_with_null_suppression. Frequencies are in Hz,
   errors and losses in dB; no sampling is involved. *)

Off[General::munfl];
grid = {20., 100., 300., 1000., 4000., 20000.};
err = {1., 2., -1.5, 0.5, -3., 1.};
mask = {1., 1., 0.25, 1., 0., 1.};
erbMix = 0.7; bandMix = 0.3;
bassW = 2.; midW = 1.; trebleW = 0.8;

rate[f_] := N[21.4*Log10[1 + 0.00437*f], 16];
rates = rate /@ grid;
n = Length[grid];
trap[axis_] := Table[
  Which[
    i == 1, (axis[[2]] - axis[[1]])/2,
    i == n, (axis[[n]] - axis[[n - 1]])/2,
    True, (axis[[i + 1]] - axis[[i - 1]])/2
  ], {i, 1, n}];
ew = trap[rates];
lw = trap[N[Log[#], 16] & /@ grid];

erbLoss = N[Sqrt[Total[err^2*ew]/Total[ew]], 16];

bandOf[f_] := Which[f <= 200., 1, f <= 4000., 2, True, 3];
bands = bandOf /@ grid;
bandRMS[e_, b_] := Module[{sel = Pick[Range[n], bands, b]},
  N[Sqrt[Total[(e[[sel]]^2)*(lw[[sel]])]/Total[lw[[sel]]]], 16]];
rb = bandRMS[err, 1]; rm = bandRMS[err, 2]; rt = bandRMS[err, 3];
bandLoss = N[(bassW*rb + midW*rm + trebleW*rt)/(bassW + midW + trebleW), 16];
combined = N[erbMix*erbLoss + bandMix*bandLoss, 16];

mscale = Table[N[If[err[[i]] < 0., mask[[i]], 1.], 16], {i, 1, n}];
erbLossMasked = N[Sqrt[Total[(err^2)*mscale*ew]/Total[ew]], 16];
rbM = bandRMS[err*Sqrt[mscale], 1]; rmM = bandRMS[err*Sqrt[mscale], 2]; rtM = bandRMS[err*Sqrt[mscale], 3];
bandLossMasked = N[(bassW*rbM + midW*rmM + trebleW*rtM)/(bassW + midW + trebleW), 16];
combinedMasked = N[erbMix*erbLossMasked + bandMix*bandLossMasked, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o01-scalar-weights.v1",
  "family" -> "O01",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "dB-domain loss on a declared grid; no sampling involved",
  "grid_hz" -> grid,
  "units" -> "dB",
  "calibration" -> "absolute deviation-from-target dB; no SPL calibration",
  "measure" -> "glasberg-moore-erb-rate-1990-v1",
  "min_freq_hz" -> 20.,
  "max_freq_hz" -> 20000.,
  "erb_mix" -> erbMix,
  "band_mix" -> bandMix,
  "band_weights" -> {bassW, midW, trebleW},
  "errors_db" -> err,
  "null_mask" -> mask,
  "erb_weights" -> ew,
  "log_quadrature_weights" -> lw,
  "erb_loss" -> erbLoss,
  "band_rms" -> {rb, rm, rt},
  "band_loss" -> bandLoss,
  "combined_loss" -> combined,
  "erb_loss_masked" -> erbLossMasked,
  "band_loss_masked" -> bandLossMasked,
  "combined_loss_masked" -> combinedMasked
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
