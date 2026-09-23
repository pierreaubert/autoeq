(* autoeq-qa case autoeq-qa.re03-candidate-loss.v1 (catalogue family RE03).
   Prepared-EQ candidate-loss recomputation: flat target versus measurement,
   residual, ERB-rate trapezoidal RMS, log-frequency trapezoidal band RMS
   (bass 20-200 / mid 200-4000 / treble 4000-20000 Hz, output weights
   2.0/1.0/0.8 over active bands), combined loss 0.7*ERB + 0.3*band.
   Independent direct-sum evaluation of the published loss composition
   (C log interpolation is the identity on this shared grid; O flat_loss;
   RA ERB-rate axis). Catches dB-factor, weight, band-edge, and
   normalization slips. *)

Off[General::munfl];
sr = 48000.;
freqs = {25., 40., 63., 100., 160., 250., 500., 1000., 2000., 4000., 8000., 16000.};
meas = N[70. + 6.*Exp[-((Log[2, #/60.])/0.8)^2] - 4.*Exp[-((Log[2, #/3000.])/0.7)^2] & /@ freqs, 16];
target = N[Table[70., {Length[freqs]}], 16];
err = N[meas - target, 16];
bandMin = 20.; bandMax = 20000.;

erbRate[f_] := N[21.4*Log10[1. + 0.00437*f], 16];
rates = erbRate /@ freqs;
n = Length[freqs];
erbW = N[Table[If[i == 1, 0.5*(rates[[2]] - rates[[1]]), If[i == n, 0.5*(rates[[n]] - rates[[n - 1]]), 0.5*(rates[[i + 1]] - rates[[i - 1]])]], {i, n}], 16];
erbLoss = N[Sqrt[Total[err^2*erbW]/Total[erbW]], 16];

logf = N[Log[freqs], 16];
quadW = N[Table[If[i == 1, 0.5*(logf[[2]] - logf[[1]]), If[i == n, 0.5*(logf[[n]] - logf[[n - 1]]), 0.5*(logf[[i + 1]] - logf[[i - 1]])]], {i, n}], 16];
bandOf[f_] := Which[bandMin <= f <= 200., 1, 200. <= f <= 4000., 2, 4000. <= f <= 20000., 3, True, 0];
bands = bandOf /@ freqs;
rmsFor[k_] := Module[{sel = Table[If[bands[[i]] == k, quadW[[i]]*err[[i]]^2, 0.], {i, n}], tot = Total[Table[If[bands[[i]] == k, quadW[[i]], 0.], {i, n}]]}, If[tot > 0., N[Sqrt[Total[sel]/tot], 16], 0.]];
rms = rmsFor /@ {1, 2, 3};
outW = {2.0, 1.0, 0.8};
active = Table[If[Total[Table[If[bands[[i]] == k, quadW[[i]], 0.], {i, n}]] > 0., 1., 0.], {k, {1, 2, 3}}];
bandLoss = N[Total[outW*rms*active]/Total[outW*active], 16];
loss = N[0.7*erbLoss + 0.3*bandLoss, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re03-candidate-loss.v1",
  "family" -> "RE03",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "units" -> "dB",
  "calibration" -> "nominal curve levels in dB; no absolute SPL calibration",
  "active_band_hz" -> {bandMin, bandMax},
  "band_edges_hz" -> {{20., 200.}, {200., 4000.}, {4000., 20000.}},
  "band_output_weights" -> outW,
  "erb_band_mix" -> {0.7, 0.3},
  "meas_db" -> meas,
  "target_db" -> target,
  "residual_db" -> err,
  "erb_loss_db" -> erbLoss,
  "band_loss_db" -> bandLoss,
  "loss_db" -> loss
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
