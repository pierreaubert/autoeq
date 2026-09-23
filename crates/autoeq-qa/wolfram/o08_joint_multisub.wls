(* autoeq-qa case autoeq-qa.o08-joint-multisub.v1 (catalogue family O08).
   Independent evaluation of the joint multi-sub scalarization from its
   documented definition: per-bin seat-to-seat level variance averaged
   over bins (never SPL summed as pressure), mean squared target error,
   and the unnormalized output penalty
   4 Max[meanLoss-3,0]^2 + mean Max[ref-cand-12,0]^2 of the seat-average
   absolute response against the declared power-sum reference. *)

Off[General::munfl];
seats = {{80., 82., 81., 79.}, {78., 83., 80., 80.}};
reference = {85., 86., 85., 84.};
target = {80., 81., 81., 80.};
weights = {1., 1., 1.};
nbins = Length[reference];

binMean[b_] := N[(seats[[1, b]] + seats[[2, b]])/2, 16];
variation = N[Sum[
  ((seats[[1, b]] - binMean[b])^2 + (seats[[2, b]] - binMean[b])^2)/2,
  {b, 1, nbins}]/nbins, 16];
targetError = N[Sum[
  ((seats[[1, b]] - target[[b]])^2 + (seats[[2, b]] - target[[b]])^2)/2,
  {b, 1, nbins}]/nbins, 16];
average = Table[binMean[b], {b, 1, nbins}];
meanLoss = N[Sum[reference[[b]] - average[[b]], {b, 1, nbins}]/nbins, 16];
nullLoss = N[Sum[Max[reference[[b]] - average[[b]] - 12., 0.]^2, {b, 1, nbins}]/nbins, 16];
outputDrive = N[4 Max[meanLoss - 3., 0.]^2 + nullLoss, 16];
total = N[weights[[1]] variation + weights[[2]] outputDrive + weights[[3]] targetError, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o08-joint-multisub.v1",
  "family" -> "O08",
  "sample_rate_hz" -> 48000.,
  "grid" -> "4-bin abstract sub band grid (bins 1..4)",
  "units" -> "dB SPL absolute; penalties in dB^2",
  "calibration" -> "absolute seat levels; declared power reference, no renormalization",
  "seat_levels_db" -> seats,
  "power_reference_db" -> reference,
  "target_db" -> target,
  "weights" -> <|"variation" -> weights[[1]], "output_drive" -> weights[[2]], "target_error" -> weights[[3]]|>,
  "seat_average_db" -> average,
  "variation" -> variation,
  "output_drive" -> outputDrive,
  "target_error" -> targetError,
  "total" -> total
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
