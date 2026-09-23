(* autoeq-qa case autoeq-qa.rm01-spl-level.v1 (catalogue family RM01).
   Independent check of SPL-calibration level arithmetic, complementing
   the rm01-schroeder-spl room-volume relation: amplitude level doubling
   adds exactly 20*Log10[2] dB, the forward/inverse pair round-trips on
   the representable path, the inverse hits exactly 1.0 at the offset
   reference and clamps above it, and the forward map floors at the
   single-precision epsilon. Compared against SplCalibration. Units dB
   SPL and linear peak; offset 100 dB. *)

Off[General::munfl];
offset = 100.;
eps32 = N[2^-23, 16];
dbspl[p_] := N[20.*Log10[Max[p, eps32]] + offset, 16];
peakFor[t_] := N[Min[Max[10.^((t - offset)/20.), 0.], 1.], 16];
doublingPairs = {{0.125, 0.25}, {0.25, 0.5}, {0.05, 0.1}};
doublingDiffs = Map[N[dbspl[#[[2]]] - dbspl[#[[1]]], 16] &, doublingPairs];
exactSixDb = N[20.*Log10[2.], 16];
targets = {40., 70., 94.};
peaks = peakFor /@ targets;
roundTrip = dbspl /@ peaks;
atOffset = peakFor[offset];
aboveClamp = peakFor[140.];
floorDb = dbspl[1.*^-9];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rm01-spl-level.v1",
  "family" -> "RM01",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "scalar level arithmetic; no sampling involved",
  "grid_hz" -> Null,
  "units" -> "dB SPL and linear peak in [0, 1]",
  "calibration" -> "spl_offset_db = 100 dB; single-precision epsilon floor",
  "spl_offset_db" -> offset,
  "doubling_pairs" -> doublingPairs,
  "doubling_diffs_db" -> doublingDiffs,
  "exact_doubling_db" -> exactSixDb,
  "targets_db_spl" -> targets,
  "peaks_for_target" -> peaks,
  "round_trip_db_spl" -> roundTrip,
  "peak_at_offset" -> atOffset,
  "peak_above_clamp" -> aboveClamp,
  "floor_db_spl" -> floorDb
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
