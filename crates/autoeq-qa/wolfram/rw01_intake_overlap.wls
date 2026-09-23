(* autoeq-qa case autoeq-qa.rw01-intake-overlap.v1 (catalogue family RW01).
   Independent resource-intake grid alignment: the supported overlap of
   two measurement grids is the closed-form support intersection
   [Max of minima, Min of maxima], or nothing when either grid is too
   short, non-positive/nonfinite, or the supports are disjoint. The
   oracle never calls supported_overlap or validate_intake_grids; it
   restates the intersection from the ordering definition, so it
   catches index-zipped averages, silent disjoint intake, and
   min/max swaps. *)

Off[General::munfl];
gridA = {20., 50., 100., 500., 1000., 5000., 10000., 20000.};
gridB = {30., 100., 1000., 8000., 16000.};
gridC = {21000., 22000.};

overlapAB = {N[Max[Min[gridA], Min[gridB]], 16], N[Min[Max[gridA], Max[gridB]], 16]};
disjointAC = (Max[Min[gridA], Min[gridC]] >= Min[Max[gridA], Max[gridC]]);

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw01-intake-overlap.v1",
  "family" -> "RW01",
  "sample_rate_hz" -> 48000.,
  "grid" -> "explicit intake grids, Hz",
  "units" -> "Hz; overlap endpoints in Hz",
  "calibration" -> "no level calibration; grids are acquisition-native supports",
  "grid_a_hz" -> gridA,
  "grid_b_hz" -> gridB,
  "grid_c_hz" -> gridC,
  "expected_overlap_ab_hz" -> overlapAB,
  "expected_disjoint_ac" -> disjointAC
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
