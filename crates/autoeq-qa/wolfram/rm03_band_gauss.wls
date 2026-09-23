(* autoeq-qa case autoeq-qa.rm03-band-gauss.v1 (catalogue family RM03).
   Independent evaluation of the correction-band intersection
   active = [Max[policyMin, obsMin], Min[policyMax, obsMax]] (legacy:
   the observation band itself; data support intersects again by the
   same max/min rule) and of the Gaussian listening-area truncation
   box mean +/- k*Sqrt[variance] with the configured boundary
   tolerance 1e-6*Max[Abs[lo], Abs[hi], 1]. Compared against
   OptimizerConfig::active_correction_band(_for_data),
   CorrectionBandPolicy::validate_against, and the continuous-area
   validation of roomeq-model. Bands in Hz, area in metres. *)

Off[General::munfl];
obsMin = 20.; obsMax = 20000.;
polMin = 40.; polMax = 16000.;
active = {N[Max[polMin, obsMin], 16], N[Min[polMax, obsMax], 16]};
legacy = {obsMin, obsMax};
dataA = {30., 18000.};
dataB = {50., 8000.};
forDataA = {N[Max[active[[1]], dataA[[1]]], 16], N[Min[active[[2]], dataA[[2]]], 16]};
forDataB = {N[Max[active[[1]], dataB[[1]]], 16], N[Min[active[[2]], dataB[[2]]], 16]};

mean = {0., 0.}; variance = {0.25, 1.}; k = 3.;
sigma = Map[N[Sqrt[#], 16] &, variance];
expectedBox = MapThread[{N[#1 - k*#2, 16], N[#1 + k*#2, 16]} &, {mean, sigma}];
tolerances = Map[N[1.*^-6*Max[Abs[#[[1]]], Abs[#[[2]]], 1.], 16] &, expectedBox];
matching = {{-1.5, 1.5}, {-3., 3.}};
mismatched = {{-1.5, 1.5}, {-3., 3.001}};
gapAxis1 = N[Abs[mismatched[[2, 2]] - expectedBox[[2, 2]]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rm03-band-gauss.v1",
  "family" -> "RM03",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "band intersections and a spatial truncation box; no sampling involved",
  "grid_hz" -> Null,
  "units" -> "correction bands in Hz; listening-area box in metres",
  "calibration" -> "no SPL calibration",
  "observation_band_hz" -> {obsMin, obsMax},
  "policy_band_hz" -> {polMin, polMax},
  "active_band_hz" -> active,
  "legacy_band_hz" -> legacy,
  "data_supports_hz" -> {dataA, dataB},
  "active_for_data_hz" -> {forDataA, forDataB},
  "gaussian" -> <|
    "mean_m" -> mean, "variance_m2" -> variance,
    "truncation_sigmas" -> k,
    "sigma_m" -> sigma,
    "expected_bounds_m" -> expectedBox,
    "boundary_tolerances_m" -> tolerances,
    "matching_bounds_m" -> matching,
    "mismatched_bounds_m" -> mismatched,
    "mismatch_gap_axis1_m" -> gapAxis1|>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
