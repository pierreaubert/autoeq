(* autoeq-qa case autoeq-qa.rm04-phase-thresholds.v1 (catalogue family RM04).
   Independent evaluation of three exact model rules: phase-wrap
   serialization wrap(p) = Mod[p + 180, 360] - 180 with nonfinite
   passthrough; crossover-cancellation acceptance (finite non-negative
   candidate and limit; accept iff candidate <= limit + 0.05 or the
   baseline improves by strictly more than 0.05); and residual
   min/max/RMS over the finite entries (empty or all-nonfinite gives
   0/0/0). Compared against CurveData phase serialization,
   crossover_cancellation_accepted, and MixedPhaseCorrectionReport::
   from_residual. Degrees and dB as declared. *)

Off[General::munfl];
phases = {180., -180., 270., -270., 360., 181., -181., 720., -540., 45.5};
wrapped = Map[N[Mod[# + 180., 360.] - 180., 16] &, phases];
tol = 0.05;
accept[c_, b_, l_] := Module[{cf = Head[c] =!= Symbol, ok},
  ok = NumericQ[c] && NumericQ[l] && c >= 0. && l >= 0.;
  If[!ok, False,
    If[c <= l + tol, True,
      If[NumericQ[b], b - c > tol, False]]]];
acceptCases = {{4., 10., 3.}, {10., 10., 3.}, {9.99, 10., 3.}, {3.05, 10., 3.},
  {3.051, 3.06, 3.}, {2., 0., -1.}, {-1., 0., 3.}};
(* Missing baselines are represented by Null and handled as absent. *)
acceptExpected = {True, False, False, True, False, False, False};
acceptMissing = {{4., Null, 3.}, {4., Null, 5.}, {3.049, Null, 3.}, {3.051, Null, 3.}};
acceptMissingExpected = {False, True, True, False};
residual = {10., -20., 5., -5., 15.};
resMin = N[Min[residual], 16];
resMax = N[Max[residual], 16];
resRms = N[Sqrt[Total[#^2 & /@ residual]/Length[residual]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rm04-phase-thresholds.v1",
  "family" -> "RM04",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "serialization wrap and scalar threshold rules; no sampling involved",
  "grid_hz" -> Null,
  "units" -> "phase in degrees; cancellation/residual in dB",
  "calibration" -> "no SPL calibration",
  "phases_deg" -> phases,
  "wrapped_deg" -> wrapped,
  "nonfinite_policy" -> "NaN and +/-Infinity pass through unwrapped",
  "cancellation_tolerance_db" -> tol,
  "accept_cases" -> acceptCases,
  "accept_expected" -> acceptExpected,
  "accept_missing_baseline" -> acceptMissing,
  "accept_missing_expected" -> acceptMissingExpected,
  "residual_deg" -> residual,
  "residual_min_deg" -> resMin,
  "residual_max_deg" -> resMax,
  "residual_rms_deg" -> resRms
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
