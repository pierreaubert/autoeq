(* autoeq-qa case autoeq-qa.c08-local-q.v1 (catalogue family C08).
   Independent piecewise log-frequency maximum-Q envelope: interior
   queries interpolate linearly in Log frequency, exterior queries hold
   the nearest endpoint; the effective limit is Min[global, local].
   Compared against autoeq_core::constraint_envelope. *)

Off[General::munfl];
knotFreqs = {100., 1000.};
knotQ = {2., 4.};
queries = {50., 100., Sqrt[100.*1000.], 1000., 2000.};

env[f_] := Module[{t = N[Log[f], 16], x0 = N[Log[knotFreqs[[1]]], 16],
    x1 = N[Log[knotFreqs[[2]]], 16]},
  Which[
    f <= knotFreqs[[1]], N[knotQ[[1]], 16],
    f >= knotFreqs[[2]], N[knotQ[[2]], 16],
    True, N[knotQ[[1]] + (knotQ[[2]] - knotQ[[1]])*(t - x0)/(x1 - x0), 16]
  ]
];
envVals = (env[#] & /@ queries);
effLoose = (N[Min[6., #], 16] & /@ envVals);
effTight = (N[Min[1.5, #], 16] & /@ envVals);

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c08-local-q.v1",
  "family" -> "C08",
  "sample_rate_hz" -> Null,
  "units" -> <|"frequency" -> "Hz", "max_q" -> "dimensionless"|>,
  "calibration" -> "none-applies-pure-constraint-envelope",
  "knot_freqs_hz" -> knotFreqs,
  "knot_max_q" -> knotQ,
  "query_freqs_hz" -> (N[#, 16] & /@ queries),
  "envelope_max_q" -> envVals,
  "global_loose" -> 6.,
  "effective_loose" -> effLoose,
  "global_tight" -> 1.5,
  "effective_tight" -> effTight
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
