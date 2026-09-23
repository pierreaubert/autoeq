(* autoeq-qa case autoeq-qa.ac03-benchmark-stats.v1 (catalogue family AC03).
   Benchmark distribution statistics on one exact small sample:
   arithmetic mean, sample standard deviation (n-1 denominator;
   std = 0 for n = 1), linear-interpolation percentiles
   (pos = q*(n-1), lerp between floor/ceil ranks), finite differences
   (None unless both sides finite), percentages, and tied-best masks
   (within 1e-6 of the maximum; None unless every entry finite).
   All derived here from the published definitions on the explicit
   inputs below. Units: score points / counts as labeled; calibration
   n/a (pure arithmetic). *)

Off[General::munfl];
data = {2., 4., 4., 4., 5., 5., 7., 9.};
sorted = Sort[data];
n = Length[data];
mean = N[Mean[data], 16];
std = N[Sqrt[Total[(# - mean)^2 & /@ data]/(n - 1)], 16];
pct[q_] := Module[{pos = q*(n - 1), lo, hi, w},
  lo = Floor[pos]; hi = Ceiling[pos];
  w = pos - lo;
  sorted[[lo + 1]]*(1 - w) + sorted[[hi + 1]]*w
];
p10 = N[pct[0.10], 16]; p25 = N[pct[0.25], 16];
p50 = N[pct[0.50], 16]; p75 = N[pct[0.75], 16]; p90 = N[pct[0.90], 16];
finiteDiff = 5. - 4.;
pct2of8 = N[2*100./8, 16];
votes = {7.1, 7.1000005, 6.9, 7.0999999};
best = Max[votes];
mask = Abs[# - best] <= 10^-6 & /@ votes;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ac03-benchmark-stats.v1",
  "family" -> "AC03",
  "sample_rate_hz" -> "n/a (pure arithmetic)",
  "units" -> "score points; percentages in %",
  "calibration" -> "n/a",
  "data" -> data,
  "mean" -> mean,
  "std_sample" -> std,
  "min" -> N[Min[data], 16],
  "max" -> N[Max[data], 16],
  "p10" -> p10, "p25" -> p25, "median" -> p50, "p75" -> p75, "p90" -> p90,
  "finite_diff_5_4" -> N[finiteDiff, 16],
  "percentage_2_of_8" -> pct2of8,
  "tie_votes" -> votes,
  "tie_eps" -> 10.^-6,
  "tied_best_mask" -> mask
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
