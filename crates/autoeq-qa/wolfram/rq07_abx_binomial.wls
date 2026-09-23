(* autoeq-qa case autoeq-qa.rq07-abx-binomial.v1 (catalogue family RQ07).
   Exact one-sided binomial upper-tail P(X >= k) at p = 1/2 from integer
   combinatorics (exact rational sum, machine-independent), the smallest
   k attaining a level alpha, and the Wilson score 95% interval from its
   closed form with the fixed quantile Z = 1.959963984540054.
   Compared against roomeq_quality::abx_p_value (mode-anchored floating
   recurrence), abx_min_correct, and wilson_ci95. *)

Off[General::munfl];
pCases = {{8, 10}, {15, 20}, {12, 12}, {0, 10}, {5, 10}};
tail[k_, n_] := If[k == 0, 1., N[Sum[Binomial[n, j], {j, k, n}]/2^n, 16]];
pValues = tail @@@ pCases;

thresholdCases = {{10, 0.05}, {20, 0.05}, {12, 0.01}, {1, 0.05}};
minCorrect[n_, a_] := Module[{ks = Select[Range[0, n], tail[#, n] <= a &]},
  If[ks == {}, Null, First[ks]]];
thresholds = minCorrect @@@ thresholdCases;

wilson[c_, t_] := Module[{z = N[1.959963984540054, 16], n = N[t, 16], p, den, cen, half},
  p = N[c, 16]/n;
  den = 1 + z^2/n;
  cen = (p + z^2/(2 n))/den;
  half = z*Sqrt[p*(1 - p)/n + z^2/(4 n^2)]/den;
  {Min[Max[cen - half, 0.], 1.], Min[Max[cen + half, 0.], 1.]}];
wilsonCases = {{8, 10}, {0, 5}, {5, 5}};
wilsonIntervals = wilson @@@ wilsonCases;

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rq07-abx-binomial.v1",
  "family" -> "RQ07",
  "wolfram_engine" -> "15.0.0",
  "p_cases" -> pCases,
  "p_values" -> pValues,
  "threshold_cases" -> thresholdCases,
  "thresholds" -> thresholds,
  "wilson_cases" -> wilsonCases,
  "wilson_intervals" -> wilsonIntervals
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
