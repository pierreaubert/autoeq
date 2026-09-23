(* autoeq-qa case autoeq-qa.o02-scalarization.v1 (catalogue family O02).
   Independent evaluation of multi-curve scalarization. Each seat carries a
   constant deviation, so by the grid-invariance of the ERB + band blend the
   per-curve flat loss equals the absolute constant (0, 4, 10 dB) on any
   in-range grid; the oracle pins that theorem and derives every reduction
   from it with direct sums:
   Average = mean; WeightedSum = Sum w*l; Minimax = Max;
   VariancePenalized = weighted mean + lambda * weighted population variance
   (weights <= 0 excluded, normalized by the surviving mass);
   fractional-tail CVaR = worst-first sort, worst alpha fraction averaged with
   one fractional split when alpha*N is not integral (alpha >= 1 is the mean,
   alpha -> 0 is the max). Scenario B pins zero-weight-seat exclusion.
   Compared against compute_response_fitness (realized-response entry),
   compute_base_fitness (parameter-vector entry) and compute_pareto_objectives
   over the same per-curve losses. *)

Off[General::munfl];
losses = {0., 4., 10.};
weights = {0.5, 0.25, 0.25};
lambda = 0.5;
alpha = 0.5;

average = N[Mean[losses], 16];
weighted = N[Total[weights*losses], 16];
minimax = N[Max[losses], 16];
wmean = N[Total[weights*losses]/Total[weights], 16];
wvar = N[Total[weights*(losses - wmean)^2]/Total[weights], 16];
varPenalized = N[wmean + lambda*wvar, 16];

sorted = Reverse[Sort[losses]];
mass = N[alpha*Length[losses], 16];
take = Table[N[Min[Max[mass - (i - 1), 0.], 1.], 16], {i, 1, Length[losses]}];
cvar = N[Total[take*sorted]/mass, 16];
cvarMean = N[Mean[losses], 16];

weightsB = {0., 0.5, 0.5};
selB = Pick[losses, Thread[weightsB > 0.]];
selWB = Pick[weightsB, Thread[weightsB > 0.]];
weightedB = N[Total[weightsB*losses], 16];
wmeanB = N[Total[selWB*selB]/Total[selWB], 16];
wvarB = N[Total[selWB*(selB - wmeanB)^2]/Total[selWB], 16];
varPenalizedB = N[wmeanB + lambda*wvarB, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o02-scalarization.v1",
  "family" -> "O02",
  "sample_rate_hz" -> 48000.,
  "sample_rate_note" -> "objective context rate; scalar math itself is rate-free dB algebra",
  "grid_hz" -> {50., 200., 1000., 5000., 15000.},
  "units" -> "dB",
  "calibration" -> "absolute deviation-from-target dB; no SPL calibration",
  "constant_residual_theorem" -> "flat loss of constant deviation c on any in-range grid equals Abs[c]",
  "seat_deviations_db" -> {0., 4., 10.},
  "per_curve_losses" -> losses,
  "weights" -> weights,
  "variance_lambda" -> lambda,
  "cvar_alpha" -> alpha,
  "average" -> average,
  "weighted_sum" -> weighted,
  "minimax" -> minimax,
  "variance_penalized" -> varPenalized,
  "cvar" -> cvar,
  "cvar_alpha1_is_mean" -> cvarMean,
  "zero_weight_weights" -> weightsB,
  "zero_weight_weighted_sum" -> weightedB,
  "zero_weight_variance_penalized" -> varPenalizedB
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
