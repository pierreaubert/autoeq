(* autoeq-qa case autoeq-qa.o14-rerank-ordering.v1 (catalogue family O14).
   Hand-enumerated shortlist reranking reference:
   - evaluator scores are stipulated inputs (lower = better); the oracle
     states the exact expected rank order, including the deterministic
     tie-break (equal scores order by candidate id) and the rule that
     non-finite scores (Infinity, Indeterminate) abort the rerank;
   - a loss-pin switch between nomination and rerank aborts as well.
   Ordering is by direct numeric sort, independent of the Rust rerank. *)

Off[General::munfl];
scores = {{"c-alpha", 1.5}, {"c-beta", 0.5}, {"c-gamma", 0.5}, {"c-delta", 2.}};
ordered = {"c-beta", "c-gamma", "c-alpha", "c-delta"};
ranks = {0, 1, 2, 3};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o14-rerank-ordering.v1",
  "family" -> "O14",
  "sample_rate_hz" -> 48000.,
  "units" -> "evaluator score, lower-better, unitless; rank is an exact integer",
  "calibration" -> "n/a (score-ordering case; scores are stipulated inputs)",
  "freq_grid_hz" -> "n/a (score-level case)",
  "evaluator" -> <|"name" -> "stage4-plumbing", "model_version" -> "0",
    "basis" -> "staged-metric:final-validation"|>,
  "loss_pin" -> <|"loss" -> "speaker-flat", "version" -> "1"|>,
  "transform_tag" -> "o14-fixed",
  "candidate_ids" -> {"c-alpha", "c-beta", "c-gamma", "c-delta"},
  "candidate_scores" -> {1.5, 0.5, 0.5, 2.},
  "candidate_sources" -> {"optimizer_run", "optimizer_run", "pareto_front", "identity"},
  "expected_order" -> ordered,
  "expected_ranks" -> ranks,
  "tie_rule" -> "equal scores break by ascending candidate id",
  "nonfinite_ids" -> {"c-bad-inf", "c-bad-nan"},
  "wrong_pin" -> <|"loss" -> "epa", "version" -> "1"|>,
  "rerun_all_cached" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
