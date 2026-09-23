(* autoeq-qa case autoeq-qa.qa03-seed-matrix.v1 (catalogue family QA03).
   Seed-distribution summary over a small known table plus stage replay.
   Five seed rows (3 accepted, 1 failed, 1 blocked): accepted fraction
   3/5, failed tail fraction 1/5, worst-tail mean of the four scored seeds
   [0.4, 0.6, 2.5, 0.5] at fraction 1/2 = (2.5+0.6)/2 = 1.55, and median of
   the accepted scores {0.4, 0.5, 0.6} = 0.5. Stage replay over the
   crossover/eq/export pipeline: a complete trace has no failures; a
   missing stage, a duplicate stage, a failed structural check, and a
   failed stage each fail loudly instead of scoring. *)

Off[General::munfl];
scores = {0.4, 0.6, 2.5, 0.5};
sorted = Sort[scores, Greater];
tailMean = N[Mean[Take[sorted, 2]], 16];
medianAccepted = N[Median[{0.4, 0.5, 0.6}], 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.qa03-seed-matrix.v1",
  "family" -> "QA03",
  "wolfram_engine" -> "15.0.0",
  "units" -> "counts, objective units (dB-like residual)",
  "calibration" -> "none required: table arithmetic plus stage contracts",
  "seed_ids" -> {11, 29, 47, 71, 101},
  "seed_status" -> {"pass", "pass", "fail", "pass", "blocked"},
  "seed_scores" -> {0.4, 0.6, 2.5, 0.5, Null},
  "counts_pass_fail_notrun_blocked" -> {3, 1, 0, 1},
  "accepted_fraction" -> N[3/5, 16],
  "failed_fraction" -> N[1/5, 16],
  "tail_fraction" -> 0.5,
  "worst_tail_mean" -> tailMean,
  "median_accepted" -> medianAccepted,
  "stages" -> {"crossover", "eq", "export"},
  "ok_failures" -> 0,
  "missing_stage_failures" -> 1,
  "missing_stage_names" -> {"export"},
  "duplicate_stage_failures" -> 1,
  "failed_check_failures" -> 1,
  "failed_stage_failures" -> 1
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
