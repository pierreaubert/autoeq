(* autoeq-qa case autoeq-qa.rw07-scorecard-pruning.v1 (catalogue family RW07).
   Independent recomputation of the pruning-matrix audit: given the row
   table (mode/class/variant/conditions/outcome), the audit counts ran
   rows and lists stale/skipped paths with reasons. Rules re-derived
   here from the workflow contract, never by calling the Rust audit:
   a row declaring no conditions is stale ("row declares no conditions")
   regardless of its outcome; otherwise Ran counts, Stale/Skipped keep
   "describe: reason". Catches miscounting, dropped stale paths, and
   empty-condition rows passing as ran. *)

Off[General::munfl];
rows = {
  <|"mode" -> "Ordinary", "class" -> "Iir", "variant" -> "Unrouted",
    "conditions" -> {"c1", "c2"}, "outcome" -> "Ran", "reason" -> ""|>,
  <|"mode" -> "Adaptive", "class" -> "Fir", "variant" -> "Routed",
    "conditions" -> {}, "outcome" -> "Ran", "reason" -> ""|>,
  <|"mode" -> "Pareto", "class" -> "Hybrid", "variant" -> "Routed",
    "conditions" -> {"c3"}, "outcome" -> "Stale",
    "reason" -> "evidence predates chain v7"|>,
  <|"mode" -> "Refinement", "class" -> "Iir", "variant" -> "Unrouted",
    "conditions" -> {"c4"}, "outcome" -> "Skipped",
    "reason" -> "fir-only lane"|>
};

describe[row_] := StringJoin[row["mode"], "/", row["class"], "/",
  row["variant"], ":", StringRiffle[row["conditions"], "+"]];

ran = 0; stale = {}; skipped = {};
Do[
  row = rows[[k]];
  If[Length[row["conditions"]] == 0,
    AppendTo[stale, StringJoin[describe[row], ": row declares no conditions"]];,
    Switch[row["outcome"],
      "Ran", ran++;,
      "Stale", AppendTo[stale, StringJoin[describe[row], ": ", row["reason"]]];,
      "Skipped", AppendTo[skipped, StringJoin[describe[row], ": ", row["reason"]]];,
      _, AppendTo[stale, StringJoin[describe[row], ": unknown outcome"]];
    ];
  ];,
  {k, Length[rows]}
];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw07-scorecard-pruning.v1",
  "family" -> "RW07",
  "sample_rate_hz" -> 48000.,
  "grid" -> "n/a: counting audit over 4 matrix rows, no frequency grid",
  "units" -> "counts and identity strings",
  "calibration" -> "synthetic pruning matrix v1",
  "audit_version" -> "workflow-pruning-audit-v1",
  "rows" -> rows,
  "expected_ran" -> ran,
  "expected_stale" -> stale,
  "expected_skipped" -> skipped
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
