(* autoeq-qa case autoeq-qa.py03-roon-correlation.v1 (catalogue family PY03).
   Independent evaluation of the Roon verifier inter-channel delay
   (scripts/roon-qa/verify_capture.py relative_delay_samples): full linear
   cross-correlation of left/right, delay = ArgMax|corr| - (len(right)-1).
   Right channel is the left pulse train delayed by 2 samples. *)

Off[General::munfl];
left = {1., 0., -1., 0., 1., 0., -1., 0.};
right = {0., 0., 1., 0., -1., 0., 1., 0.};
n = Length[left]; m = Length[right];
corr[k_] := Sum[left[[i]]*right[[i - k]], {i, Max[1, 1 + k], Min[n, m + k]}];
lags = Range[-(m - 1), n - 1];
vals = corr[#] & /@ lags;
absvals = Abs[#] & /@ vals;
best = First[Flatten[Position[absvals, Max[absvals]]]];
delay = N[lags[[best]], 16];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.py03-roon-correlation.v1",
  "family" -> "PY03",
  "sample_rate_hz" -> 48000.,
  "grid" -> "sample index, 8-sample synthetic PCM",
  "units" -> "integer samples",
  "calibration" -> "full linear correlation, ArgMax|.|, minus (len-1)",
  "left" -> left,
  "right" -> right,
  "expected_delay_samples" -> delay
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
