(* autoeq-qa case autoeq-qa.qa02-heldout-perturb.v1 (catalogue family QA02).
   Declared corpus partition plus explicit perturbation construction.
   (a) Held-out separation: per-scenario held-out seat counts from a fixed
   capture table; a seat that both trains and validates, or a training
   scenario with no held-out seat, is an error, never a silent pass.
   (b) Deterministic perturbation: Clean noise is the identity; Noisy
   noise applies the declared xorshift64 magnitude perturbation
   errDb = centered*rmsDb*Sqrt[3], gain = 10^(errDb/20), phase untouched.
   (c) Fixed-candidate rescore: worst-tail mean of [3,1,2,8] at fraction
   1/2 is (8+3)/2 = 5.5. *)

Off[General::munfl];
xorshift[state_] := Module[{s = BitAnd[state, 2^64 - 1]},
  s = BitAnd[BitXor[s, BitShiftLeft[s, 13]], 2^64 - 1];
  s = BitXor[s, BitShiftRight[s, 7]];
  s = BitAnd[BitXor[s, BitShiftLeft[s, 17]], 2^64 - 1];
  s];
transferRe = {1., 0., -1., 0.};
transferIm = {0., 1., 0., -1.};
rmsDb = 0.75;
seed = 11;
state = seed;
perturbed = Table[
  Module[{u, centered, errDb, mag},
   state = xorshift[state];
   u = N[state/2^64, 16];
   centered = (u - 0.5)*2.;
   errDb = centered*rmsDb*Sqrt[3.];
   mag = N[10^(errDb/20), 16];
   {N[transferRe[[i]]*mag, 16], N[transferIm[[i]]*mag, 16]}],
  {i, 1, 4}];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.qa02-heldout-perturb.v1",
  "family" -> "QA02",
  "wolfram_engine" -> "15.0.0",
  "units" -> "dB, linear complex transfer",
  "calibration" -> "none required: partition logic plus synthetic draws",
  "held_out_counts" -> <|"scA" -> 1, "scB" -> 1|>,
  "leak_is_error" -> True,
  "no_heldout_trains_is_error" -> True,
  "transfer_re" -> transferRe,
  "transfer_im" -> transferIm,
  "rms_db" -> rmsDb,
  "seed" -> seed,
  "clean_is_identity" -> True,
  "perturbed_re_im" -> perturbed,
  "tail_scores" -> {3., 1., 2., 8.},
  "tail_fraction" -> 0.5,
  "worst_tail_mean" -> N[5.5, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
