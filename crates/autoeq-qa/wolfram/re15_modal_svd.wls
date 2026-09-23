(* autoeq-qa case autoeq-qa.re15-modal-svd.v1 (catalogue family RE15).
   High-precision SVD of a small complex response matrix, derived with
   the engine eigensolver (never transliterated Rust code). Compares
   singular values, retained energy of the rank-1 truncation, the
   rank-1 subspace projector (invariant under eigenvector sign/rotation)
   and the projection residual. *)

Off[General::munfl];
h = N[{{1 + 0.5 I, 0.3 - 0.2 I}, {-0.4 + 0.1 I, 0.8 - 0.3 I}}, 16];
sv = N[SingularValueList[h, Tolerance -> 0], 16];
total = sv[[1]]^2 + sv[[2]]^2;
energy1 = N[sv[[1]]^2/total, 16];
{uu, ss, vv} = N[SingularValueDecomposition[h], 16];
u1 = uu[[All, 1]];
proj = Table[N[u1[[i]] Conjugate[u1[[j]]], 16], {i, 2}, {j, 2}];
resid = N[Norm[h - proj . h, "Frobenius"], 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re15-modal-svd.v1",
  "family" -> "RE15",
  "sample_rate_hz" -> 48000.,
  "grid" -> "single 2x2 complex response matrix (no frequency grid)",
  "units" -> <|"matrix" -> "linear pressure", "energy" -> "fraction"|>,
  "calibration" -> "synthetic",
  "matrix_re_im" -> (complexPair[#] & /@ Flatten[h]),
  "singular_values" -> sv,
  "rank1_retained_energy" -> energy1,
  "rank1_projector_re_im" -> (complexPair[#] & /@ Flatten[proj]),
  "projection_residual_frobenius" -> resid
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
