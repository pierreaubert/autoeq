(* autoeq-qa case autoeq-qa.o12-pareto-mixed.v1 (catalogue family O12).
   Independent evaluation of two O12 quantities from their documented
   definitions. (1) Pareto dominance enumeration on a hand-designed
   objective set {loss, filter count} with the convergence tie-break
   (exact tie prefers converged) and non-finite-loss filtering:
   A(10,1,T) and B(5,2,T) are mutually non-dominating; C(20,3,T) and
   F(8,2,T) are dominated by B; D(5,2,F) loses the exact tie to B;
   E(NaN,1,F) is unrankable. Front = {A, B}. (2) Mixed speaker loss
   (0.5 + lwSlope)^2 + (pirOgSlope - pirEqSlope)^2 with least-squares
   slopes against log2(f) over 100..10000 Hz: lw carries an exact
   +1 dB/octave tilt (80 + Log2[f/1000]), pir is flat, PEQ is zero, so
   the mixed loss is exactly 2.25. *)

Off[General::munfl];
losses = {10., 5., 20., 5., Indeterminate, 8.};
counts = {1, 2, 3, 2, 1, 2};
converged = {True, True, True, False, False, True};
rankable[i_] := NumericQ[losses[[i]]] && losses[[i]] != Infinity && losses[[i]] != -Infinity;
dominates[j_, i_] := rankable[j] && rankable[i] && j != i && And[
  losses[[j]] <= losses[[i]], counts[[j]] <= counts[[i]],
  Or[losses[[j]] < losses[[i]], counts[[j]] < counts[[i]],
    And[converged[[j]], ! converged[[i]],
      losses[[j]] == losses[[i]], counts[[j]] == counts[[i]]]]];
front = Select[Range[6], rankable[#] && ! Or @@ Table[dominates[j, #], {j, 1, 6}] &];

freqs = {100., 200., 400., 800., 1600., 3200., 6400., 10000.};
lw = Table[N[80 + Log2[f/1000.], 16], {f, freqs}];
pir = Table[80., {8}];
slopeOf[y_] := Module[{xs, n2, sx, sy, sxy, sx2},
  xs = N[Log2[#] & /@ freqs, 16];
  n2 = Length[xs];
  sx = Total[xs]; sy = Total[y];
  sxy = Total[Table[xs[[i]] y[[i]], {i, 1, n2}]];
  sx2 = Total[xs^2];
  N[(sxy - sx sy/n2)/(sx2 - sx^2/n2), 16]];
lwSlope = slopeOf[lw];
pirSlope = slopeOf[pir];
mixed = N[(0.5 + lwSlope)^2 + (pirSlope - pirSlope)^2, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o12-pareto-mixed.v1",
  "family" -> "O12",
  "sample_rate_hz" -> 48000.,
  "grid" -> "100..10000 Hz 8-point grid for slopes; abstract 2-objective set for Pareto",
  "units" -> "dB SPL, Hz, dB/octave; Pareto losses dimensionless with filter counts",
  "calibration" -> "absolute spin levels (flat 80 dB pir, tilted lw); no normalization in mixed loss",
  "pareto_losses" -> {10., 5., 20., 5., "NaN", 8.},
  "pareto_counts" -> counts,
  "pareto_converged" -> converged,
  "front_indices" -> (front - 1),
  "freqs_hz" -> freqs,
  "lw_db" -> lw,
  "pir_db" -> pir,
  "lw_slope_db_per_oct" -> lwSlope,
  "pir_slope_db_per_oct" -> pirSlope,
  "mixed_loss" -> mixed
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
