(* autoeq-qa case autoeq-qa.o13-backend-objectives.v1 (catalogue family O13).
   Independent reference for optimizer-adapter agreement:
   - bounded quadratic f[x] = (x0 - 1/4)^2 + (x1 + 1/2)^2 on
     [-2,2]^2, evaluated at fixed candidates by direct substitution;
     analytic minimizer {1/4, -1/2} with minimum 0;
   - feasibility = box membership (boundary counts as feasible);
   - tiny 1-D filter-fit landscape g[x] = (x + 3)^2 on [-12,12] plus a
     quadratic-penalty variant h[x] = (x+3)^2 + w*Max[0,x-2]^2, w = 10^4,
     mirroring the optimizer's squared-penalty convention;
   - backend registry expectations (canonical names, alias map).
   No identical-trajectory demand: stochastic search is checked only by an
   objective gap plus exact feasibility and reported-value consistency in
   the Rust test. *)

Off[General::munfl];
f[c_] := N[(c[[1]] - 1/4)^2 + (c[[2]] + 1/2)^2, 16];
g[x_] := N[(x + 3)^2, 16];
h[x_] := N[(x + 3)^2 + 1.*^4*Max[0., x - 2]^2, 16];
feasible2[c_, lo_, hi_] := lo[[1]] <= c[[1]] <= hi[[1]] && lo[[2]] <= c[[2]] <= hi[[2]];
feasible1[x_, lo_, hi_] := lo <= x <= hi;

lo2 = {-2., -2.}; hi2 = {2., 2.};
candidates = {{1.5, 1.5}, {0.25, -0.5}, {0., 0.}, {-2., 2.}};
outside = {3., 0.};
onbound = {-2., -2.};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o13-backend-objectives.v1",
  "family" -> "O13",
  "sample_rate_hz" -> 48000.,
  "units" -> "objective units (squared parameter distance); parameters unitless box variables",
  "calibration" -> "n/a (adapter-level quadratic/fit case; no acoustic calibration)",
  "freq_grid_hz" -> "n/a (parameter-level case)",
  "bounds_2d" -> {lo2, hi2},
  "initial_2d" -> {1.5, 1.5},
  "candidates_2d" -> candidates,
  "candidate_values_2d" -> N[f /@ candidates, 16],
  "outside_2d" -> outside,
  "outside_feasible" -> False,
  "onbound_2d" -> onbound,
  "onbound_feasible" -> True,
  "onbound_value" -> N[f[onbound], 16],
  "analytic_minimizer" -> {0.25, -0.5},
  "analytic_minimum" -> 0.,
  "optimizer_gap_tolerance" -> 1.*^-2,
  "reported_value_tolerance" -> 1.*^-9,
  "scalar_backends" -> {"autoeq:cmaes", "autoeq:de", "autoeq:cobyla", "autoeq:isres"},
  "alias_map" -> {{"cma-es", "autoeq:cmaes"}, {"cobyla", "autoeq:cobyla"},
    {"de", "autoeq:de"}, {"isres", "autoeq:isres"}},
  "unknown_backend" -> "autoeq:nope",
  "scalar_unsupported_backend" -> "autoeq:nsga2",
  "fit_bounds_1d" -> {-12., 12.},
  "fit_candidates" -> {0., -3., 6.},
  "fit_values" -> N[g /@ {0., -3., 6.}, 16],
  "fit_penalty_weight" -> 1.*^4,
  "fit_penalty_candidates" -> {4., 0., 2.},
  "fit_penalty_values" -> N[h /@ {4., 0., 2.}, 16],
  "fit_minimizer" -> -3.,
  "fit_minimum" -> 0.
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
