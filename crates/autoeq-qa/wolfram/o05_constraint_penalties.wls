(* autoeq-qa case autoeq-qa.o05-constraint-penalties.v1 (catalogue family O05).
   Independent evaluation of the optimizer constraint/penalty contract from
   published definitions (direct sums, octave log-ratios, log-frequency
   envelope interpolation), NOT by transliterating the Rust loops:
   - ceiling violation   = Max[0, Max[spl - maxDb]] (non-finite -> +Inf)
   - min-gain violation  = Max over peak filters of
                           If[Abs[g] < 0.1, 0, Max[0, minDb - Abs[g]]]
   - spacing violation   = Max[0, reqOct - MinPairDist],
                           MinPairDist = Min |Log2[fj/fi]| over all pairs
   - crossover violation = Max over adjacent pairs
                           (logXi + sep - logX(i+1)) (single xover -> 0)
   - local-Q projection  = Min[globalMaxQ, log-interp envelope at center]
   - gain projection     = clamp to boost/cut envelope at decoded center
   - penalty term        = weight * violation^2 (squared convention)
   - feasibility         = every signed constraint value <= 0, including an
                           exactly-active (== 0) fixture.
   Frequencies below use the Pk layout [Log10[Hz], Q, gain]. *)

Off[General::munfl];
nonFiniteQ[x_] := MatchQ[x, Infinity | -Infinity | Indeterminate];
ceilingViol2[spl_List, maxDb_] := If[AnyTrue[spl, nonFiniteQ],
  "plusInf", N[Max[0., Max[spl - maxDb]], 16]];

minGainViol[gains_List, minDb_] := N[Max[Table[
  If[Abs[g] < 0.1, 0., Max[0., minDb - Abs[g]]], {g, gains}]], 16];

pairOct[freqs_List] := Min[Flatten[Table[
  Abs[Log[2, freqs[[j]]/freqs[[i]]]], {i, 1, Length[freqs] - 1}, {j, i + 1, Length[freqs]}]]];
spacingViol[logxs_List, req_] := Module[{freqs = N[10^logxs, 16], d},
  d = N[pairOct[freqs], 16];
  {N[Max[0., req - d], 16], N[req - d, 16], d}];

xoverViol[logs_List, sep_] := If[Length[logs] <= 1, 0.,
  N[Max[Table[logs[[i]] + sep - logs[[i + 1]], {i, 1, Length[logs] - 1}]], 16]];

envBound[knots_List, f_] := Module[{n = Length[knots], seg, f0, b0, f1, b1, t},
  If[f <= knots[[1, 1]], Return[N[knots[[1, 2]], 16]]];
  If[f >= knots[[n, 1]], Return[N[knots[[n, 2]], 16]]];
  seg = First[Select[Partition[knots, 2, 1], #[[1, 1]] <= f <= #[[2, 1]] &]];
  {f0, b0} = seg[[1]]; {f1, b1} = seg[[2]];
  t = (Log[f] - Log[f0])/(Log[f1] - Log[f0]);
  N[b0 + t*(b1 - b0), 16]];

(* --- fixtures --- *)
sr = 48000.;
splA = {-1., 2., 4.5, 3.};
maxDb = 3.;
splExact = {-1., 3.};

gainsOk = {5., -3.};
gainsViol = {5., 0.5};
gainsExact = {1., -1.};
gainsRemoved = {0.05, -3.};
minDbReq = 1.;

xsA = {2., 1., 5., 3., 1., -3.};
logxsA = {2., 3.};
logxsClose = {3., 3.2};
logxsExact = {3., N[3 + Log[10, 2], 16]};
reqOct = 1.;

xoverOk = {2.5, 3.};
xoverBad = {3., 2.5};
xoverExact = {2.5, 2.6};
sepReq = 0.1;

qKnots = {{100., 2.}, {1000., 6.}};
centerQ = 500.;
globalQ = 10.;
globalQbinding = 3.;
qLocalCap = envBound[qKnots, centerQ];
qEffLoose = Min[globalQ, qLocalCap];
qEffTight = Min[globalQbinding, qLocalCap];

boostKnots = {{100., 6.}, {1000., 2.}};
cutKnots = {{100., -6.}, {1000., -2.}};
boostAt1000 = envBound[boostKnots, 1000.];
cutAt100 = envBound[cutKnots, 100.];
boostAtMid = envBound[boostKnots, N[10^2.5, 16]];

wCeil = 1.*^4; wMin = 1.*^3; wSp = 1.*^3;
penCeil = wCeil*ceilingViol2[splA, maxDb]^2;
penMin = wMin*minGainViol[gainsViol, minDbReq]^2;
penSp = wSp*spacingViol[logxsClose, reqOct][[1]]^2;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o05-constraint-penalties.v1",
  "family" -> "O05",
  "sample_rate_hz" -> sr,
  "units" -> "dB for gains/violations, octaves for spacing, Log10[Hz] for crossover params, linear Q",
  "calibration" -> "n/a (parameter-level constraint case; no acoustic calibration)",
  "freq_grid_hz" -> "n/a (parameter-level case; centers carried per fixture)",
  "layout" -> "Pk [Log10[Hz], Q, gain] per filter",
  "max_db" -> maxDb,
  "spl_a" -> splA,
  "ceiling_viol_a" -> N[ceilingViol2[splA, maxDb], 16],
  "ceiling_feasible_a" -> False,
  "spl_exact" -> splExact,
  "ceiling_viol_exact" -> N[ceilingViol2[splExact, maxDb], 16],
  "ceiling_feasible_exact" -> True,
  "ceiling_viol_nonfinite" -> ceilingViol2[{-1., Infinity, 3.}, maxDb],
  "min_db_req" -> minDbReq,
  "mingain_viol_ok" -> N[minGainViol[gainsOk, minDbReq], 16],
  "mingain_viol_case" -> N[minGainViol[gainsViol, minDbReq], 16],
  "mingain_viol_exact" -> N[minGainViol[gainsExact, minDbReq], 16],
  "mingain_viol_removed" -> N[minGainViol[gainsRemoved, minDbReq], 16],
  "xs_a" -> xsA,
  "req_oct" -> reqOct,
  "spacing_viol_a" -> spacingViol[logxsA, reqOct][[1]],
  "spacing_fc_a" -> spacingViol[logxsA, reqOct][[2]],
  "spacing_dist_a" -> spacingViol[logxsA, reqOct][[3]],
  "spacing_viol_close" -> spacingViol[logxsClose, reqOct][[1]],
  "spacing_fc_close" -> spacingViol[logxsClose, reqOct][[2]],
  "spacing_dist_close" -> spacingViol[logxsClose, reqOct][[3]],
  "spacing_viol_exact" -> spacingViol[logxsExact, reqOct][[1]],
  "spacing_fc_exact" -> spacingViol[logxsExact, reqOct][[2]],
  "spacing_dist_exact" -> spacingViol[logxsExact, reqOct][[3]],
  "n_drivers" -> 3,
  "min_log_separation" -> sepReq,
  "xover_ok" -> xoverOk,
  "xover_viol_ok" -> xoverViol[xoverOk, sepReq],
  "xover_bad" -> xoverBad,
  "xover_viol_bad" -> xoverViol[xoverBad, sepReq],
  "xover_exact" -> xoverExact,
  "xover_viol_exact" -> xoverViol[xoverExact, sepReq],
  "q_knots" -> qKnots,
  "q_center_hz" -> centerQ,
  "q_before" -> 8.,
  "q_local_cap" -> N[qLocalCap, 16],
  "q_global_loose" -> globalQ,
  "q_after_loose" -> N[qEffLoose, 16],
  "q_bound_by_global_loose" -> False,
  "q_global_tight" -> globalQbinding,
  "q_after_tight" -> N[qEffTight, 16],
  "q_bound_by_global_tight" -> True,
  "boost_knots" -> boostKnots,
  "cut_knots" -> cutKnots,
  "gain_before_boost" -> 5.,
  "gain_center_boost_hz" -> 1000.,
  "gain_after_boost" -> N[boostAt1000, 16],
  "gain_before_cut" -> -8.,
  "gain_center_cut_hz" -> 100.,
  "gain_after_cut" -> N[cutAt100, 16],
  "gain_before_mid" -> 5.,
  "gain_center_mid_hz" -> N[10^2.5, 16],
  "gain_after_mid" -> N[boostAtMid, 16],
  "penalty_w_ceiling" -> wCeil,
  "penalty_ceiling" -> N[penCeil, 16],
  "penalty_w_mingain" -> wMin,
  "penalty_mingain" -> N[penMin, 16],
  "penalty_w_spacing" -> wSp,
  "penalty_spacing" -> N[penSp, 16],
  "penalty_disabled" -> 0.,
  "diag_boost_observed" -> 5.,
  "diag_boost_bound" -> 2.,
  "diag_boost_excess" -> 3.,
  "diag_cut_observed" -> -8.,
  "diag_cut_bound" -> -6.,
  "diag_cut_excess" -> 2.,
  "diag_exact_excess" -> 0.
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
