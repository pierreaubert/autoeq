(* autoeq-qa case autoeq-qa.ra07-spatial-stats.v1 (catalogue family RA07).
   Independent spatial-statistics derivation from the published definitions,
   not a transliteration of the Rust loops:
   - weighted power average: 10 log10(sum_k w_k 10^(L_k/10)) per bin;
   - weighted dB spread: sqrt(sum w (L-mean)^2 / (1-sum w^2)) with the
     1/N denominator floor (weights here give 1-sum w^2 = 0.62 > 1/3,
     so the floor is inactive and the oracle pins the unbiased branch);
   - correction-depth mask: min+(1-min)/(1+Exp[-((thr-s)/wid)]) with
     thr = 3 dB, wid = 2 dB, min = 0.1 and NO log-frequency smoothing
     (mask_smoothing_octaves = 0), so the closed form is exact;
   - bootstrap band + CVaR over B = 6 EXPLICIT resample index lists
     (no RNG matching): resampled weights are the source weights at the
     drawn indices renormalized; per-bin percentiles use linear
     interpolation on sorted values at q = 0.05/0.5/0.95 (alpha = 0.10);
     per-bin sample std uses /(B-1); CVaR(0.25) is the mean of the
     ceil(0.25*B) = 2 lowest resampled means per bin.
   Compared against try_rms_average_weighted / try_spatial_std_dev_weighted /
   correction_depth_mask, manual resampling through try_rms_average_weighted
   over the golden index lists, and the deterministic single-curve
   bootstrap_band collapse. Levels are dB SPL; no time domain, so no
   sample rate applies (recorded as Null). *)

Off[General::munfl];
grid = {63., 125., 250., 500.};
seats = {{70., 72., 74., 76.}, {73., 71., 75., 76.}, {68., 74., 72., 79.}};
w = {0.5, 0.3, 0.2};
nb = Length[grid]; ns = Length[seats];

powAvg = Table[N[10*Log10[Sum[w[[k]]*10^(seats[[k, b]]/10), {k, 1, ns}]], 16], {b, 1, nb}];

wmean[b_] := N[Sum[w[[k]]*seats[[k, b]], {k, 1, ns}], 16];
denom = N[1 - Sum[w[[k]]^2, {k, 1, ns}], 16];
spread = Table[
   Module[{m = wmean[b], v},
    v = N[Sum[w[[k]]*(seats[[k, b]] - m)^2, {k, 1, ns}], 16];
    N[Sqrt[v/Max[denom, 1/ns]], 16]], {b, 1, nb}];

mask = Table[N[0.1 + 0.9/(1 + Exp[-((3.0 - spread[[b]])/2.0)]), 16], {b, 1, nb}];

idx = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {1, 2, 3}, {1, 1, 2}, {2, 3, 3}};
nbRes = Length[idx];
resMean = Table[
   Module[{draw = idx[[l]], rw, s},
    rw = (w[[#]] & /@ draw); s = N[Total[rw], 16]; rw = N[rw/s, 16];
    Table[N[10*Log10[Sum[rw[[k]]*10^(seats[[draw[[k]], b]]/10), {k, 1, 3}]], 16],
     {b, 1, nb}]], {l, 1, nbRes}];

pct[vals_, q_] := Module[{s = Sort[N[vals, 16]], n = Length[vals], pos, lo, hi, t},
   pos = N[q*(n - 1) + 1, 16]; lo = Floor[pos]; hi = Ceiling[pos];
   If[lo == hi, s[[lo]], (t = pos - lo; s[[lo]]*(1 - t) + s[[hi]]*t)]];
lower = Table[N[pct[resMean[[All, b]], 0.05], 16], {b, 1, nb}];
median = Table[N[pct[resMean[[All, b]], 0.5], 16], {b, 1, nb}];
upper = Table[N[pct[resMean[[All, b]], 0.95], 16], {b, 1, nb}];
bandStd = Table[
   Module[{v = N[resMean[[All, b]], 16], m},
    m = Total[v]/Length[v];
    N[Sqrt[Total[((# - m)^2 & /@ v)]/(Length[v] - 1)], 16]], {b, 1, nb}];
cvar = Table[
   Module[{s = Sort[N[resMean[[All, b]], 16]]}, N[Total[Take[s, 2]]/2, 16]],
   {b, 1, nb}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra07-spatial-stats.v1",
  "family" -> "RA07",
  "sample_rate_hz" -> Null,
  "grid_hz" -> grid,
  "grid_units" -> "Hz",
  "sample_units" -> "dB SPL",
  "calibration" -> "analytic plant; synthetic seat levels, no measurement",
  "seat_spl_db" -> seats,
  "weights" -> w,
  "power_average_db" -> powAvg,
  "spread_denominator" -> denom,
  "spread_db" -> spread,
  "mask_threshold_db" -> 3.0,
  "mask_transition_db" -> 2.0,
  "mask_min_depth" -> 0.1,
  "mask_smoothing_octaves" -> 0.0,
  "correction_mask" -> mask,
  "bootstrap_alpha" -> 0.10,
  "resample_index_lists_1based" -> idx,
  "resampled_means_db" -> resMean,
  "band_lower_db" -> lower,
  "band_median_db" -> median,
  "band_upper_db" -> upper,
  "band_std_db" -> bandStd,
  "cvar_level" -> 0.25,
  "cvar_lowest_count" -> 2,
  "cvar_db" -> cvar,
  "single_seat_spl_db" -> seats[[1]]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
