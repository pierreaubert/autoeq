(* autoeq-qa case autoeq-qa.m04-ingest-adapters.v1 (catalogue family M04).
   Independent evaluation of the measurement-ingestion adapters through
   their public interfaces: (a) log-frequency interpolation of an SPL
   trace with exact geometric-midpoint probes, plus phase interpolation
   across the +/-180-degree branch cut (unwrapped before averaging);
   (b) positive-only clamping (negatives untouched); (c) the
   normalize-and-interpolate display shift (log-frequency-weighted mean
   over 1000-2000 Hz subtracted exactly once) with the piecewise ln(f)
   trapezoid mean evaluated directly. *)

Off[General::munfl];

(* --- (a) log-frequency interpolation --- *)
inF = {100., 1000., 10000.};
inSpl = {0., 10., 0.};
logInterp[f_, f0_, f1_, v0_, v1_] :=
  N[v0 + (v1 - v0)*(Log[f] - Log[f0])/(Log[f1] - Log[f0]), 16];
g1 = N[Sqrt[100.*1000.], 16];
g2 = N[Sqrt[1000.*10000.], 16];
interpF = {100., g1, 1000., g2, 10000.};
interpSpl = {inSpl[[1]], logInterp[g1, 100., 1000., 0., 10.],
  inSpl[[2]], logInterp[g2, 1000., 10000., 10., 0.], inSpl[[3]]};
(* Phase branch cut: 170 deg and -170 deg unwrap to 170/190; midpoint 180. *)
phaseMid = N[(170. + 190.)/2, 16];

(* --- (b) positive-only clamp at +12 dB --- *)
clampIn = {-15., -1., 0., 1., 10., 25.};
clampOut = Map[If[# > 0, Min[#, 12.], #] &, clampIn];

(* --- (c) normalize-and-interpolate shift, band 1000-2000 Hz --- *)
normF = {100., 500., 1000., 1500., 2000., 10000.};
sAt[f_] := If[f <= 1000., logInterp[f, 100., 1000., 0., 10.],
  logInterp[f, 1000., 10000., 10., 0.]];
interpNorm = sAt /@ normF;
trapMean[] := Module[{pts, total = 0., width = 0., a, b, va, vb},
  pts = Select[normF, 1000. <= # <= 2000. &];
  Do[
    a = pts[[k]]; b = pts[[k + 1]];
    va = sAt[a]; vb = sAt[b];
    total += 0.5*(va + vb)*(Log[b] - Log[a]);
    width += (Log[b] - Log[a]);,
  {k, Length[pts] - 1}];
  N[total/width, 16]];
bandMean = trapMean[];
normSpl = N[# - bandMean & /@ interpNorm, 16];
normMeanCheck = N[bandMean - bandMean, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.m04-ingest-adapters.v1",
  "family" -> "M04",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> interpF,
  "units" -> "SPL dB; frequencies Hz; phase degrees",
  "calibration" -> "none (conditioning adapters preserve absolute level except the declared display shift)",
  "grid_note" -> "targets on the shared input grid lines; geometric midpoints never zip unequal grids",
  "input_freqs_hz" -> inF, "input_spl_db" -> inSpl,
  "interp_freqs_hz" -> interpF, "interp_spl_db" -> interpSpl,
  "phase_branch_pair_deg" -> {170., -170.},
  "phase_midpoint_deg" -> phaseMid,
  "clamp_input_db" -> clampIn, "clamp_output_db" -> clampOut,
  "clamp_max_db" -> 12.,
  "norm_freqs_hz" -> normF,
  "norm_interpolated_db" -> interpNorm,
  "norm_band_mean_db" -> bandMean,
  "norm_output_db" -> normSpl,
  "norm_remean_db" -> normMeanCheck
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
