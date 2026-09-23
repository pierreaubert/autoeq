(* autoeq-qa case autoeq-qa.m03-mic-clock.v1 (catalogue family M03).
   Independent evaluation of (a) microphone calibration correction with
   the declared SUBTRACTION sign (corrected = measured - mic, linear
   interpolation of the unwrapped calibration, coherence multiplied);
   (b) the affine clock-marker least-squares fit
   observed = reference*(1+rate_ppm/1e6) + offset with closed-form
   slope/intercept, residuals, RMS residual, and accumulated drift;
   (c) the identity time-map (zero drift/offset reproduces the samples).
   Direct sums only. *)

Off[General::munfl];

(* --- (a) mic calibration --- *)
calF = {100., 1000., 10000.};
calMag = {1., 2., 3.};
calPhase = {10., 20., 30.};
calCoh = {1., 0.9, 0.8};
curveF = {100., 500., 1000., 5000., 10000.};
curveSpl = {80., 81., 82., 81., 80.};
curvePhase = {5., 5., 5., 5., 5.};
curveCoh = {1., 1., 1., 1., 1.};

interpAt[f_] := Module[{i},
  If[f <= calF[[1]], Return[{calMag[[1]], calPhase[[1]], calCoh[[1]]}]];
  If[f >= calF[[-1]], Return[{calMag[[-1]], calPhase[[-1]], calCoh[[-1]]}]];
  i = First[Flatten[Position[calF, x_ /; x >= f]]];
  Module[{x0 = calF[[i - 1]], x1 = calF[[i]], t},
    t = N[(f - x0)/(x1 - x0), 16];
    {(1 - t)*calMag[[i - 1]] + t*calMag[[i]],
     (1 - t)*calPhase[[i - 1]] + t*calPhase[[i]],
     (1 - t)*calCoh[[i - 1]] + t*calCoh[[i]]}]];

cals = interpAt /@ curveF;
corrSpl = Table[N[curveSpl[[i]] - cals[[i, 1]], 16], {i, Length[curveF]}];
corrPhase = Table[N[curvePhase[[i]] - cals[[i, 2]], 16], {i, Length[curveF]}];
corrCoh = Table[N[curveCoh[[i]]*cals[[i, 3]], 16], {i, Length[curveF]}];

(* --- (b) affine clock fit, 50 ppm + 1 ms offset --- *)
ref = {0., 10., 20.};
obs = {0.001, 10.*1.00005 + 0.001, 20.*1.00005 + 0.001};
n = Length[ref];
mref = Mean[ref]; mobs = Mean[obs];
cov = Sum[(ref[[i]] - mref)*(obs[[i]] - mobs), {i, n}];
var = Sum[(ref[[i]] - mref)^2, {i, n}];
slope = N[cov/var, 16];
intercept = N[mobs - slope*mref, 16];
ratePpm = N[(slope - 1.)*1.*10^6, 16];
resid = Table[N[obs[[i]] - (slope*ref[[i]] + intercept), 16], {i, n}];
rms = N[Sqrt[Mean[#^2 & /@ resid]], 16];
accum20 = N[ratePpm*1.*10^-6*20., 16];

(* --- (c) identity time-map --- *)
idSamples = {1., 2., 3., 4.};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.m03-mic-clock.v1",
  "family" -> "M03",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> curveF,
  "units" -> "SPL dB; calibration/phase degrees; coherence dimensionless; clock seconds and ppm",
  "calibration" -> "mic cal nodes 100/1000/10000 Hz applied with minus sign; clock markers s vs s",
  "grid_note" -> "curve grid inside calibration support; no extrapolation",
  "cal_freqs_hz" -> calF, "cal_mag_db" -> calMag,
  "cal_phase_deg" -> calPhase, "cal_coherence" -> calCoh,
  "curve_spl_db" -> curveSpl, "curve_phase_deg" -> curvePhase,
  "curve_coherence" -> curveCoh,
  "corrected_spl_db" -> corrSpl,
  "corrected_phase_deg" -> corrPhase,
  "corrected_coherence" -> corrCoh,
  "markers_reference_s" -> ref, "markers_observed_s" -> obs,
  "fit_offset_s" -> intercept, "fit_rate_ppm" -> ratePpm,
  "fit_residuals_s" -> resid, "fit_rms_residual_s" -> rms,
  "fit_n_markers" -> n,
  "accumulated_offset_20s" -> accum20,
  "identity_samples" -> idSamples,
  "identity_corrected" -> idSamples
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
