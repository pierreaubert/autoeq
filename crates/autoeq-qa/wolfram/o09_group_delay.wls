(* autoeq-qa case autoeq-qa.o09-group-delay.v1 (catalogue family O09).
   Independent evaluation of the documented nonuniform group-delay
   stencil tau_g = -dphi/domega on a wrapped, nonuniformly spaced phase
   grid (NumPy-style unwrap: subtract round(diff/360)*360 per step),
   with gd[1] = gd[2] endpoint copy, the group-delay standard deviation
   (IR duration proxy), least-squares detrended phase deviation against
   a zero target, and the magnitude/phase loss combination. The fixture
   is a ~1 ms bulk delay plus small shape deviations, with one genuine
   wrap between bins 4 and 5 and no aliased (>180 deg true-step) jumps. *)

Off[General::munfl];
freqs = {100., 150., 300., 450., 700.};
wrapped = {-36., -42., -126., -154., 108.};

unwrap[ph_] := Module[{out = N[ph, 16], off = 0.},
  Do[
    off = off - 360. Round[((ph[[i]] + off) - out[[i - 1]])/360.];
    out[[i]] = ph[[i]] + off,
    {i, 2, Length[ph]}];
  out
];
unwrapped = unwrap[wrapped];
gd = Table[
  If[i == 1, 0.,
    -N[(unwrapped[[i]] - unwrapped[[i - 1]]) Degree/(2 Pi (freqs[[i]] - freqs[[i - 1]])), 16]*1000.],
  {i, 1, Length[freqs]}];
gd[[1]] = gd[[2]];
meanGd = N[Total[gd]/Length[gd], 16];
duration = N[Sqrt[Total[(# - meanGd)^2 & /@ gd]/Length[gd]], 16];

(* Phase deviation: unwrap measured and target, remove best-fit
   constant + linear trend of the difference, report RMS residual. *)
target = Table[0., {Length[freqs]}];
diff = unwrapped - target;
n = Length[freqs];
meanF = N[Total[freqs]/n, 16];
meanD = N[Total[diff]/n, 16];
varF = N[Total[(# - meanF)^2 & /@ freqs], 16];
slope = N[Total[Table[(freqs[[i]] - meanF) (diff[[i]] - meanD), {i, 1, n}]]/varF, 16];
intercept = N[meanD - slope meanF, 16];
deviation = N[Sqrt[Total[Table[(diff[[i]] - (intercept + slope freqs[[i]]))^2, {i, 1, n}]]/n], 16];
magPhaseLoss = N[2.0 + 0.25 deviation, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o09-group-delay.v1",
  "family" -> "O09",
  "sample_rate_hz" -> 48000.,
  "grid" -> "nonuniform Hz grid {100,150,300,450,700}",
  "units" -> "Hz, degrees, ms",
  "calibration" -> "phase-only fixture; no level calibration involved",
  "freqs_hz" -> freqs,
  "phase_wrapped_deg" -> wrapped,
  "phase_unwrapped_deg" -> unwrapped,
  "group_delay_ms" -> gd,
  "ir_duration_ms" -> duration,
  "phase_deviation_deg" -> deviation,
  "magnitude_error" -> 2.0,
  "phase_weight" -> 0.25,
  "magnitude_phase_loss" -> magPhaseLoss
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
