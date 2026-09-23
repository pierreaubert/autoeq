(* autoeq-qa case autoeq-qa.o04-bass-boost.v1 (catalogue family O04).
   Independent evaluation of the explicit bass-boost target formulas:
   Linear is a piecewise ramp (0 below start, rising start->peak, falling
   peak->end, 0 at/above end); Harman is maxBoost*Exp[-((f-peak)/width)^2/2]
   with width = (end-start)/2, zero strictly outside [start, end]; a disabled
   config yields zeros. Compared against compute_bass_boost_curve for the
   Linear, Harman and disabled configurations. Units are dB on a declared
   grid; the target is added to the base target curve. *)

Off[General::munfl];
freqs = {10., 20., 40., 60., 130., 200., 250.};
startF = 20.; peakF = 60.; endF = 200.; maxBoost = 4.;

linear[f_] := N[Which[
  f < startF, 0.,
  f < peakF, maxBoost*(f - startF)/(peakF - startF),
  f < endF, maxBoost*(1 - (f - peakF)/(endF - peakF)),
  True, 0.
], 16];
width = N[(endF - startF)/2, 16];
harman[f_] := N[If[f < startF || f > endF, 0., maxBoost*Exp[-0.5*((f - peakF)/width)^2]], 16];

linearVals = linear /@ freqs;
harmanVals = harman /@ freqs;
zeros = Table[0., {Length[freqs]}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o04-bass-boost.v1",
  "family" -> "O04",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "target-curve offset on a declared grid; no sampling involved",
  "grid_hz" -> freqs,
  "units" -> "dB",
  "calibration" -> "additive target offset; no SPL calibration",
  "config" -> <|
    "start_freq_hz" -> startF, "peak_freq_hz" -> peakF,
    "end_freq_hz" -> endF, "max_boost_db" -> maxBoost|>,
  "linear_db" -> linearVals,
  "harman_db" -> harmanVals,
  "disabled_db" -> zeros
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
