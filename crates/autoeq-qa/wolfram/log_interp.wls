(* autoeq-qa case autoeq-qa.log-interp.v1 (catalogue family C03).
   Piecewise-affine interpolation of SPL on the Log-frequency axis with
   slope extrapolation outside the knot range: below the first knot the
   first-segment slope extends, above the last knot the last-segment
   slope extends. Exact knots reproduce their values. Compared against
   autoeq_core::curve_transforms::interpolate_log_space on an
   all-positive grid (plain Log axis, no DC pivot). *)

Off[General::munfl];
knots = {20., 100., 1000., 10000., 20000.};
values = {80., 82., 85., 83., 80.};
outputs = {10., 20., 50., 200., 500., 2000., 8000., 15000., 20000., 25000.};

x = Log[knots];
slope[i_] := (values[[i + 1]] - values[[i]])/(x[[i + 1]] - x[[i]]);
interp[f_] := Module[{t = Log[f], pos},
  pos = LengthWhile[x, # < t &];
  Which[
    pos == 0, values[[1]] + slope[1]*(t - x[[1]]),
    pos >= Length[x], values[[-1]] + slope[Length[x] - 1]*(t - x[[-1]]),
    True, values[[pos]] + slope[pos]*(t - x[[pos]])
  ]
];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.log-interp.v1",
  "family" -> "C03",
  "axis" -> "log-frequency-piecewise-affine-slope-extrapolation",
  "knot_freqs_hz" -> knots,
  "knot_spl_db" -> values,
  "output_freqs_hz" -> outputs,
  "interp_spl_db" -> (N[interp[#], 16] & /@ outputs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
