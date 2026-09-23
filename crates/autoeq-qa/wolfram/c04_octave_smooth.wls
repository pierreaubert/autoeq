(* autoeq-qa case autoeq-qa.c04-octave-smooth.v1 (catalogue family C04).
   1/1-octave smoothing as the log-frequency mean of the piecewise-linear
   interpolant: for each center f, window [f/2^(1/2), f*2^(1/2)] clamped to
   the data support, mean = Integral[v d ln f]/Integral[d ln f] over the
   clamped window with v affine in ln f per segment. Direct per-window
   summation here (independent of the Rust prefix-integral path). *)

Off[General::munfl];
knots = {50., 100., 200., 400., 800., 1600., 3200., 6400., 12800.};
values = {80., 81., 86., 84., 80., 79., 76., 76., 77.};
bpo = 1.;
half = 2.^(1./(2.*bpo));

xs = Log[knots];
segVal[i_, t_] := values[[i]] + (values[[i + 1]] - values[[i]])*(t - xs[[i]])/(xs[[i + 1]] - xs[[i]]);
winMean[f_] := Module[{lo, hi, a, b, num = 0., den = 0., s0, s1, u0, u1},
  lo = Max[Log[f/half], xs[[1]]];
  hi = Min[Log[f*half], xs[[-1]]];
  Do[
    s0 = xs[[i]]; s1 = xs[[i + 1]];
    u0 = Max[s0, lo]; u1 = Min[s1, hi];
    If[u1 > u0,
      num += 0.5*(segVal[i, u0] + segVal[i, u1])*(u1 - u0);
      den += (u1 - u0);
    ],
    {i, 1, Length[xs] - 1}
  ];
  If[den > 0, num/den, segVal[1, lo]]
];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c04-octave-smooth.v1",
  "family" -> "C04",
  "sample_rate_hz" -> "not-applicable",
  "grid_hz" -> knots,
  "units" -> "dB-SPL",
  "calibration" -> "none",
  "window" -> "one-octave-log-frequency-mean-half-window-sqrt2-edge-clamped",
  "bands_per_octave" -> bpo,
  "knot_freqs_hz" -> knots,
  "knot_spl_db" -> values,
  "smoothed_spl_db" -> (N[winMean[#], 16] & /@ knots)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
