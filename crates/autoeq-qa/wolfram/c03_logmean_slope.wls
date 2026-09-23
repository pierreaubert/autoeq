(* autoeq-qa case autoeq-qa.c03-logmean-slope.v1 (catalogue family C03).
   Independent log-frequency mean of a piecewise-linear dB response
   (exact trapezoid integration in x = Log f, clipped to the requested
   band) and independent log2-frequency least-squares slope (normal
   equations in x = Log2 f, dB per octave). Compared against
   autoeq_core::curve_transforms::mean_over_log_frequency and
   regression_slope_per_octave_in_range. *)

Off[General::munfl];
mfreqs = {100., 200., 400.};
mspl = {0., 10., 0.};

logMean[freqs_, vals_, lo_, hi_] := Module[
  {xl = N[Log[Max[lo, freqs[[1]]]], 16],
   xu = N[Log[Min[hi, freqs[[-1]]]], 16],
   integral = 0., width = 0., n = Length[freqs], i,
   x0, x1, v0, v1, a, b, vAt},
  For[i = 1, i < n, i++,
    x0 = N[Log[freqs[[i]]], 16]; x1 = N[Log[freqs[[i + 1]]], 16];
    v0 = vals[[i]]; v1 = vals[[i + 1]];
    a = Max[x0, xl]; b = Min[x1, xu];
    If[b > a,
      vAt[x_] := v0 + (v1 - v0)*(x - x0)/(x1 - x0);
      integral += 0.5*(vAt[a] + vAt[b])*(b - a);
      width += b - a;
    ];
  ];
  N[integral/width, 16]
];
meanFull = logMean[mfreqs, mspl, 100., 400.];
meanClip = logMean[mfreqs, mspl, 150., 300.];

sgrid = {100., 200., 400., 800.};
tilt = (N[80. - 6.*Log[2, #/100.], 16] & /@ sgrid);
flat = {5., 5., 5., 5.};
slope[vals_] := Module[{x = (N[Log[2, #], 16] & /@ sgrid), n = Length[sgrid],
    sx, sy, sxx, sxy, cov, var},
  sx = Total[x]; sy = Total[vals];
  sxx = Total[x^2]; sxy = Total[x*vals];
  cov = sxy - sx*sy/n; var = sxx - sx*sx/n;
  N[cov/var, 16]
];
slopeTilt = slope[tilt];
slopeFlat = slope[flat];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c03-logmean-slope.v1",
  "family" -> "C03",
  "sample_rate_hz" -> Null,
  "units" -> <|"frequency" -> "Hz", "spl" -> "dB",
    "mean" -> "dB", "slope" -> "dB-per-octave"|>,
  "calibration" -> "none-applies-relative-curve-transforms",
  "mean_freqs_hz" -> mfreqs,
  "mean_spl_db" -> mspl,
  "mean_band_full_hz" -> {100., 400.},
  "mean_full_db" -> meanFull,
  "mean_band_clipped_hz" -> {150., 300.},
  "mean_clipped_db" -> meanClip,
  "slope_freqs_hz" -> sgrid,
  "slope_tilt_spl_db" -> tilt,
  "slope_tilt_db_per_octave" -> slopeTilt,
  "slope_flat_spl_db" -> flat,
  "slope_flat_db_per_octave" -> slopeFlat
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
