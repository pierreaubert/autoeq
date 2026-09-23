(* autoeq-qa case autoeq-qa.ac01-cli-spacing-score.v1 (catalogue family AC01).
   CLI scoring and constraints through the kernels the CLI wrappers
   delegate to: adjacent octave spacings of sorted Peak centers
   (spacing = log2(f[i+1]/f[i]), minimum checked against 1.0 oct) and
   the Olive/Welti/McMullin headphone preference score
   114.49 - 12.62*SD - 15.52*|slope| with population SD and LS slope of
   deviation vs log2(f) over [50, 10000] Hz. Also the zero-free EQ
   trace for the same filters (RBJ direct SOS sums, dB). Sample rate
   48000 Hz; grids explicit; units dB/octave, dB SPL relative, score
   points; calibration relative. *)

Off[General::munfl];
sr = 48000.;
filterFreqs = {400., 1600., 100.};
filterQ = {1., 1., 1.};
filterGains = {2., 0., -1.};
minSpacingOct = 1.;
sorted = Sort[filterFreqs];
spacings = N[Log[2, #[[2]]/#[[1]]] & /@ Partition[sorted, 2, 1], 16];
minSpacing = Min[spacings];
pass = minSpacing >= minSpacingOct;

grid = {100., 250., 500., 1000., 2000., 4000., 8000., 16000.};
peakSpl[fc_, qq_, gainDb_, f_] := Module[{a, w0, sn, cw, al, bb0, bb1, bb2, aa0, aa1, aa2, z, h},
  a = N[10^(gainDb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 qq);
  bb0 = 1 + al*a; bb1 = -2*cw; bb2 = 1 - al*a;
  aa0 = 1 + al/a; aa1 = -2*cw; aa2 = 1 - al/a;
  z = Exp[-I N[2 Pi f/sr, 16]];
  h = (bb0 + bb1*z + bb2*z^2)/(aa0 + aa1*z + aa2*z^2);
  20*Log[10, Abs[h]]
];
eq = Table[Total[Table[peakSpl[filterFreqs[[k]], filterQ[[k]], filterGains[[k]], grid[[i]]], {k, 3}]], {i, Length[grid]}];
eq = N[eq, 16];

hfreqs = {50., 100., 1000., 5000., 10000.};
dev = {1., 2., 3., 4., 5.};
mean = Mean[dev];
sd = N[Sqrt[Mean[(# - mean)^2 & /@ dev]], 16];
xx = N[Log[2, #] & /@ hfreqs, 16];
nn = Length[xx];
sxx = Total[xx]; syy = Total[dev]; sxxy = Total[xx*dev]; sxx2 = Total[xx^2];
slope = (nn*sxxy - sxx*syy)/(nn*sxx2 - sxx^2);
score = N[114.49 - 12.62*sd - 15.52*Abs[slope], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ac01-cli-spacing-score.v1",
  "family" -> "AC01",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "units" -> "dB SPL relative; spacings in octaves; score points",
  "calibration" -> "relative; fixed synthetic deviation",
  "filter_freqs_hz" -> filterFreqs,
  "filter_q" -> filterQ,
  "filter_gains_db" -> filterGains,
  "min_spacing_oct" -> minSpacingOct,
  "sorted_freqs_hz" -> sorted,
  "adjacent_spacings_oct" -> spacings,
  "min_spacing_oct_value" -> N[minSpacing, 16],
  "spacing_pass" -> pass,
  "eq_response_db" -> eq,
  "headphone_freqs_hz" -> hfreqs,
  "headphone_deviation_db" -> dev,
  "headphone_sd_db" -> sd,
  "headphone_slope_db_per_oct" -> N[slope, 16],
  "headphone_score" -> score
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
