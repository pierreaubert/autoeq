(* autoeq-qa case autoeq-qa.c04-gaussian-smooth.v1 (catalogue family C04).
   Index-domain Gaussian smoothing: sigma is in SAMPLES (not octaves).
   Kernel w[k] = Exp[-0.5 (k/sigma)^2], radius Ceiling[3 sigma], normalized
   over the full kernel; at the edges the output renormalizes over the
   in-range weights. sigma <= 0 is the identity. Direct double sum. *)

Off[General::munfl];
signal = {3., 1., 4., 1., 5., 9., 2.};
sigma = 1.;
radius = Ceiling[3*sigma];
raw = Table[N[Exp[-0.5*(k/sigma)^2], 16], {k, -radius, radius}];
kernel = raw/Total[raw];
smoothIdx[sig_, i_] := Module[{num = 0., den = 0., j},
  Do[
    j = i + k;
    If[1 <= j <= Length[sig],
      num += sig[[j]]*kernel[[k + radius + 1]];
      den += kernel[[k + radius + 1]];
    ],
    {k, -radius, radius}
  ];
  If[den > 0, num/den, sig[[i]]]
];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c04-gaussian-smooth.v1",
  "family" -> "C04",
  "sample_rate_hz" -> "not-applicable",
  "grid_hz" -> "index-domain-samples",
  "units" -> "same-as-input",
  "calibration" -> "none",
  "sigma_samples" -> sigma,
  "kernel_radius" -> radius,
  "kernel" -> kernel,
  "input" -> signal,
  "smoothed" -> (N[smoothIdx[signal, #], 16] & /@ Range[Length[signal]]),
  "identity_sigmas" -> {0., -1.},
  "identity_expected" -> signal
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
