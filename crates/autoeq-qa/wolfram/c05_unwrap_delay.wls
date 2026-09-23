(* autoeq-qa case autoeq-qa.c05-unwrap-delay.v1 (catalogue family C05).
   Phase unwrap (degrees): remove 360 deg wraps by subtracting
   360*Round[(p[i]-p[i-1])/360] cumulatively. Delay fit: ordinary
   least-squares slope of excess phase (radians) vs frequency (Hz);
   delay = -slope/(2 Pi) with intercept removed; residual is the
   non-linear remainder. Excess phase itself is total minus minimum. *)

Off[General::munfl];
wrapped = {170., 175., -175., -170., -165.};
unwrapFold[{acc_, off_}, p_] := Module[{wraps = Round[(p - acc)/360.]},
  {p, off - wraps*360.}
];
steps = FoldList[unwrapFold, {wrapped[[1]], 0.}, Rest[wrapped]];
unwrapped = Prepend[#[[1]] + #[[2]] & /@ Rest[steps], wrapped[[1]]];

freqs = Table[100.*i, {i, 1, 10}];
tauMs = 1.;
excess = N[-360.*freqs*(tauMs/1000.), 16];
minpart = {0.5, -1., 1.5, -2., 2.5, -2.5, 3., -3., 0., 1.};
total = excess + minpart;
excessCheck = total - minpart;

n = Length[freqs];
rad = excess*Pi/180.;
sf = Total[freqs]; sp = Total[rad];
sf2 = Total[freqs^2]; sfp = Total[freqs*rad];
den = n*sf2 - sf^2;
slope = (n*sfp - sf*sp)/den;
intercept = (sp - slope*sf)/n;
delayMs = N[-slope/(2*Pi)*1000., 16];
resid = excess - (slope*freqs + intercept)*180./Pi;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c05-unwrap-delay.v1",
  "family" -> "C05",
  "sample_rate_hz" -> "not-applicable",
  "grid_hz" -> freqs,
  "units" -> "degrees-and-ms",
  "calibration" -> "none",
  "wrapped_deg" -> wrapped,
  "unwrapped_deg" -> N[unwrapped, 16],
  "total_phase_deg" -> N[total, 16],
  "min_phase_deg" -> minpart,
  "excess_phase_deg" -> N[excessCheck, 16],
  "freqs_hz" -> freqs,
  "true_delay_ms" -> tauMs,
  "fitted_delay_ms" -> delayMs,
  "residual_deg" -> N[resid, 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
