(* autoeq-qa case autoeq-qa.ra02-probe-delay.v1 (catalogue family RA02).
   Independent direct-sum cross-correlation oracle: a 64-sample seeded
   binary-phase probe at 8 kHz, recorded with an exact integer delay
   and attenuation plus trailing silence. The oracle
   evaluates R[k] = Sum probe[n]*recorded[n+k] directly in the time
   domain (no FFT, no Hilbert envelope) and reports the argmax lag,
   arrival time, and gain from the autocorrelation peak ratio.
   Catches sample/ms conversion slips and 10-vs-20 dB gain errors. *)

Off[General::munfl];
sr = 8000.;
n = 64;
seed = 42;
delay = 37;
atten = 0.5;
tail = 64;
SeedRandom[seed];
probe = N[RandomChoice[{-1., 1.}, n], 16];
recorded = Join[Table[0., {delay}], atten*probe, Table[0., {tail}]];
m = Length[recorded];
xc = Table[Sum[probe[[i]]*recorded[[i + k]], {i, 1, n}], {k, 0, m - n}];
kmax = Ordering[xc, -1][[1]] - 1;
auto = Sum[probe[[i]]^2, {i, 1, n}];
gainLin = N[xc[[kmax + 1]]/auto, 16];
gainDb = N[20*Log10[gainLin], 16];
arrivalMs = N[kmax*1000./sr, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra02-probe-delay.v1",
  "family" -> "RA02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> "time-domain samples at 8000 Hz",
  "units" -> "samples; ms; dB (20log10 pressure)",
  "calibration" -> "digital full-scale, unity probe",
  "timing_origin" -> "probe onset at sample 0",
  "probe_seed" -> seed,
  "probe_kind" -> "binary-phase +-1",
  "probe" -> N[probe, 16],
  "recorded" -> N[recorded, 16],
  "delay_samples" -> delay,
  "attenuation_linear" -> atten,
  "arrival_samples" -> kmax,
  "arrival_ms" -> arrivalMs,
  "gain_linear" -> gainLin,
  "gain_db" -> gainDb
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
