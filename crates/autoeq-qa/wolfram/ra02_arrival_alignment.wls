(* autoeq-qa case autoeq-qa.ra02-arrival-alignment.v1 (catalogue family RA02).
   Known-arrival alignment oracle: a 301-sample fixture (240 leading
   silent samples, a 0.9 peak, an exponential tail) whose threshold onset
   is exactly sample 240 = 5.0 ms at 48 kHz, plus a three-channel arrival
   map whose alignment delays are max-minus-arrival against the slowest
   channel. Catches off-by-one onset, sample/ms conversion, and
   align-to-fastest (min instead of max) defects. *)

Off[General::munfl];
sr = 48000.;
silence = 240;
peak = 0.9;
decay = 0.7;
tailLen = 60;
tail = Table[N[peak*decay^k, 16], {k, 1, tailLen}];
samples = Join[Table[0., {silence}], {peak}, tail];
arrivalSamples = silence;
arrivalMs = N[silence*1000./sr, 16];
arrivals = <|"A" -> 5., "B" -> 7.5, "C" -> 2.|>;
maxA = Max[Values[arrivals]];
delays = Association[Rule[#, N[maxA - arrivals[#], 16]] & /@ Keys[arrivals]];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra02-arrival-alignment.v1",
  "family" -> "RA02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> "time-domain samples at 48000 Hz",
  "units" -> "samples; ms",
  "calibration" -> "digital full-scale",
  "timing_origin" -> "recording start (sample 0)",
  "threshold_policy" -> "-40 dB peak-relative onset",
  "samples" -> N[samples, 16],
  "arrival_samples" -> arrivalSamples,
  "arrival_ms" -> arrivalMs,
  "arrivals_ms" -> arrivals,
  "alignment_delays_ms" -> delays
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
