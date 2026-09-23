(* autoeq-qa case autoeq-qa.re22-limiter-gain-envelope.v1 (catalogue family RE22).
   Independent limiter-lookahead contract and composite gain envelope:
   lookahead latency is Floor[lookaheadMs * sr / 1000] samples
   (5 ms -> 240 at 48 kHz, 220 at 44.1 kHz); the ceiling clamps at
   Min[requested, -1] dBFS; the realized composite report takes the
   band peak over the correction band, the full-span peak and its
   frequency, and the within-envelope flag against the gain cap.
   The oracle never calls the Rust limiter or envelope code; it
   evaluates the closed forms and the Max/Position reductions, so it
   catches ms-vs-sample slips, missing clamps, and peak-picking errors. *)

Off[General::munfl];
lookaheadMs = 5.;
rates = {48000., 44100.};
latencies = Floor[lookaheadMs*#/1000.] & /@ rates;

requestedCeilings = {-3., 0.};
ceilings = Min[#, -1.] & /@ requestedCeilings;

freqs = {20., 50., 100., 200., 500., 1000., 2000.};
realizedDb = {1., 2., 5.5, 3., 4., 6.2, 0.5};
band = {50., 500.};
maxGainDb = 6.;

inBand = Pick[realizedDb, Map[band[[1]] <= # <= band[[2]] &, freqs]];
bandPeak = Max[inBand];
spanPeak = Max[realizedDb];
spanPos = First[First[Position[realizedDb, spanPeak]]];
spanFreq = freqs[[spanPos]];
within = spanPeak <= maxGainDb;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re22-limiter-gain-envelope.v1",
  "family" -> "RE22",
  "grid" -> "explicit shared 7-point grid, Hz",
  "units" -> "latency in integer samples; gains in dB",
  "calibration" -> "digital full-scale ceiling in dBFS; envelope relative to 0 dB correction",
  "lookahead_ms" -> lookaheadMs,
  "sample_rates_hz" -> rates,
  "latency_samples" -> latencies,
  "requested_ceilings_dbfs" -> requestedCeilings,
  "ceilings_dbfs" -> ceilings,
  "freqs_hz" -> freqs,
  "realized_db" -> realizedDb,
  "band_hz" -> band,
  "max_gain_db" -> maxGainDb,
  "band_peak_gain_db" -> N[bandPeak, 16],
  "full_span_peak_gain_db" -> N[spanPeak, 16],
  "full_span_peak_freq_hz" -> spanFreq,
  "within_envelope" -> within
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
