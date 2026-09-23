(* autoeq-qa case autoeq-qa.rq06-tone-render.v1 (catalogue family RQ06).
   Independent sine-tone synthesis: raw samples Sin[2 Pi f t], exact peak
   1.0 (10 integer cycles at 1000 Hz / 8000 Hz over 0.01 s), peak
   normalization to the declared SYNTHESIS_PEAK headroom (1 dB below full
   scale), RMS of the normalized tone = peak/Sqrt[2], and the closed-form
   sample count. Invalid frequency/duration inputs must be refused. *)

Off[General::munfl];
sr = 8000.;
freq = 1000.;
dur = 0.01;
seed = 7;
n = 80;
peak = 0.89125094;

raw = Table[N[Sin[2 Pi freq (i - 1)/sr], 16], {i, 1, n}];
rawPeak = N[Max[Abs[raw]], 16];
samples = N[peak*raw, 16];
rms = N[Sqrt[Mean[samples^2]], 16];
rmsClosed = N[peak/Sqrt[2.], 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rq06-tone-render.v1",
  "family" -> "RQ06",
  "wolfram_engine" -> "15.0.0",
  "sample_rate_hz" -> sr,
  "units" -> "linear amplitude, Hz, s",
  "calibration" -> "peak headroom SYNTHESIS_PEAK = 0.89125094 (-1 dBFS)",
  "freq_hz" -> freq,
  "duration_s" -> dur,
  "seed" -> seed,
  "frames" -> n,
  "raw_peak" -> rawPeak,
  "synthesis_peak" -> peak,
  "samples" -> samples,
  "rms" -> rms,
  "rms_closed_form" -> rmsClosed,
  "invalid_freq_hz" -> 4000.,
  "invalid_duration_s" -> 0.0005
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
