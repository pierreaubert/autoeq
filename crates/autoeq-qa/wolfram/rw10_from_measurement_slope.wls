(* autoeq-qa case autoeq-qa.rw10-from-measurement-slope.v1 (catalogue family RW10).
   Independent derivation of the workflow FromMeasurement target slope:
   ordinary least-squares regression of SPL vs log2(f) over the
   [200, 10000] Hz window, plus the in-window log-band mean level the
   target anchors to. The fixture SPL is exactly affine in log2(f)
   with slope -1.5 dB/oct through 80 dB at 1 kHz, so the closed form
   must return that slope; the oracle recomputes the sums directly,
   never by calling the Rust estimator. Catches log10/log2 mixups,
   window-exclusion slips, and missing mean subtraction. *)

Off[General::munfl];
freqs = {200., 300., 500., 800., 1200., 2000., 3500., 6000., 9000.};
trueSlope = -1.5;
anchorFreq = 1000.;
anchorSpl = 80.;
spl = (N[anchorSpl + trueSlope*(Log[2, #] - Log[2, anchorFreq]), 16] & /@ freqs);

winMin = 200.; winMax = 10000.;
xs = Log[2, #] & /@ freqs;
n = Length[freqs];
sx = Total[xs]; sy = Total[spl];
sxx = Total[xs^2]; sxy = Total[xs*spl];
slope = N[(n*sxy - sx*sy)/(n*sxx - sx^2), 16];
meanSpl = N[Mean[spl], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw10-from-measurement-slope.v1",
  "family" -> "RW10",
  "sample_rate_hz" -> 48000.,
  "grid" -> "explicit 9-point log-spanning grid 200..9000 Hz",
  "units" -> "dB SPL, dB/octave",
  "calibration" -> "synthetic affine-log fixture: -1.5 dB/oct through 80 dB at 1 kHz",
  "window_hz" -> {winMin, winMax},
  "freqs_hz" -> freqs,
  "spl_db" -> spl,
  "expected_slope_db_per_oct" -> slope,
  "expected_window_mean_db" -> meanSpl
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
