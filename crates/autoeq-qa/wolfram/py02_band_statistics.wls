(* autoeq-qa case autoeq-qa.py02-band-statistics.v1 (catalogue family PY02).
   Independent evaluation of the report-side band-mean statistic
   (scripts/src/acoustic_report.py band_mean): arithmetic mean of finite
   SPL points inside the closed band [lo, hi]; null when empty. *)

Off[General::munfl];
freqs = {100., 500., 1000., 2000., 3000., 5000.};
spl = {68., 70., 72., 71., 73., 69.};
lo = 500.; hi = 3000.;
inband = Select[Transpose[{freqs, spl}], (#[[1]] >= lo && #[[1]] <= hi) &];
mean = N[Mean[inband[[All, 2]]], 16];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.py02-band-statistics.v1",
  "family" -> "PY02",
  "sample_rate_hz" -> 48000.,
  "grid_hz" -> freqs,
  "units" -> "dB SPL",
  "calibration" -> "arithmetic mean over closed band, no weighting",
  "band_lo_hz" -> lo,
  "band_hi_hz" -> hi,
  "spl_db" -> spl,
  "band_mean_db" -> mean,
  "empty_band_is_null" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
