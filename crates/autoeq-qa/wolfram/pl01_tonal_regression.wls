(* autoeq-qa case autoeq-qa.pl01-tonal-regression.v1 (catalogue family PL01).
   Tonal-balance regression: ordinary least-squares fit of SPL against
   log2(f) over [fmin, fmax], slope in dB/octave, intercept restated at
   1 kHz, plus regression-line evaluation. Fixture SPL lies exactly on
   y = slopeTrue*log2(f/1000) + iceptTrue so the closed-form normal
   equations must recover the slope/intercept. Independent of the Rust
   loop; N[...,16]. Units dB SPL relative; no sample rate involved
   (frequency ratios only); calibration relative. *)

Off[General::munfl];
freqs = {100., 200., 400., 800., 1600., 3200.};
fmin = 100.; fmax = 3200.;
slopeTrue = -1.5; iceptTrue = 2.;
spl = N[slopeTrue*Log[2, #/1000.] + iceptTrue & /@ freqs, 16];
lineFreqs = {100., 1000., 10000.};

x = N[Log[2, #] & /@ freqs, 16];
n = Length[x];
sx = Total[x]; sy = Total[spl];
sxy = Total[x*spl]; sx2 = Total[x^2];
denom = n*sx2 - sx^2;
slope = (n*sxy - sx*sy)/denom;
iceptRaw = (sy*sx2 - sx*sxy)/denom;
icept1k = iceptRaw + slope*Log[2, 1000.];
line = N[icept1k + slope*Log[2, #/1000.] & /@ lineFreqs, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.pl01-tonal-regression.v1",
  "family" -> "PL01",
  "sample_rate_hz" -> "n/a (frequency ratios only)",
  "freqs_hz" -> freqs,
  "fmin_hz" -> fmin,
  "fmax_hz" -> fmax,
  "units" -> "dB SPL relative",
  "calibration" -> "relative; exact-line fixture",
  "slope_true_db_per_oct" -> slopeTrue,
  "intercept_true_db_at_1k" -> iceptTrue,
  "spl_db" -> spl,
  "slope_db_per_oct" -> N[slope, 16],
  "intercept_db_at_1k" -> N[icept1k, 16],
  "line_freqs_hz" -> lineFreqs,
  "line_spl_db" -> line
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
