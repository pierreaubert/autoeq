(* autoeq-qa case autoeq-qa.re17-interchannel-deviation.v1 (catalogue family RE17).
   Independent inter-channel deviation: each channel is normalized by its
   own mean over the analysis band, then the per-frequency max-min spread
   feeds midrange RMS/peak and passband RMS. Plus the log-frequency
   upper-band level reference (constant offset reproduces exactly).
   Direct arithmetic only. *)

Off[General::munfl];
freqs = {100., 200., 400., 800., 1600., 3200., 6400., 10000.};
chA = {80., 80., 80., 80., 80., 80., 80., 80.};
offsets = {0., 1., 2., 1., 0., -1., -2., -1.};
chB = chA + offsets;
f3 = 150.;
band = Select[Range[Length[freqs]], f3 <= freqs[[#]] <= 10000. &];
meanA = N[Mean[chA[[band]]], 16];
meanB = N[Mean[chB[[band]]], 16];
nA = chA - meanA;
nB = chB - meanB;
spread = Table[N[Max[nA[[i]], nB[[i]]] - Min[nA[[i]], nB[[i]]], 16], {i, Length[freqs]}];
midIdx = Select[Range[Length[freqs]], 200. <= freqs[[#]] <= 4000. &];
midVals = spread[[midIdx]];
midRms = N[Sqrt[Mean[midVals^2]], 16];
peak = N[Max[midVals], 16];
peakFreq = freqs[[midIdx[[First[First[Position[midVals, peak]]]]]]];
passVals = spread[[band]];
passRms = N[Sqrt[Mean[passVals^2]], 16];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re17-interchannel-deviation.v1",
  "family" -> "RE17",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> freqs,
  "grid" -> "shared identical channel grid (no resampling)",
  "units" -> <|"spl" -> "dB"|>,
  "calibration" -> "mic-1",
  "channel_a_spl_db" -> chA,
  "channel_b_spl_db" -> chB,
  "f3_hz" -> f3,
  "channel_a_band_mean_db" -> meanA,
  "channel_b_band_mean_db" -> meanB,
  "spread_db_per_freq" -> spread,
  "midrange_rms_db" -> midRms,
  "midrange_peak_db" -> peak,
  "midrange_peak_freq_hz" -> peakFreq,
  "passband_rms_db" -> passRms,
  "upper_band_reference" -> <|
    "curve_spl_db" -> 80.,
    "target_spl_db" -> 77.5,
    "correction_max_hz" -> 300.,
    "expected_offset_db" -> 2.5
  |>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
