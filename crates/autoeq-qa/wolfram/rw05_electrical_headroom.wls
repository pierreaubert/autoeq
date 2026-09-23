(* autoeq-qa case autoeq-qa.rw05-electrical-headroom.v1 (catalogue family RW05).
   Independent transfer-matrix headroom: one serialized input->output
   path with a +6 dB gain stage gives transfer G = 10^(6/20) on every
   grid bin; with declared input peak limit p the sampled output
   amplitude is p*|G| per bin (same input added complexly, independent
   inputs added by magnitude). Peak = max over bins (first bin wins
   ties by strict comparison), peak_dBFS = 20 Log10[peak], required
   attenuation = Max[peak_dBFS, 0]. The oracle never calls the replay
   path; it restates the published accumulation, so it catches
   dB-factor slips, peak/limit swaps, and grid changes. *)

Off[General::munfl];
sr = 48000.;
freqs = {100., 500., 1000., 5000., 10000.};
gainDb = 6.;
inputPeak = 0.25;

g = N[10^(gainDb/20), 16];
amps = Table[N[inputPeak*g, 16], {Length[freqs]}];
peak = Max[amps];
peakDb = N[20*Log10[peak], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw05-electrical-headroom.v1",
  "family" -> "RW05",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 5-point grid, Hz",
  "units" -> "linear amplitude rel. full scale; dBFS",
  "calibration" -> "declared steady-sine input peak limit, no acoustic data",
  "freqs_hz" -> freqs,
  "gain_db" -> gainDb,
  "input_peak_limit" -> inputPeak,
  "amplitudes" -> amps,
  "peak_amplitude" -> N[peak, 16],
  "peak_frequency_hz" -> freqs[[1]],
  "peak_dbfs" -> peakDb,
  "required_attenuation_db" -> N[Max[peakDb, 0.], 16]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
