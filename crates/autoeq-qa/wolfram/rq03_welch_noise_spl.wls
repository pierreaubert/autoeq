(* autoeq-qa case autoeq-qa.rq03-welch-noise-spl.v1 (catalogue family RQ03).
   Independent pressure-calibrated Welch noise estimate for a
   bin-centered 1500 Hz tone (bin 64 of a 2048 frame at 48 kHz):
   periodic Hann window w[i] = 0.5 - 0.5 Cos[2 Pi i/N], half hops
   with the end-anchored final frame, per-frame mean removal,
   unnormalized forward DFT by direct sum, one-sided power
   (factor 2, DC/Nyquist factor 1) averaged over frames and
   normalized by rate*Sum[w^2]*frames, calibrated by
   pascalsPerSample^2 with flat 0 dB log-frequency correction, and
   bin-summed nominal octaves with SPL = 10 Log10[power/p0^2],
   p0 = 20 µPa. The Rust test calls calibrated_capture_noise;
   agreement checks the window convention (periodic Hann, not
   symmetric), the noise-power (not coherent-gain) normalization,
   mean removal, one-sided factors, and Pa2/Hz-to-SPL scaling. *)

Off[General::munfl];
rate = 48000.;
size = 2048;
toneBin = 64;
toneHz = N[toneBin*rate/size, 16];
amp = 0.5;
recordLen = 6144;
xs = Table[N[amp*Cos[2 Pi toneBin n/size], 16], {n, 0, recordLen - 1}];
hop = size/2;
starts = Range[0, recordLen - size, hop];
window = Table[N[0.5 - 0.5*Cos[2 Pi i/size], 16], {i, 0, size - 1}];
winSq = Total[window^2];
norm = N[rate*winSq*Length[starts], 16];
frameMean[s_] := Total[Take[xs, {s + 1, s + size}]]/N[size, 16];
dftMagSq[s_, k_] := Module[{m = frameMean[s]},
  Abs[Sum[(xs[[s + j + 1]] - m)*window[[j + 1]]*
    Exp[-I N[2 Pi j k/size, 16]], {j, 0, size - 1}]]^2
];
sided[k_] := If[k == 0 || k == size/2, 1., 2.];
psd[k_] := N[sided[k]*Total[dftMagSq[#, k] & /@ starts]/norm, 16];

pascalsPerSample = 2.;
calGain = N[pascalsPerSample^2, 16];
pressurePsd[k_] := N[psd[k]*calGain, 16];
spacing = N[rate/size, 16];
octLo = N[2000/Sqrt[2], 16];
octHi = N[2000*Sqrt[2], 16];
octBins = Select[Range[1, size/2],
  #*spacing >= octLo && #*spacing < octHi &];
octPower = Total[(pressurePsd[#]*spacing) & /@ octBins];
p0 = 20*^-6;
octSpl = N[10*(Log10[octPower] - Log10[p0^2]), 16];
spotBins = Range[61, 67];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rq03-welch-noise-spl.v1",
  "family" -> "RQ03",
  "sample_rate_hz" -> rate,
  "grid" -> "FFT bins of one 2048 frame; octave sums over bin centers",
  "units" -> <|"psd" -> "Pa^2/Hz", "spl" -> "dB re 20 uPa"|>,
  "calibration" -> "flat 0 dB correction; pascals per sample declared",
  "tone" -> <|"freq_hz" -> toneHz, "bin" -> toneBin,
    "amplitude" -> amp, "record_samples" -> recordLen|>,
  "frame_samples" -> size,
  "frame_starts" -> starts,
  "bin_spacing_hz" -> spacing,
  "window" -> "periodic Hann",
  "window_sum_sq" -> N[winSq, 16],
  "psd_normalization" -> norm,
  "pascals_per_sample" -> pascalsPerSample,
  "valid_band_hz" -> {100., 12000.},
  "spot_bins" -> spotBins,
  "spot_bin_freqs_hz" -> (N[#*spacing, 16] & /@ spotBins),
  "spot_pressure_psd" -> (pressurePsd /@ spotBins),
  "octave_2k_bins" -> octBins,
  "octave_2k_power" -> N[octPower, 16],
  "octave_2k_spl_db" -> octSpl,
  "unavailable_octave" -> "31.5"
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
