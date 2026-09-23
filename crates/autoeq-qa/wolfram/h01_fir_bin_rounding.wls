(* autoeq-qa case autoeq-qa.h01-fir-bin-rounding.v1 (catalogue family H01).
   Independent budget for the FIR test-helper bin rounding in
   tests/fir_tests/compute.rs: zero-padded FFT of length N=16 over 4 taps,
   freqStep = sr/N, query bin = Round(f/freqStep), reported dB at the bin
   center via the direct DFT sum (unit-scale FFT, no normalization), versus
   the exact-frequency DTFT. Both from direct sums, never the Rust FFT. *)

Off[General::munfl];
coeffs = {0.25, 0.25, 0.25, 0.25};
sr = 48000.; nfft = 16;
step = N[sr/nfft, 16];
queries = {1000., 3000., 9000.};
dft[f_] := Sum[coeffs[[k + 1]]*Exp[-I*2*Pi*f*k/sr], {k, 0, 3}];
binOf[f_] := Round[f/step];
binDb[f_] := Module[{b = binOf[f], h},
  h = Sum[coeffs[[k + 1]]*Exp[-I*2*Pi*(b*step)*k/sr], {k, 0, 3}];
  N[20*Log10[Max[Abs[h], 1.*10^-10]], 16]];
exactDb[f_] := N[20*Log10[Max[Abs[dft[f]], 1.*10^-10]], 16];
bins = binOf[#] & /@ queries;
binDbs = binDb[#] & /@ queries;
exactDbs = exactDb[#] & /@ queries;
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.h01-fir-bin-rounding.v1",
  "family" -> "H01",
  "sample_rate_hz" -> sr,
  "grid_hz" -> queries,
  "units" -> "dB (20 log10 magnitude, floor 1e-10)",
  "calibration" -> "N=16 zero-padded DFT, bin=Round(f/step); exact DTFT budgeted separately",
  "taps" -> coeffs,
  "fft_size" -> nfft,
  "bins" -> bins,
  "bin_db" -> binDbs,
  "exact_db" -> exactDbs
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
