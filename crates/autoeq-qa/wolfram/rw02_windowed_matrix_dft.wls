(* autoeq-qa case autoeq-qa.rw02-windowed-matrix-dft.v1 (catalogue family RW02).
   Independent windowed matrix-IR transform: symmetric Hann weights
   w[n] = 1/2 (1 - Cos[2 Pi n/(N-1)]) multiply each supplied ear IR,
   then the direct DFT sum X[k] = Sum_n x[n] Exp[-I 2 Pi k n/N] gives
   the half spectrum (k = 0..N/2). The oracle never calls the engine
   FFT or build_matrix_spectrum; it restates windowing and the DFT
   from their definitions, so it catches window omission, ear/speaker
   orientation swaps, and dropped bins. Matrix orientation under test:
   values = {L0, L1, R0, R1} per bin (left-ear block then right-ear). *)

Off[General::munfl];
sr = 48000.;
n = 8;
irL0 = {1., 0.5, 0.25, 0.125, 0., 0., 0., 0.};
irR0 = {0., 0.75, -0.25, 0.125, 0., 0., 0., 0.};
irL1 = {0.5, -0.5, 0.25, 0., 0.125, 0., 0., 0.};
irR1 = {0.25, 0.25, 0.25, -0.125, 0., 0., 0., 0.};

hann = Table[N[1/2 (1 - Cos[2 Pi i/(n - 1)]), 16], {i, 0, n - 1}];
win[ir_] := ir*hann;
dft[ir_, k_] := Sum[win[ir][[i + 1]]*Exp[-I N[2 Pi k i/n, 16]], {i, 0, n - 1}];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
bins = Range[0, n/2];
spectra = {
  {(complexPair[dft[irL0, #]] & /@ bins), (complexPair[dft[irR0, #]] & /@ bins)},
  {(complexPair[dft[irL1, #]] & /@ bins), (complexPair[dft[irR1, #]] & /@ bins)}
};
flatPair[{a_, b_}] := {a, b};
matrixValues = Table[
  Flatten[{flatPair[complexPair[dft[irL0, k]]], flatPair[complexPair[dft[irL1, k]]],
    flatPair[complexPair[dft[irR0, k]]], flatPair[complexPair[dft[irR1, k]]]}],
  {k, bins}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw02-windowed-matrix-dft.v1",
  "family" -> "RW02",
  "sample_rate_hz" -> sr,
  "grid" -> "5 DFT bins of an 8-point windowed IR at 48 kHz",
  "units" -> "linear magnitude, radians; IR samples dimensionless",
  "calibration" -> "unity input scale; symmetric Hann window, no FDW",
  "fft_size" -> n,
  "window" -> "symmetric Hann",
  "hann_weights" -> hann,
  "ir_l0" -> irL0,
  "ir_r0" -> irR0,
  "ir_l1" -> irL1,
  "ir_r1" -> irR1,
  "spectra_speaker_ear_bin_re_im" -> spectra,
  "matrix_values_bin_re_im_flat" -> matrixValues
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
