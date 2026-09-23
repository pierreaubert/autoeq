(* autoeq-qa case autoeq-qa.re08-split-band-sum.v1 (catalogue family RE08).
   Independent evaluation of the mixed-crossover frequency split plus a
   Butterworth-4 (24 dB/oct) parallel band sum with bulk delay, from
   published definitions only. (1) Split rule (the versioned contract in
   roomeq-engine mixed_crossover.rs via split_curve_at_frequency):
   splitIdx is the first grid index with f >= fc (0-based split_idx
   in Rust); the low band keeps 0-based indices through split_idx + 2
   and the high band starts at split_idx - 3 (6 shared points). (2) BW4 transfers from the published
   RBJ cookbook lowpass/highpass equations with the Butterworth
   second-order-section Q values 1/(2 Sin[Pi/8]) and 1/(2 Sin[3 Pi/8]);
   driver 0 (flat 80 dB, 0 deg) takes the low-pass branch with +1.5 dB
   gain, driver 1 (flat 80 dB, 0 deg) the high-pass branch with -0.5 dB
   gain and a 2.5 ms bulk delay applied as Exp[-I 2 Pi f tau]; branches
   sum as complex pressure, reported with 20 Log10 magnitudes floored at
   1e-12. No excess-phase or FIR-ripple claim is made here. *)

Off[General::munfl];
sr = 48000.;
fc = 500.;
n = 100;
freqs = N[10^Range[Log10[20.], Log10[20000.], (Log10[20000.] - Log10[20.])/(n - 1)], 16];

splitIdx = First[FirstPosition[freqs, f_ /; f >= fc]];
lowFreqs = Take[freqs, splitIdx + 2];
highFreqs = Drop[freqs, splitIdx - 4];

(* Butterworth 4th-order prototype section Q values (published). *)
q1 = N[1/(2 Sin[Pi/8]), 16];
q2 = N[1/(2 Sin[3 Pi/8]), 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0];
cs = Cos[w0];

(* RBJ lowpass / highpass coefficients for one section Q. *)
lpCoeffs[q_] := Module[{al = sn/(2 q)},
  {{(1 - cs)/2, 1 - cs, (1 - cs)/2}, {1 + al, -2 cs, 1 - al}}];
hpCoeffs[q_] := Module[{al = sn/(2 q)},
  {{(1 + cs)/2, -(1 + cs), (1 + cs)/2}, {1 + al, -2 cs, 1 - al}}];
sectionResp[coeffs_, f_] := Module[{z = Exp[-I N[2 Pi f/sr, 16]], b, a},
  b = coeffs[[1]]; a = coeffs[[2]];
  (b[[1]] + b[[2]] z + b[[3]] z^2)/(a[[1]] + a[[2]] z + a[[3]] z^2)];
lpBranch[f_] := sectionResp[lpCoeffs[q1], f] * sectionResp[lpCoeffs[q2], f];
hpBranch[f_] := sectionResp[hpCoeffs[q1], f] * sectionResp[hpCoeffs[q2], f];

g0 = N[10^(1.5/20), 16];
g1 = N[10^(-0.5/20), 16];
tau1 = 0.0025;
resp[f_] := Module[{z0, z1},
  z0 = N[10^(80/20), 16] * g0 * lpBranch[f];
  z1 = N[10^(80/20), 16] * g1 * hpBranch[f] * Exp[-I N[2 Pi f, 16] tau1];
  z0 + z1];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
comb = resp /@ freqs;
mags = (N[20 Log10[Max[Abs[#], 10^-12]], 16] & /@ comb);
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re08-split-band-sum.v1",
  "family" -> "RE08",
  "sample_rate_hz" -> sr,
  "grid" -> "logspace(100, 20 Hz, 20000 Hz)",
  "units" -> "Hz, dB SPL, degrees, ms",
  "calibration" -> "absolute SPL preserved per branch; flat 80 dB drivers; no output normalization",
  "split_rule" -> "first index with f >= fc; low keeps 0-based through split_idx+2, high starts at split_idx-3",
  "split_freq_hz" -> fc,
  "split_index_1based" -> splitIdx,
  "low_freqs_hz" -> lowFreqs,
  "high_freqs_hz" -> highFreqs,
  "crossover" -> "Butterworth4",
  "crossover_freq_hz" -> fc,
  "section_q" -> {q1, q2},
  "driver_spl_db" -> {80., 80.},
  "driver_phase_deg" -> {0., 0.},
  "gains_db" -> {1.5, -0.5},
  "delays_ms" -> {0., 2.5},
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair /@ comb),
  "combined_db" -> mags
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
