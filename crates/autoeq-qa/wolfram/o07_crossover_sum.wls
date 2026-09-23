(* autoeq-qa case autoeq-qa.o07-crossover-sum.v1 (catalogue family O07).
   Independent evaluation of a 2-driver LR4 crossover with coherent
   acoustic summation, from the published RBJ cookbook lowpass/highpass
   design equations (never a transliteration of the Rust crossover
   builder). Driver 0 (80 dB, 0 deg) takes the low-pass branch and
   driver 1 (80 dB, 180 deg = inverted polarity) the high-pass branch
   of an LR4 (two cascaded Butterworth Q=1/Sqrt[2] sections) at 1000 Hz.
   Gains {+2,-1} dB and delays {0, 0.25} ms are applied as
   10^(g/20) and Exp[-I 2 Pi f tau]; branches are summed as complex
   pressure and reported as complex pairs plus 20 Log10 magnitudes
   floored at 1e-12, exactly the documented signal path. *)

Off[General::munfl];
sr = 48000.;
fc = 1000.;
n = 100;
freqs = N[10^Range[Log10[20.], Log10[20000.], (Log10[20000.] - Log10[20.])/(n - 1)], 16];

(* Butterworth 2nd-order prototype Q, doubled (cascaded) for LR4. *)
q = N[1/Sqrt[2], 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0];
cs = Cos[w0];
alpha = sn/(2 q);

(* RBJ lowpass coefficients. *)
lb0 = (1 - cs)/2; lb1 = 1 - cs; lb2 = (1 - cs)/2;
a0 = 1 + alpha; a1 = -2 cs; a2 = 1 - alpha;
(* RBJ highpass coefficients (same denominator). *)
hb0 = (1 + cs)/2; hb1 = -(1 + cs); hb2 = (1 + cs)/2;

branchMag2[b0_, b1_, b2_, f_] := Module[{z = Exp[-I N[2 Pi f/sr, 16]]},
  ((b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2))^2
];

g0 = N[10^(2/20), 16];
g1 = N[10^(-1/20), 16];
tau1 = 0.00025;

resp[f_] := Module[{lp, hp, z0, z1},
  lp = branchMag2[lb0, lb1, lb2, f];
  hp = branchMag2[hb0, hb1, hb2, f];
  z0 = N[10^(80/20), 16] * g0 * lp;
  z1 = N[10^(80/20), 16] * Exp[I Pi] * g1 * hp * Exp[-I N[2 Pi f, 16] tau1];
  z0 + z1
];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
comb = resp /@ freqs;
mags = (N[20 Log10[Max[Abs[#], 10^-12]], 16] & /@ comb);
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o07-crossover-sum.v1",
  "family" -> "O07",
  "sample_rate_hz" -> sr,
  "grid" -> "logspace(100, 20 Hz, 20000 Hz)",
  "units" -> "Hz, dB SPL, degrees, ms",
  "calibration" -> "absolute SPL preserved per driver; flat 80 dB drivers; no output normalization",
  "crossover" -> "LinkwitzRiley4",
  "crossover_freq_hz" -> fc,
  "driver_spl_db" -> {80., 80.},
  "driver_phase_deg" -> {0., 180.},
  "gains_db" -> {2., -1.},
  "delays_ms" -> {0., 0.25},
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair /@ comb),
  "combined_db" -> mags
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
