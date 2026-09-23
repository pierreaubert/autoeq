(* autoeq-qa case autoeq-qa.ac02-apo-roundtrip-gap.v1 (catalogue family AC02).
   APO export objective gap: one Peak filter at fractional 997.3 Hz is
   serialized with integer-Hz centers (997 Hz). The oracle parses those
   delivered rounded parameters (embedded below as inputs), rebuilds
   both responses with independent RBJ direct SOS sums, and evaluates
   the residual-RMS objective before/after rounding plus the absolute
   gap. It also pins the exact delivered APO filter line. Sample rate
   48000 Hz; grid explicit; units relative dB SPL; calibration
   relative; magnitude-domain agreement only. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 250., 500., 1000., 2000., 4000., 8000., 16000.};
fcOrig = 997.3; fcRounded = 997.; qq = 2.5; gainDb = 4.;
inputSpl = {0.5, 1., 1.5, 2., 1.5, 1., 0.5, 0.};
target = Table[0., {Length[grid]}];

peakSpl[fc_, f_] := Module[{a, w0, sn, cw, al, bb0, bb1, bb2, aa0, aa1, aa2, z, h},
  a = N[10^(gainDb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 qq);
  bb0 = 1 + al*a; bb1 = -2*cw; bb2 = 1 - al*a;
  aa0 = 1 + al/a; aa1 = -2*cw; aa2 = 1 - al/a;
  z = Exp[-I N[2 Pi f/sr, 16]];
  h = (bb0 + bb1*z + bb2*z^2)/(aa0 + aa1*z + aa2*z^2);
  20*Log[10, Abs[h]]
];
eqBefore = N[peakSpl[fcOrig, #] & /@ grid, 16];
eqAfter = N[peakSpl[fcRounded, #] & /@ grid, 16];
rmsBefore = N[Sqrt[Mean[(target - (inputSpl + eqBefore))^2]], 16];
rmsAfter = N[Sqrt[Mean[(target - (inputSpl + eqAfter))^2]], 16];
gap = N[Abs[rmsAfter - rmsBefore], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ac02-apo-roundtrip-gap.v1",
  "family" -> "AC02",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "units" -> "dB SPL relative",
  "calibration" -> "relative; flat target reference",
  "fc_orig_hz" -> fcOrig,
  "fc_rounded_hz" -> fcRounded,
  "q" -> qq,
  "gain_db" -> gainDb,
  "input_spl_db" -> inputSpl,
  "target_db" -> target,
  "apo_filter_line" -> "Filter  1: ON PK Fc   997 Hz Gain +4.00 dB Q 2.50",
  "eq_before_db" -> eqBefore,
  "eq_after_db" -> eqAfter,
  "rms_before_db" -> rmsBefore,
  "rms_after_db" -> rmsAfter,
  "objective_gap_db" -> gap
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
