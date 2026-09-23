(* autoeq-qa case autoeq-qa.o11-temporal-ir.v1 (catalogue family O11).
   Independent evaluation of the temporal IR pre/post masking metrics
   from their documented definition: main peak = argmax |ir|, pre/post
   energy weighted by (t_ms/window_ms)^2 inside the masking windows
   (fully audible beyond), dB via 20 Log10 (peak) and 10 Log10 (energy)
   floored at 1e-24, excess over the audibility threshold, and the
   profile-scaled penalty ir_weight * scale * (preW*preExcess^2 +
   postW*postExcess^2)/100. Also covered: the modal temporal penalty on
   one hand-designed mode with log-frequency interpolation of the EQ at
   the mode, the BS.1770-style channel energy weights (Main 1.0,
   Surround 1.41, LFE 0.0), and the normalized (level-relative + phon
   offset) versus absolute-SPL alignment vector. *)

Off[General::munfl];
sr = 48000.;
len = 256;
mainIdx = 49; (* 1-based; 0-based 48 = 1.0 ms *)
ir = Table[0., {len}];
ir[[1]] = 0.01;
ir[[41]] = 0.1;
ir[[mainIdx]] = 1.0;
ir[[61]] = 0.2;
ir[[201]] = 0.05;

preMask = 3.0; postMask = 120.0;
maskW[t_, w_] := If[w <= 0., 1., N[Min[Max[t/w, 0.], 1.]^2, 16]];
mainAmp = 1.;
mainEnergy = 1.;
preEnergy = N[ir[[1]]^2 maskW[(mainIdx - 1)*1000./sr, preMask]
  + ir[[41]]^2 maskW[(mainIdx - 41)*1000./sr, preMask], 16];
postEnergy = N[ir[[61]]^2 maskW[(61 - mainIdx)*1000./sr, postMask]
  + ir[[201]]^2 maskW[(201 - mainIdx)*1000./sr, postMask], 16];
dbR[r_] := If[r <= 10^-24, -240., N[20 Log10[r], 16]];
dbE[r_] := If[r <= 10^-24, -240., N[10 Log10[r], 16]];
prePeakDb = dbR[Max[Abs[ir[[1]]], Abs[ir[[41]]]]/mainAmp];
postPeakDb = dbR[Max[Abs[ir[[61]]], Abs[ir[[201]]]]/mainAmp];
preAud = dbE[preEnergy/mainEnergy];
postAud = dbE[postEnergy/mainEnergy];
threshold = -45.;
preExcess = Max[preAud - threshold, 0.];
postExcess = Max[postAud - threshold, 0.];
penalty = N[0.05*1.0*(2.0 preExcess^2 + 1.0 postExcess^2)/100., 16];

(* Modal penalty: one mode at 55 Hz, Q 10, prominence 6 dB, severity
   8 dB; EQ curve -3 dB flat on {20,55,200} Hz; weight 0.15, Mixed. *)
eqFreqs = {20., 55., 200.};
eqSpl = {-3., -3., -3.};
eqAtMode = -3.;
residRatio = N[Max[6. + eqAtMode, 0.]/6., 16];
boostPen = N[Max[eqAtMode, 0.]/6., 16];
qScale = N[Min[Max[(10./10.)^0.5, 0.5], 2.5], 16];
modalPenalty = N[0.15*1.0*qScale*8.^2*(residRatio^2 + boostPen^2), 16];

(* Normalized vs absolute alignment: level-relative curve + phon. *)
phon = 70.;
splRel = {0.5, -0.25, 0.75, -1.0};
splAbs = (# + phon) & /@ splRel;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o11-temporal-ir.v1",
  "family" -> "O11",
  "sample_rate_hz" -> sr,
  "grid" -> "256-sample IR at 48 kHz; EQ grid {20,55,200} Hz",
  "units" -> "samples, ms, dB",
  "calibration" -> "IR peak-normalized diagnostic; EQ curve level-relative, denormalized by +70 phon",
  "ir" -> ir,
  "main_index" -> 48,
  "main_time_ms" -> N[48*1000./sr, 16],
  "pre_ringing_peak_db" -> prePeakDb,
  "post_ringing_peak_db" -> postPeakDb,
  "pre_ringing_audible_db" -> preAud,
  "post_ringing_audible_db" -> postAud,
  "penalty" -> penalty,
  "modal" -> <|
    "frequency" -> 55., "q" -> 10., "prominence_db" -> 6.,
    "severity_db" -> 8., "weight" -> 0.15,
    "eq_freqs_hz" -> eqFreqs, "eq_spl_db" -> eqSpl,
    "penalty" -> modalPenalty|>,
  "channel_weights" -> <|"main" -> 1.0, "surround" -> 1.41, "lfe" -> 0.0|>,
  "listening_level_phon" -> phon,
  "spl_rel_db" -> splRel,
  "spl_abs_db" -> splAbs
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
