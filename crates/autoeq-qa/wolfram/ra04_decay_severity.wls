(* autoeq-qa case autoeq-qa.ra04-decay-severity.v1 (catalogue family RA04).
   Independent mode-decay and severity arithmetic: log-frequency
   interpolation of the published artificial/music decay-threshold tables,
   the Q-decay relation Q = f T pi / ln(1000), the temporal severity
   20 log10(q / qMax) (0 below threshold), the validated 32-250 Hz domain,
   and the measured-decay severity from an RT60 estimate with a 0.5
   confidence gate. Also pins the damped-sinusoid plant used for the
   regression check: x(t) = exp(-b t) sin(2 pi f t) with
   b = 3 ln(10) / RT60. Compared against
   roomeq_analysis::{temporal_targets, impulse_analysis::mode_decay}. Decay
   times are in seconds, Q dimensionless, severity in dB. *)

Off[General::munfl];
art = {{32., 0.90}, {50., 0.60}, {63., 0.45}, {100., 0.25}, {200., 0.17}, {250., 0.15}};
mus = {{32., 0.85}, {50., 0.70}, {63., 0.51}, {100., 0.30}, {200., 0.15}, {250., 0.12}};
domLo = 32.; domHi = 250.;

interp[f_, tbl_] := Module[{fc = Max[f, tbl[[1, 1]]], last = Length[tbl], a, r = tbl[[Length[tbl], 2]]},
  If[fc >= tbl[[last, 1]], tbl[[last, 2]],
   Do[If[fc >= tbl[[i, 1]] && fc <= tbl[[i + 1, 1]],
     a = N[(Log[fc] - Log[tbl[[i, 1]]])/(Log[tbl[[i + 1, 1]]] - Log[tbl[[i, 1]]]), 16];
     r = N[tbl[[i, 2]] + a*(tbl[[i + 1, 2]] - tbl[[i, 2]]), 16];
     Break[]], {i, 1, last - 1}];
   r]];
qForDecay[f_, t_] := N[f*t*Pi/Log[1000.], 16];
maxQ[f_, music_] := qForDecay[f, interp[f, If[music, mus, art]]];
severity[f_, q_, music_] := Module[{mq = maxQ[f, music]},
  N[If[q <= mq, 0., 20.*Log10[q/mq]], 16]];
inDomain[f_] := domLo <= f <= domHi;

probeF = {32., 50., 63., 75., 100., 200., 250.};
thrArt = interp[#, art] & /@ probeF;
thrMus = interp[#, mus] & /@ probeF;
qArt = qForDecay[#, interp[#, art]] & /@ probeF;

sevBelow = severity[100., 3., False];
sevAbove = severity[100., 30., False];
sevMusic = severity[63., 30., True];

(* Measured-decay severity pins: confident 0.9 s RT60 at 80 Hz, both tables. *)
rtTrue = 0.9; rtFreq = 80.; conf = 0.9; confGate = 0.5;
measArt = N[If[rtTrue > interp[rtFreq, art], 20.*Log10[rtTrue/interp[rtFreq, art]], 0.], 16];
measMus = N[If[rtTrue > interp[rtFreq, mus], 20.*Log10[rtTrue/interp[rtFreq, mus]], 0.], 16];

(* Damped-sinusoid regression plant: envelope rate from the -60 dB definition. *)
regCases = {{80., 0.5}, {80., 1.2}};
beta[rt_] := N[3.*Log[10.]/rt, 16];
plant = Table[With[{fr = c[[1]], rt = c[[2]]},
   <|"frequency_hz" -> fr, "rt60_s" -> rt, "beta_per_s" -> beta[rt]|>], {c, regCases}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra04-decay-severity.v1",
  "family" -> "RA04",
  "sample_rate_hz" -> 48000.,
  "sampling_note" -> "threshold/severity algebra is sample-free; 48 kHz pins the regression-plant IR synthesis",
  "grid_units" -> "Hz",
  "time_units" -> "seconds",
  "severity_units" -> "dB",
  "calibration" -> "published study threshold tables, 32-250 Hz validated domain",
  "artificial_table" -> art,
  "music_table" -> mus,
  "domain_lo_hz" -> domLo,
  "domain_hi_hz" -> domHi,
  "probe_freqs_hz" -> probeF,
  "threshold_artificial_s" -> thrArt,
  "threshold_music_s" -> thrMus,
  "max_q_artificial" -> qArt,
  "severity_below_db" -> sevBelow,
  "severity_above_db" -> sevAbove,
  "severity_music_db" -> sevMusic,
  "measured_rt60_s" -> rtTrue,
  "measured_freq_hz" -> rtFreq,
  "measured_confidence" -> conf,
  "confidence_gate" -> confGate,
  "measured_severity_artificial_db" -> measArt,
  "measured_severity_music_db" -> measMus,
  "regression_plant" -> plant,
  "regression_tolerance_rel" -> 0.15
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
