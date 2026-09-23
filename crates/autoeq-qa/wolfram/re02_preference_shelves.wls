(* autoeq-qa case autoeq-qa.re02-preference-shelves.v1 (catalogue family RE02).
   Independent evaluation of the CEA/preference shelf layer: a low-shelf
   (+4 dB at 200 Hz) and a high-shelf (-3 dB at 8000 Hz) from the
   published RBJ cookbook shelf equations with S = 1
   (beta = Sqrt[2 A]), evaluated directly as
   (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2), z^-1 = Exp[-I 2 Pi f/sr],
   applied to a flat 85 dB curve: spl = 85 + 20 Log10[Abs[H]].
   The oracle never calls generate_preference_filters or
   simulate_correction; it re-derives the shelves from the preference
   parameters. Catches shelf mixups, sign flips, and 10-vs-20 dB slips. *)

Off[General::munfl];
sr = 48000.;
levelDb = 85.;
bassDb = 4.;
bassFc = 200.;
trebleDb = -3.;
trebleFc = 8000.;
freqs = {20., 50., 100., 200., 500., 1000., 2000., 4000., 8000., 12000., 16000., 20000.};

shelfCoeffs[fc_, gdb_, kind_] := Module[{a, w0, sn, cs, beta},
  a = N[10^(gdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0];
  cs = Cos[w0];
  beta = Sqrt[2*a];
  If[kind == "low",
    {a*((a + 1) - (a - 1)*cs + beta*sn),
     2*a*((a - 1) - (a + 1)*cs),
     a*((a + 1) - (a - 1)*cs - beta*sn),
     (a + 1) + (a - 1)*cs + beta*sn,
     -2*((a - 1) + (a + 1)*cs),
     (a + 1) + (a - 1)*cs - beta*sn},
    {a*((a + 1) + (a - 1)*cs + beta*sn),
     -2*a*((a - 1) + (a + 1)*cs),
     a*((a + 1) + (a - 1)*cs - beta*sn),
     (a + 1) - (a - 1)*cs + beta*sn,
     2*((a - 1) - (a + 1)*cs),
     (a + 1) - (a - 1)*cs - beta*sn}]
];

{b0l, b1l, b2l, a0l, a1l, a2l} = shelfCoeffs[bassFc, bassDb, "low"];
{b0h, b1h, b2h, a0h, a1h, a2h} = shelfCoeffs[trebleFc, trebleDb, "high"];

section[f_, b0_, b1_, b2_, a0_, a1_, a2_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];

response[f_] := section[f, b0l, b1l, b2l, a0l, a1l, a2l] *
  section[f, b0h, b1h, b2h, a0h, a1h, a2h];
splOut[f_] := N[levelDb + 20*Log10[Abs[response[f]]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re02-preference-shelves.v1",
  "family" -> "RE02",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 12-point grid, Hz",
  "units" -> "dB SPL on an absolute 85 dB flat input curve",
  "calibration" -> "flat 85 dB reference curve, preference shelves only",
  "level_db" -> levelDb,
  "bass_shelf" -> <|"freq_hz" -> bassFc, "gain_db" -> bassDb|>,
  "treble_shelf" -> <|"freq_hz" -> trebleFc, "gain_db" -> trebleDb|>,
  "design" -> "RBJ-cookbook-shelves S=1",
  "freqs_hz" -> freqs,
  "spl_db" -> (splOut[#] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
