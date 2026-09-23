(* autoeq-qa case autoeq-qa.c02-free-types.v1 (catalogue family C02).
   Free-model layout: every filter carries an explicit type code decoded
   by Floor(code) (0->peak, 3->lowshelf, 4->highshelf, ...); freq = 10^log.
   Realized transfer uses the published RBJ cookbook shelf/peak designs
   (raw-a0 form) cascaded by complex multiplication, evaluated with
   z^-1 = Exp[-I 2 Pi f/sr]. Integer-center codes are used so the
   peq2x round trip is exact. *)

Off[General::munfl];
sr = 48000.;
grid = {50., 100., 200., 500., 1000., 2000., 8000., 12000., 16000.};

typeNames = {"peak", "lowpass", "highpass", "lowshelf", "highshelf",
  "highpass-variable-q", "bandpass", "notch", "allpass",
  "lowshelf-orfanidis", "highshelf-orfanidis", "peak-matched"};
decodeName[code_] := typeNames[[Floor[code] + 1]];

(* Free model x: {type, log10f, q, gain} x3: lowshelf@200 +4, peak@1k +6, highshelf@8k -3. *)
xvec = {3., 2.3010299956639813, 0.7, 4., 0., 3., 1., 6., 4., 3.9030899869919438, 0.9, -3.};
filts = {
  <|"type" -> "lowshelf", "freq_hz" -> N[10^2.3010299956639813, 16], "q" -> 0.7, "gain_db" -> 4.|>,
  <|"type" -> "peak", "freq_hz" -> N[10^3., 16], "q" -> 1., "gain_db" -> 6.|>,
  <|"type" -> "highshelf", "freq_hz" -> N[10^3.9030899869919438, 16], "q" -> 0.9, "gain_db" -> -3.|>
};

rbjPeak[fc_, q_, gdb_] := Module[{a, w0, sn, cw, al},
  a = N[10^(gdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 q);
  {1 + al*a, -2*cw, 1 - al*a, 1 + al/a, -2*cw, 1 - al/a}
];
rbjLowshelf[fc_, gdb_] := Module[{a, w0, sn, cw, be},
  a = N[10^(gdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; be = Sqrt[2 a];
  {a*((a + 1) - (a - 1)*cw + be*sn),
   2*a*((a - 1) - (a + 1)*cw),
   a*((a + 1) - (a - 1)*cw - be*sn),
   (a + 1) + (a - 1)*cw + be*sn,
   -2*((a - 1) + (a + 1)*cw),
   (a + 1) + (a - 1)*cw - be*sn}
];
rbjHighshelf[fc_, gdb_] := Module[{a, w0, sn, cw, be},
  a = N[10^(gdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; be = Sqrt[2 a];
  {a*((a + 1) + (a - 1)*cw + be*sn),
   -2*a*((a - 1) + (a + 1)*cw),
   a*((a + 1) + (a - 1)*cw - be*sn),
   (a + 1) - (a - 1)*cw + be*sn,
   2*((a - 1) - (a + 1)*cw),
   (a + 1) - (a - 1)*cw - be*sn}
];

design[f_] := Switch[f[["type"]],
  "peak", rbjPeak[f[["freq_hz"]], f[["q"]], f[["gain_db"]]],
  "lowshelf", rbjLowshelf[f[["freq_hz"]], f[["gain_db"]]],
  "highshelf", rbjHighshelf[f[["freq_hz"]], f[["gain_db"]]]
];
coeffs = design /@ filts;
stage[{b0_, b1_, b2_, a0_, a1_, a2_}, f_] := Module[{z = Exp[-I N[2 Pi f/sr, 16]]},
  (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
];
htotal[f_] := Times @@ (stage[#, f] & /@ coeffs);
complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c02-free-types.v1",
  "family" -> "C02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> grid,
  "units" -> "linear-complex-transfer",
  "calibration" -> "none",
  "type_code_names" -> typeNames,
  "decode_probe_codes" -> {0., 0.999, 3., 4., 11.999},
  "decode_probe_names" -> (decodeName /@ {0., 0.999, 3., 4., 11.999}),
  "x" -> xvec,
  "roundtrip_x" -> xvec,
  "filters" -> (<|"type" -> #["type"], "freq_hz" -> #["freq_hz"], "q" -> #["q"], "gain_db" -> #["gain_db"]|> & /@ filts),
  "freqs_hz" -> grid,
  "response_re_im" -> (complexPair[htotal[#]] & /@ grid)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
