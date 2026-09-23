(* autoeq-qa case autoeq-qa.c02-peq-layouts.v1 (catalogue family C02).
   Independent check of every PeqModel parameter layout plus the realized
   cascade transfer. Layout table states params-per-filter and the fixed
   filter type at each position of a 3-filter vector (from the model
   definitions, not from Rust). Realized transfer: decode x (freq = 10^log),
   build each RBJ cookbook peak ((b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2),
   raw-a0 form, ratio is normalization-invariant), multiply the cascade,
   evaluate at the exact grid with z^-1 = Exp[-I 2 Pi f/sr]. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 500., 1000., 2000., 4000., 8000., 16000.};

(* Layout contract: {model, paramsPerFilter, fixed types at positions 1..3
   of a 3-filter vector; "free" = decoded from the type parameter}. *)
models = {"pk", "hp-pk", "hp-pk-lp", "ls-pk", "ls-pk-hs", "free-pk-free", "free"};
ppf = {3, 3, 3, 3, 3, 4, 4};
fixedTypes = {
  {"peak", "peak", "peak"},
  {"highpass-variable-q", "peak", "peak"},
  {"highpass-variable-q", "peak", "lowpass"},
  {"lowshelf", "peak", "peak"},
  {"lowshelf", "peak", "highshelf"},
  {"free", "peak", "free"},
  {"free", "free", "free"}
};

(* Realized-transfer fixture: Pk model, two peaks. *)
xvec = {3., 1., 6., 3.6020599913279625, 0.8, -4.};
filts = {
  <|"type" -> "peak", "freq_hz" -> N[10^3., 16], "q" -> 1., "gain_db" -> 6.|>,
  <|"type" -> "peak", "freq_hz" -> N[10^3.6020599913279625, 16], "q" -> 0.8, "gain_db" -> -4.|>
};

rbjPeak[fc_, q_, gdb_] := Module[{a, w0, sn, cw, al},
  a = N[10^(gdb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 q);
  {1 + al*a, -2*cw, 1 - al*a, 1 + al/a, -2*cw, 1 - al/a}
];

coeffs = rbjPeak[#["freq_hz"], #["q"], #["gain_db"]] & /@ filts;
stage[{b0_, b1_, b2_, a0_, a1_, a2_}, f_] := Module[{z = Exp[-I N[2 Pi f/sr, 16]]},
  (b0 + b1*z + b2*z^2)/(a0 + a1*z + a2*z^2)
];
htotal[f_] := Times @@ (stage[#, f] & /@ coeffs);
complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c02-peq-layouts.v1",
  "family" -> "C02",
  "sample_rate_hz" -> sr,
  "grid_hz" -> grid,
  "units" -> "linear-complex-transfer",
  "calibration" -> "none",
  "layout_models" -> models,
  "layout_params_per_filter" -> ppf,
  "layout_fixed_types_3" -> fixedTypes,
  "x" -> xvec,
  "filters" -> (<|"type" -> #["type"], "freq_hz" -> #["freq_hz"], "q" -> #["q"], "gain_db" -> #["gain_db"]|> & /@ filts),
  "freqs_hz" -> grid,
  "response_re_im" -> (complexPair[htotal[#]] & /@ grid)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
