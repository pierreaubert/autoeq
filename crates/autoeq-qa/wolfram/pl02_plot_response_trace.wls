(* autoeq-qa case autoeq-qa.pl02-plot-response-trace.v1 (catalogue family PL02).
   Displayed plot numerical traces from fixed Wolfram curves and fixed
   filters: combined EQ response (sum of two RBJ peak dB responses),
   corrected trace (input + EQ) and error trace (deviation - EQ).
   Each SOS evaluated independently as
   (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2), z^-1 = Exp[-I 2 Pi f/sr].
   Sample rate 48000 Hz; grid explicit; units relative dB SPL;
   calibration relative; magnitude-domain agreement only. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 250., 500., 1000., 2000., 4000., 8000., 16000.};
inputSpl = {1., 2., 2.5, 3., 2., 1., 0.5, 0.};
deviation = {0.5, 1., 1.5, 2., 1., 0.5, -0.5, -1.};
filters = {{500., 0.8, 4.}, {3000., 2., -5.}};

peakSpl[fc_, qq_, gainDb_, f_] := Module[{a, w0, sn, cw, al, bb0, bb1, bb2, aa0, aa1, aa2, z, h},
  a = N[10^(gainDb/40), 16];
  w0 = N[2 Pi fc/sr, 16];
  sn = Sin[w0]; cw = Cos[w0]; al = sn/(2 qq);
  bb0 = 1 + al*a; bb1 = -2*cw; bb2 = 1 - al*a;
  aa0 = 1 + al/a; aa1 = -2*cw; aa2 = 1 - al/a;
  z = Exp[-I N[2 Pi f/sr, 16]];
  h = (bb0 + bb1*z + bb2*z^2)/(aa0 + aa1*z + aa2*z^2);
  20*Log[10, Abs[h]]
];
eq = Table[Total[Table[peakSpl[filters[[k, 1]], filters[[k, 2]], filters[[k, 3]], grid[[i]]], {k, Length[filters]}]], {i, Length[grid]}];
eq = N[eq, 16];
corrected = N[inputSpl + eq, 16];
err = N[deviation - eq, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.pl02-plot-response-trace.v1",
  "family" -> "PL02",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "units" -> "dB SPL relative",
  "calibration" -> "relative; fixed synthetic curves",
  "filters" -> (<|"fc_hz" -> #[[1]], "q" -> #[[2]], "gain_db" -> #[[3]]|> & /@ filters),
  "input_spl_db" -> inputSpl,
  "deviation_db" -> deviation,
  "eq_response_db" -> eq,
  "corrected_spl_db" -> corrected,
  "error_db" -> err
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
