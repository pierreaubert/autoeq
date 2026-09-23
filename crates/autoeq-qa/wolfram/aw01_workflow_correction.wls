(* autoeq-qa case autoeq-qa.aw01-workflow-correction.v1 (catalogue family AW01).
   Speaker workflow composition through the public in-memory
   autoeq-workflow API: "flat"/"harman" target construction on a fixed
   grid plus realized corrected response (input + fixed Peak filter) and
   residual RMS. The RBJ cookbook peak transfer is evaluated here as an
   independent direct SOS sum (b0+b1 z^-1+b2 z^-2)/(a0+a1 z^-1+a2 z^-2),
   z^-1 = Exp[-I 2 Pi f/sr]; the target shapes follow the published
   definitions ("flat" = 0 dB; "harman" = -0.8*log2(f/1000)). Sample
   rate 48000 Hz; grid explicit; units relative dB SPL; calibration
   relative (0 dB = flat-target reference); magnitude-domain agreement
   only (no phase/time/spatial claims). *)

Off[General::munfl];
sr = 48000.;
grid = {100., 250., 500., 1000., 2000., 4000., 8000., 16000.};
inputSpl = {2., 3.5, 5., 6., 5., 3., 1., 0.};
fc = 1000.; q = 1.; gainDb = -6.;

a = N[10^(gainDb/40), 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0]; cw = Cos[w0]; alpha = sn/(2 q);
b0 = 1 + alpha*a; b1 = -2*cw; b2 = 1 - alpha*a;
a0 = 1 + alpha/a; a1 = -2*cw; a2 = 1 - alpha/a;
h[f_] := (b0 + b1*Exp[-I N[2 Pi f/sr, 16]] + b2*Exp[-I N[2 Pi f/sr, 16]]^2) / (a0 + a1*Exp[-I N[2 Pi f/sr, 16]] + a2*Exp[-I N[2 Pi f/sr, 16]]^2);
splDb[v_] := 20*Log[10, Abs[v]];

eq = N[splDb[h[#]] & /@ grid, 16];
corrected = N[inputSpl + eq, 16];
residual = N[-corrected, 16];
rms = N[Sqrt[Mean[residual^2]], 16];
harman = N[-0.8*Log[2, #/1000.] & /@ grid, 16];
flat = Table[0., {Length[grid]}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.aw01-workflow-correction.v1",
  "family" -> "AW01",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "units" -> "dB SPL relative",
  "calibration" -> "relative; 0 dB = flat-target reference",
  "filter" -> <|"type" -> "Peak", "fc_hz" -> fc, "q" -> q, "gain_db" -> gainDb|>,
  "input_spl_db" -> inputSpl,
  "flat_target_db" -> flat,
  "harman_target_db" -> harman,
  "eq_response_db" -> eq,
  "corrected_spl_db" -> corrected,
  "residual_db" -> residual,
  "residual_rms_db" -> rms
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
