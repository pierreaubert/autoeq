(* autoeq-qa case autoeq-qa.rc02-route-matrix-prediction.v1 (catalogue family RC02).
   Independent predeclared physical-output prediction: each route
   transfer is gainLin * Exp[-I 2 Pi f delayMs/1000] (times the RBJ
   peak section for route R), each plant is the direct DTFT of its
   measured IR taps, and every source/seat prediction is the
   route-matrix times source product
   Sum_paths dsp[path](f) * plant[output, seat](f) * trialGain,
   reported as 20 Log10[Abs[sum]] + offsetDb with Arg in degrees.
   The Rust test replays the routes with RealizedDsp, DTFTs the
   plants with the shared FIR kernel, and forms the same sums. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 500., 1000., 2000., 5000., 10000.};
offsetDb = 0.;

gainL = 0.;
delayL = 1.;
gainR = -6.;
delayR = 0.5;
fc = 2000.;
q = 2.;
peakDb = 3.;

a = N[10^(peakDb/40), 16];
w0 = N[2 Pi fc/sr, 16];
sn = Sin[w0];
cw = Cos[w0];
alpha = sn/(2 q);
b0 = 1 + alpha*a;
b1 = -2*cw;
b2 = 1 - alpha*a;
a0 = 1 + alpha/a;
a1 = -2*cw;
a2 = 1 - alpha/a;
peakR[f_] := Module[{w = N[2 Pi f/sr, 16], z},
  z = Exp[-I w];
  (b0 + b1 z + b2 z^2)/(a0 + a1 z + a2 z^2)
];
routeL[f_] := N[10^(gainL/20), 16]*Exp[-I N[2 Pi f delayL/1000, 16]];
routeR[f_] := N[10^(gainR/20), 16]*Exp[-I N[2 Pi f delayR/1000, 16]]*peakR[f];

plantS1 = {0.5, 0.25};
plantS2 = {0.5, -0.25, 0.125};
plantDtft[taps_, f_] := Sum[taps[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, Length[taps] - 1}];

trials = {<|"source" -> "L", "gains" -> <|"L" -> 1.|>|>,
  <|"source" -> "LplusR", "gains" -> <|"L" -> 1., "R" -> 0.5|>|>};
seats = {<|"seat" -> "seat1", "taps" -> plantS1|>,
  <|"seat" -> "seat2", "taps" -> plantS2|>};

predict[trial_, seat_] := Module[{g = trial["gains"], p = seat["taps"]},
  Table[
    g["L"]*routeL[f]*plantDtft[p, f] +
      If[KeyExistsQ[g, "R"], g["R"]*routeR[f]*plantDtft[p, f], 0.],
    {f, grid}]
];
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};
predictions = Flatten[Table[
  Module[{sums = predict[trial, seat]},
    <|"source" -> trial["source"], "seat" -> seat["seat"],
      "sum_re_im" -> (pair /@ sums),
      "mag_db" -> (N[20*Log10[Abs[#]] + offsetDb, 16] & /@ sums),
      "phase_deg" -> (N[Arg[#]*180/Pi, 16] & /@ sums)|>],
  {trial, trials}, {seat, seats}], 1];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rc02-route-matrix-prediction.v1",
  "family" -> "RC02",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 6-point grid in Hz covering the declared band",
  "units" -> <|"transfer" -> "complex-linear", "magnitude" -> "dB", "phase" -> "degrees"|>,
  "calibration" -> "relative digital transfer; offset added, never fitted",
  "routes" -> {
    <|"input" -> "L", "output" -> "sub", "gain_db" -> gainL, "delay_ms" -> delayL|>,
    <|"input" -> "R", "output" -> "sub", "gain_db" -> gainR, "delay_ms" -> delayR,
      "peak" -> <|"filter_type" -> "peak", "freq_hz" -> fc, "q" -> q, "db_gain" -> peakDb|>|>},
  "plants" -> {
    <|"output" -> "sub", "seat" -> "seat1", "taps" -> (N[#, 16] & /@ plantS1)|>,
    <|"output" -> "sub", "seat" -> "seat2", "taps" -> (N[#, 16] & /@ plantS2)|>},
  "output" -> "sub",
  "magnitude_offset_db" -> offsetDb,
  "grid_hz" -> grid,
  "trial_gains" -> {<|"source" -> "L", "gains" -> <|"L" -> 1.|>|>,
    <|"source" -> "LplusR", "gains" -> <|"L" -> 1., "R" -> 0.5|>|>},
  "predictions" -> predictions
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
