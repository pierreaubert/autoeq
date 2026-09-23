(* autoeq-qa case autoeq-qa.re14-combined-curves.v1 (catalogue family RE14).
   Independent per-seat complex sums for the MSO transfer matrix: each
   sub path applies gain, polarity, delay slope and (sub 0) one RBJ
   allpass, then seats sum coherently. The oracle replicates the exact
   log-spaced evaluation grid (base points plus merged measurement knots)
   and the closed-form path factors; weighted mean and population
   variance across seats follow from the summed seat SPLs. *)

Off[General::munfl];
sr = 48000.;
measFreqs = {20., 26., 34., 44., 57., 74., 96., 124., 150., 175., 200.};
nBase = 80;
logMin = N[Log10[20.], 16];
logMax = N[Log10[200.], 16];
base = Table[N[10^(logMin + (logMax - logMin)*(i - 1)/(nBase - 1)), 16], {i, nBase}];
dedup[list_] := Module[{out = {}},
   Do[If[out == {} || Abs[x - Last[out]] >= 10^-8, AppendTo[out, x]], {x, list}];
   out];
grid = dedup[Sort[Join[base, measFreqs]]];
(* per-sub/per-seat measured {spl, phase} (flat: interpolation-exact) *)
m00 = {82., 5.}; m01 = {80., -5.};
m10 = {78., 30.}; m11 = {79., -40.};
gainsDb = {1.5, -2.5};
delaysMs = {0.5, 1.25};
pols = {1., -1.};
(* RBJ allpass on sub 0: fc=60, Q=1 *)
w0 = N[2 Pi 60./sr, 16];
al = N[Sin[w0]/2, 16];
cs = N[Cos[w0], 16];
b0 = 1 - al; b1 = -2 cs; b2 = 1 + al;
a0 = 1 + al; a1n = -2 cs; a2 = 1 - al;
allpass[f_] := Module[{z = Exp[-I N[2 Pi f/sr, 16]]},
   ((b0 + b1 z + b2 z^2)/a0)/(1 + (a1n/a0) z + (a2/a0) z^2)];
pathGain[sub_, f_] :=
  N[10^(gainsDb[[sub]]/20), 16] * pols[[sub]] *
   Exp[-I N[2 Pi f delaysMs[[sub]]/1000., 16]] *
   If[sub == 1, allpass[f], 1];
measPhasor[m_, f_] :=
  N[10^(m[[1]]/20), 16] * Exp[I N[m[[2]] Degree, 16]];
seatZ[seatM0_, seatM1_] := Table[
   pathGain[1, f] measPhasor[seatM0, f] + pathGain[2, f] measPhasor[seatM1, f],
   {f, grid}];
z0 = seatZ[m00, m10];
z1 = seatZ[m01, m11];
splOf[z_] := N[20 Log10[Max[Abs[z], 10^-12]], 16];
argOf[z_] := N[Arg[z]/Degree, 16];
s0 = splOf[#] & /@ z0;
s1 = splOf[#] & /@ z1;
w = {0.7, 0.3};
wmean = Table[N[(w[[1]] s0[[i]] + w[[2]] s1[[i]])/(w[[1]] + w[[2]]), 16], {i, Length[grid]}];
wvar = Table[Module[{m = (s0[[i]] + s1[[i]])/2},
   N[((s0[[i]] - m)^2 + (s1[[i]] - m)^2)/2, 16]], {i, Length[grid]}];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re14-combined-curves.v1",
  "family" -> "RE14",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "grid" -> "log-spaced base (80 pts, 20-200 Hz) plus merged measurement knots, 1e-8 dedup",
  "units" -> <|"spl" -> "dB", "phase" -> "degrees", "delay" -> "ms"|>,
  "calibration" -> "mic-1",
  "meas_freqs_hz" -> measFreqs,
  "sub_seat_spl_db" -> {{82., 80.}, {78., 79.}},
  "sub_seat_phase_deg" -> {{5., -5.}, {30., -40.}},
  "gains_db" -> gainsDb,
  "delays_ms" -> delaysMs,
  "polarities" -> {False, True},
  "allpass_sub0" -> <|"fc_hz" -> 60., "q" -> 1.|>,
  "seat_weights" -> w,
  "seat_response_re_im" -> {complexPair[#] & /@ z0, complexPair[#] & /@ z1},
  "seat_spl_db" -> {s0, s1},
  "seat_phase_deg" -> {argOf[#] & /@ z0, argOf[#] & /@ z1},
  "weighted_mean_spl_db" -> wmean,
  "seat_variance_db2" -> wvar
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
