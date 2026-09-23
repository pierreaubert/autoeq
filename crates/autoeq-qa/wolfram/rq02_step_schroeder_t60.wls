(* autoeq-qa case autoeq-qa.rq02-step-schroeder-t60.v1 (catalogue family RQ02).
   Independent IR/step/envelope/decay evaluation for the exponential
   fixture h[n] = r^n: the step response is the explicit cumulative
   sum; the Schroeder decay is the reverse-energy integral
   10 Log10[Sum_{k>=n} h[k]^2 / Sum_all h^2] normalized to 0 dB at
   direct; the T60 fit is the textbook least-squares line over the
   declared -5..-35 dB segment with T60 = -60/slope and R-squared;
   the Hilbert envelope of the bin-centered cosine uses the
   documented unnormalized-forward / 1/N-inverse FFT convention with
   doubled positive frequencies. The Rust test calls step_response,
   schroeder_decay_db, fit_t60, and hilbert_envelope. *)

Off[General::munfl];
r = 0.99;
n = 512;
rate = 48000.;
ir = Table[N[r^k, 16], {k, 0, n - 1}];
step = Accumulate[ir];
total = Total[ir^2];
tail = Reverse[Accumulate[Reverse[ir^2]]];
decay = (N[10*Log10[Max[#/total, 1*^-30]], 16] & /@ tail);
times = Table[N[k/rate, 16], {k, 0, n - 1}];
fromDb = -5.;
toDb = -35.;
sel = Select[Transpose[{times, decay}], #[[2]] <= fromDb && #[[2]] >= toDb &];
m = Length[sel];
meanT = Total[sel[[All, 1]]]/m;
meanD = Total[sel[[All, 2]]]/m;
slope = Total[(#[[1]] - meanT)*(#[[2]] - meanD) & /@ sel]/
  Total[(#[[1]] - meanT)^2 & /@ sel];
intercept = meanD - slope*meanT;
ssTot = Total[(#[[2]] - meanD)^2 & /@ sel];
ssRes = Total[(#[[2]] - (slope*#[[1]] + intercept))^2 & /@ sel];
r2 = 1 - ssRes/ssTot;
t60 = -60/slope;

hn = 256;
cycles = 8;
cosine = Table[N[Cos[2 Pi cycles k/hn], 16], {k, 0, hn - 1}];
dft = Table[Sum[cosine[[j + 1]]*Exp[-I N[2 Pi j k/hn, 16]], {j, 0, hn - 1}], {k, 0, hn - 1}];
analytic = Table[
  Which[k == 0, dft[[1]], EvenQ[hn] && k == hn/2, dft[[k + 1]],
    k < hn/2, 2*dft[[k + 1]], True, 0.], {k, 0, hn - 1}];
inv = Table[(1/N[hn, 16])*Sum[analytic[[j + 1]]*Exp[I N[2 Pi j k/hn, 16]], {j, 0, hn - 1}], {k, 0, hn - 1}];
envelope = (N[Abs[#], 16] & /@ inv);

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rq02-step-schroeder-t60.v1",
  "family" -> "RQ02",
  "sample_rate_hz" -> rate,
  "grid" -> "tap indices from time zero; decay times in seconds",
  "units" -> <|"samples" -> "linear amplitude", "decay" -> "dB", "t60" -> "seconds"|>,
  "calibration" -> "synthetic exponential fixture; decay normalized to 0 dB at direct",
  "decay_ratio" -> r,
  "taps" -> n,
  "ir" -> ir,
  "step" -> (N[#, 16] & /@ step),
  "decay_db" -> decay,
  "fit_from_db" -> fromDb,
  "fit_to_db" -> toDb,
  "fit_points" -> m,
  "t60_seconds" -> N[t60, 16],
  "r_squared" -> N[r2, 16],
  "hilbert" -> <|"taps" -> hn, "cycles" -> cycles,
    "cosine" -> cosine, "envelope" -> envelope|>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
