(* autoeq-qa case autoeq-qa.re23-ctc-regularized.v1 (catalogue family RE23).
   Independent regularized 2x2 crosstalk-cancellation solve: per-bin
   closed-form Tikhonov normal equations (H^H H + beta I) F = H^H with
   the frequency-dependent beta branches, elementwise and row-sum
   headroom (both no-ops here, verified), SVD condition number, ideal
   reconstruction error, realized four-FIR synthesis through the exact
   bulk-delay/mirror/IDFT prescription, and delivered-response
   residual/headroom metrics. Direct sums only. *)

Off[General::munfl];
sr = 48000.;
n = 16;
lat = 8;
h11 = N[0.9 + 0.1 I, 16]; h12 = N[0.25 - 0.15 I, 16];
h21 = N[0.2 + 0.2 I, 16]; h22 = N[0.8 - 0.1 I, 16];
hh = {{h11, h12}, {h21, h22}};
betaLf = N[10^(-30./20), 16];
betaMid = N[10^(-40./20), 16];
betaHf = N[10^(-20./20), 16];
maxGain = N[10^(24./20), 16];
betaOf[k_] := If[k == 0, betaLf, If[k <= 2, betaMid, betaHf]];
ident = {{1, 0}, {0, 1}};
sol[k_] := Module[{g, f},
   g = ConjugateTranspose[hh] . hh + betaOf[k] ident;
   f = Inverse[g] . ConjugateTranspose[hh] . ident;
   f];
clamp[z_] := If[Abs[z] > maxGain && Abs[z] > 0, z maxGain/Abs[z], z];
headroom[m_] := Module[{rows},
   rows = Table[
     Module[{r = m[[s]], nrm = Total[Abs[m[[s]]]]},
      If[nrm > maxGain && nrm > 0, r maxGain/nrm, r]],
     {s, 2}];
   rows];
ff = Table[headroom[Map[clamp, sol[k], {2}]], {k, 0, 8}];
freqs = Table[N[k sr/n, 16], {k, 0, 8}];
betas = betaOf[#] & /@ Range[0, 8];
sv = N[SingularValueList[hh, Tolerance -> 0], 16];
cond = N[Max[sv]/Min[sv], 16];
reconErr = Table[N[Mean[Abs[Flatten[hh . ff[[k + 1]] - ident]]^2], 16], {k, 0, 8}];
meanRecon = N[Mean[reconErr], 16];
maxF = N[Max[Abs[Flatten[ff]]], 16];
maxFiltGain = N[20 Log10[maxF], 16];
maxRow = N[Max[Table[Total[Abs[ff[[k + 1, s]]]], {k, 0, 8}, {s, 2}]], 16];
maxSumGain = N[20 Log10[maxRow], 16];
(* realized FIR: bulk delay, Hermitian mirror, unnormalized IDFT, /N *)
firOf[s_, e_] := Module[{x, y, taps},
   x = Table[N[ff[[k + 1, s, e]] Exp[-I 2 Pi k lat/n], 16], {k, 0, 8}];
   y = Table[0, {n}];
   Do[y[[k + 1]] = x[[k + 1]], {k, 0, 8}];
   Do[y[[n - k + 1]] = Conjugate[x[[k + 1]]], {k, 1, 7}];
   taps = Table[N[Re[Sum[y[[j + 1]] Exp[I 2 Pi j t/n], {j, 0, n - 1}]]/n, 16], {t, 0, n - 1}];
   taps];
taps = Table[firOf[s, e], {s, 1, 2}, {e, 1, 2}];
(* delivered response: unnormalized forward DFT, latency undo *)
specOf[t_] := Table[N[Sum[t[[m + 1]] Exp[-I 2 Pi k m/n], {m, 0, n - 1}], 16], {k, 0, 8}];
specs = Table[specOf[taps[[s, e]]], {s, 1, 2}, {e, 1, 2}];
deliv[k_] := Module[{fh, dd},
   fh = Table[specs[[s, e, k + 1]], {s, 1, 2}, {e, 1, 2}];
   dd = hh . fh;
   dd = Table[N[dd[[r, c]] Exp[I 2 Pi k lat/n], 16], {r, 1, 2}, {c, 1, 2}];
   dd];
tgtErr = Flatten[Table[
    Abs[deliv[k][[e, e]] - 1], {k, 0, 8}, {e, 1, 2}]];
xtalk = Flatten[Table[
    {Abs[deliv[k][[1, 2]]], Abs[deliv[k][[2, 1]]]}, {k, 0, 8}]];
meanTgt = N[Sqrt[Mean[tgtErr^2]], 16];
worstTgt = N[Max[tgtErr], 16];
meanX = N[Sqrt[Mean[xtalk^2]], 16];
worstX = N[Max[xtalk], 16];
meanXDb = N[20 Log10[Max[meanX, 10^-12]], 16];
worstXDb = N[20 Log10[Max[worstX, 10^-12]], 16];
ampDb[v_] := N[20 Log10[Max[v, 10^-12]], 16];
bal = Table[
   Abs[ampDb[Abs[deliv[k][[1, 1]]]] - ampDb[Abs[deliv[k][[2, 2]]]]],
   {k, 0, 8}];
meanBal = N[Mean[bal], 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re23-ctc-regularized.v1",
  "family" -> "RE23",
  "sample_rate_hz" -> sr,
  "fir_taps" -> n,
  "latency_samples" -> lat,
  "grid" -> "9 half-spectrum bins at k*3000 Hz (N=16)",
  "units" -> <|"transfer" -> "linear", "gain" -> "dB", "taps" -> "linear"|>,
  "calibration" -> "synthetic",
  "plant_re_im" -> {complexPair[h11], complexPair[h12], complexPair[h21], complexPair[h22]},
  "beta_db" -> <|"lf" -> -30., "mid" -> -40., "hf" -> -20.|>,
  "max_gain_db" -> 24.,
  "bin_freqs_hz" -> freqs,
  "bin_betas" -> betas,
  "inverse_re_im_per_bin" -> ((complexPair[#] & /@ Flatten[#]) & /@ ff),
  "condition_number" -> cond,
  "recon_error_per_bin" -> reconErr,
  "mean_reconstruction_error" -> meanRecon,
  "max_filter_gain_db" -> maxFiltGain,
  "max_electrical_sum_gain_db" -> maxSumGain,
  "fir_taps_row_major" -> taps,
  "mean_target_error" -> meanTgt,
  "worst_target_error" -> worstTgt,
  "mean_crosstalk_db" -> meanXDb,
  "worst_crosstalk_db" -> worstXDb,
  "mean_channel_balance_db" -> meanBal
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
