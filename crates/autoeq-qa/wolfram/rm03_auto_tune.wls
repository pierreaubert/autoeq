(* autoeq-qa case autoeq-qa.rm03-auto-tune.v1 (catalogue family RM03).
   Independent derivation of the deterministic auto-tune analysis from
   its documented contract: in-band mean removal, index-domain window-5
   mean smoothing, RMS/peak/dip of the smoothed deviations, strict-local
   peak (>= 3 dB) / dip (<= -4 dB) counting with 1/6-octave separation,
   3-dB-bandwidth feature Q, needed boost/cut clamps, main-channel
   filter-count arithmetic, Q-bound branch, and the zero-boost envelope
   anchored at max(Schroeder, F3). Compared against
   resolve_auto_optimizer_config on the same analytic curve. *)

Off[General::munfl];
freqs = {25., 35., 50., 70., 100., 140., 200., 280., 400., 560., 800., 1150.};
spl = {0., 0., 7., 9., 8., 0., 0., 0., -8., -10., -9., 0.};
minF = 25.; maxF = 1150.;
n = Length[freqs];
mean = N[Total[spl]/n, 16];
dev = Map[N[# - mean, 16] &, spl];
smooth = Table[
  Module[{lo = Max[i - 2, 1], hi = Min[i + 2, n]},
    N[Total[Take[dev, {lo, hi}]]/(hi - lo + 1), 16]],
  {i, 1, n}];
rms = N[Sqrt[Total[#^2 & /@ smooth]/n], 16];
maxPeak = N[Max[Join[{0.}, smooth]], 16];
maxDip = N[Min[Join[{0.}, smooth]], 16];
(* Strict-local extrema with 1/6-octave separation. *)
count = 0; lastF = -1.; qEst = -1.;
Do[
  Module[{f = freqs[[i]], v = smooth[[i]], prev = smooth[[i - 1]], next = smooth[[i + 1]], sep},
    sep = If[lastF < 0., 10.^9, N[Log[2, f/lastF], 16]];
    If[((v > prev && v > next && v >= 3.) || (v < prev && v < next && v <= -4.)) && sep >= 1./6.,
      count++; lastF = f;
      Module[{cAbs = Abs[v], hp, lo = f, hi = f, j},
        hp = Max[cAbs - 3., 0.];
        For[j = i - 1, j >= 1, j--, lo = freqs[[j]]; If[Abs[smooth[[j]]] <= hp, Break[]]];
        For[j = i + 1, j <= n, j++, hi = freqs[[j]]; If[Abs[smooth[[j]]] <= hp, Break[]]];
        If[hi > lo, qEst = Max[qEst, N[f/(hi - lo), 16]]];];];],
  {i, 2, n - 1}];
modalQ = If[qEst < 0., Null, qEst];
span = N[Max[Log[2, maxF/minF], 0.25], 16];
est = Ceiling[span*1.15];
est = Max[est, count + 1];
Which[rms >= 7., est += 3, rms >= 4.5, est += 2, rms >= 2.5, est += 1];
est += 1; (* Schroeder 250 Hz lies inside (25, 1150). *)
numFilters = Min[Max[est, 1], 12];
modalClamped = N[Min[Max[If[qEst < 0., 5., qEst], 3.], 10.], 16];
maxQ = N[Min[Max[modalClamped, 4.], 6.], 16];
minQ = 0.5;
(* Main channel: boost cap 4 dB, cut cap 12 dB. *)
neededBoost = N[Min[Max[-maxDip + 1., 2.], 4.], 16];
neededCut = N[Min[Max[maxPeak + 2., 6.], 12.], 16];
cutoff = 250.; release = N[Min[cutoff*1.25, maxF], 16];
envelope = {{minF, 0.}, {cutoff, 0.}, {release, neededBoost}, {maxF, neededBoost}};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rm03-auto-tune.v1",
  "family" -> "RM03",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "index-domain curve analysis on a declared grid; no sampling involved",
  "grid_hz" -> freqs,
  "units" -> "dB SPL deviations; Q dimensionless; envelope (Hz, dB)",
  "calibration" -> "mean-removed deviations; no SPL calibration",
  "curve_spl_db" -> spl,
  "context" -> <|
    "is_sub_channel" -> False, "effective_min_hz" -> minF,
    "effective_max_hz" -> maxF, "detected_f3_hz" -> 65.,
    "schroeder_hz" -> 250., "target_tilt_active" -> False,
    "broadband_enabled" -> False|>,
  "smoothed_deviations_db" -> smooth,
  "rms_db" -> rms,
  "max_peak_db" -> maxPeak,
  "max_dip_db" -> maxDip,
  "problem_count" -> count,
  "modal_q_estimate" -> modalQ,
  "octave_span" -> span,
  "num_filters" -> numFilters,
  "min_q" -> minQ,
  "max_q" -> maxQ,
  "max_db" -> neededBoost,
  "min_db" -> N[-neededCut, 16],
  "boost_envelope" -> envelope
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
