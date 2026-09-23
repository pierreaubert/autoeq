(* autoeq-qa case autoeq-qa.ra03-mode-null-masks.v1 (catalogue family RA03).
   Independent FDW-free mode/null analysis on a synthetic resonance+notch:
   a Lorentzian peak (60 Hz, +10 dB, half-width 3 Hz) and a Lorentzian dip
   (140 Hz, -8 dB, half-width 8.75 Hz) on 80 dB SPL over a 400-point linear
   grid 20-300 Hz. Peak prominence uses the median SPL in a +/-1 octave
   window (center excluded); Q comes from linear-interpolated -3/+3 dB
   crossings with edge-bounded mirroring; the null-suppression mask is a
   raised-cosine dip to 0 inside each null half-bandwidth (taper 0.5).
   Compared against roomeq_analysis::impulse_analysis::{detect_room_modes,
   detect_narrow_nulls, build_null_suppression_mask}. No time sampling is
   involved; units are dB SPL over Hz. *)

Off[General::munfl];
n = 400; fLo = 20.; fHi = 300.;
freqs = N[Table[fLo + (fHi - fLo)*i/(n - 1), {i, 0, n - 1}], 16];
peakC = 60.; peakH = 10.; peakHW = 3.;
dipC = 140.; dipD = 8.; dipHW = 8.75;
spl = N[80. + peakH/(1 + ((# - peakC)/peakHW)^2) - dipD/(1 + ((# - dipC)/dipHW)^2) & /@ freqs, 16];

schroeder = 300.; minQ = 3.; minProm = 3.; minDepth = 4.; taper = 0.5;
ratio = N[2^(1/6), 16];

neighbors[i_] := Module[{fc = freqs[[i]], lo, hi, nb},
  lo = fc/ratio; hi = fc*ratio;
  nb = Select[Range[n], # != i && freqs[[#]] >= lo && freqs[[#]] <= hi &];
  If[nb == {}, DeleteDuplicates[{Max[i - 1, 1], Min[i + 1, n]}], nb]];
isPeak[i_] := And @@ Table[spl[[i]] > spl[[j]], {j, neighbors[i]}];
isDip[i_] := And @@ Table[spl[[i]] < spl[[j]], {j, neighbors[i]}];
baseline[i_] := Module[{fc = freqs[[i]], vals},
  vals = Table[If[j != i && freqs[[j]] >= fc/2 && freqs[[j]] <= 2*fc, spl[[j]], Nothing], {j, 1, n}];
  If[vals == {}, spl[[i]], N[Median[vals], 16]]];

crossLeft[i_, thr_, wantAbove_] := Module[{j = i - 1, t, denom},
  While[j >= 1 && If[wantAbove, spl[[j]] > thr, spl[[j]] < thr], j--];
  If[j < 1, None,
   denom = spl[[j + 1]] - spl[[j]];
   If[Abs[denom] < 1*^-12, freqs[[j]],
    t = N[(thr - spl[[j]])/denom, 16];
    N[freqs[[j]] + Clip[t, {0, 1}]*(freqs[[j + 1]] - freqs[[j]]), 16]]]];
crossRight[i_, thr_, wantAbove_] := Module[{j = i + 1, t, denom},
  While[j <= n && If[wantAbove, spl[[j]] > thr, spl[[j]] < thr], j++];
  If[j > n, None,
   denom = spl[[j]] - spl[[j - 1]];
   If[Abs[denom] < 1*^-12, freqs[[j]],
    t = N[(thr - spl[[j - 1]])/denom, 16];
    N[freqs[[j - 1]] + Clip[t, {0, 1}]*(freqs[[j]] - freqs[[j - 1]]), 16]]]];
bandwidth[fc_, flo_, fhi_] := Which[
  flo =!= None && fhi =!= None, N[fhi - flo, 16],
  flo =!= None, With[{v = fc - flo}, N[v + Min[v, freqs[[n]] - fc], 16]],
  fhi =!= None, With[{v = fhi - fc}, N[v + Min[v, fc - freqs[[1]]], 16]],
  True, N[freqs[[n]] - freqs[[1]], 16]];

rawModes = Reap[Do[
    If[freqs[[i]] <= schroeder && isPeak[i],
     prom = N[spl[[i]] - baseline[i], 16];
     If[prom >= minProm,
      thr = N[spl[[i]] - 3., 16];
      bw = bandwidth[freqs[[i]], crossLeft[i, thr, True], crossRight[i, thr, True]];
      q = If[bw > 0, N[freqs[[i]]/bw, 16], 0.];
      If[q >= minQ, Sow[N[{freqs[[i]], q, prom}, 16]]]]], {i, 2, n - 1}]][[2]];
modes = If[rawModes == {}, {}, rawModes[[1]]];
rawNulls = Reap[Do[
    If[isDip[i],
     depth = N[baseline[i] - spl[[i]], 16];
     If[depth >= minDepth,
      thr = N[spl[[i]] + 3., 16];
      bw = bandwidth[freqs[[i]], crossLeft[i, thr, False], crossRight[i, thr, False]];
      q = If[bw > 0, N[freqs[[i]]/bw, 16], 0.];
      If[q >= minQ, Sow[N[{freqs[[i]], q, depth}, 16]]]]], {i, 2, n - 1}]][[2]];
nulls = If[rawNulls == {}, {}, rawNulls[[1]]];

mask = Table[1., {n}];
Do[
  fq = e[[1]]; fqQ = e[[2]];
  bw = fq/fqQ; fIn = bw*0.5*(1 - taper); fOut = bw*0.5;
  Do[df = Abs[freqs[[i]] - fq];
   If[df <= fOut,
    w = If[df <= fIn, 0., N[0.5*(1 - Cos[Pi*(df - fIn)/(fOut - fIn)]), 16]];
    mask[[i]] = Min[mask[[i]], w]], {i, 1, n}],
  {e, nulls}];
mask = N[mask, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra03-mode-null-masks.v1",
  "family" -> "RA03",
  "sample_rate_hz" -> Null,
  "sampling_note" -> "no time sampling; linear frequency grid only",
  "grid_hz" -> freqs,
  "grid_units" -> "Hz",
  "spl_db" -> spl,
  "spl_units" -> "dB SPL",
  "spl_calibration" -> "synthetic 80 dB reference with Lorentzian peak/dip",
  "peak_center_hz" -> peakC,
  "peak_height_db" -> peakH,
  "peak_halfwidth_hz" -> peakHW,
  "dip_center_hz" -> dipC,
  "dip_depth_db" -> dipD,
  "dip_halfwidth_hz" -> dipHW,
  "schroeder_hz" -> schroeder,
  "min_mode_q" -> minQ,
  "min_prominence_db" -> minProm,
  "min_null_depth_db" -> minDepth,
  "mask_taper_frac" -> taper,
  "modes_f_q_prom" -> modes,
  "nulls_f_q_depth" -> nulls,
  "null_mask" -> mask
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
