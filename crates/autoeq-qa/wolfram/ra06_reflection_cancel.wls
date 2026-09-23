(* autoeq-qa case autoeq-qa.ra06-reflection-cancel.v1 (catalogue family RA06).
   Independent low-passed reflection-cancellation transfer and recurrence:
   y[n] = x[n] - g LP(x[n-d]) with g = 0.4, d = 24 samples, and LP a 4th-
   order Butterworth low-pass at 500 Hz (48 kHz) built from the published
   pole formula Qk = 1/(2 cos((2k-1) pi/8)) and the RBJ cookbook low-pass
   design equations. H(f) = 1 - g exp(-j 2 pi f d/sr) LP(f) is evaluated as
   a direct SOS product; the 64-sample impulse response comes from an
   independent direct-form recurrence. Compared against the same cascade
   built with autoeq_core::iir (Biquad/BiquadFilterType/peq_butterworth_q)
   filtered sample-by-sample (recurrence), by explicit convolution with the
   measured LP impulse, and by SOS transfer (H). Samples are linear
   amplitude; no measurement calibration is involved (analytic plant). *)

Off[General::munfl];
sr = 48000.; fc = 500.; order = 4; g = 0.4; d = 24;
qs = N[Table[1/(2*Cos[(2*k - 1)*Pi/(2*order)]), {k, 1, order/2}], 16];
w0 = N[2*Pi*fc/sr, 16]; cw = N[Cos[w0], 16]; sn = N[Sin[w0], 16];
sos = Table[
   alpha = N[sn/(2*qs[[k]]), 16];
   b0 = N[(1 - cw)/2, 16]; b1 = N[1 - cw, 16]; b2 = N[(1 - cw)/2, 16];
   a0 = N[1 + alpha, 16]; a1 = N[-2*cw, 16]; a2 = N[1 - alpha, 16];
   N[{b0, b1, b2, a0, a1, a2}, 16], {k, 1, Length[qs]}];
sosN = Table[c/c[[4]], {c, sos}];

lpSection[sec_, f_] := Module[{z = N[Exp[-I*2*Pi*f/sr], 16]},
  (sec[[1]] + sec[[2]]*z + sec[[3]]*z^2)/(1 + sec[[5]]*z + sec[[6]]*z^2)];
lpResp[f_] := Product[lpSection[sosN[[k]], f], {k, 1, Length[sosN]}];
hResp[f_] := N[1 - g*Exp[-I*N[2*Pi*f*d/sr, 16]]*lpResp[f], 16];

probes = {20., 50., 100., 200., 500., 1000., 2000., 5000.};
hPairs = Table[{N[Re[hResp[f]], 16], N[Im[hResp[f]], 16]}, {f, probes}];

nt = 64;
x = Table[If[t == 1, 1., 0.], {t, 1, nt}];
stepSection[xin_, c_] := Module[{y = Table[0., {Length[xin]}]},
  Do[y[[t]] = c[[1]]*xin[[t]]
     + If[t > 1, c[[2]]*xin[[t - 1]], 0.]
     + If[t > 2, c[[3]]*xin[[t - 2]], 0.]
     - If[t > 1, c[[5]]*y[[t - 1]], 0.]
     - If[t > 2, c[[6]]*y[[t - 2]], 0.], {t, 1, Length[xin]}];
  y];
lpOut = Fold[stepSection, x, sosN];
lpD = Table[If[t - d >= 1, lpOut[[t - d]], 0.], {t, 1, nt}];
yTime = N[x - g*lpD, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra06-reflection-cancel.v1",
  "family" -> "RA06",
  "sample_rate_hz" -> sr,
  "grid_hz" -> probes,
  "grid_units" -> "Hz",
  "sample_units" -> "linear amplitude",
  "calibration" -> "analytic plant; unit-impulse input, no measurement",
  "lp_cutoff_hz" -> fc,
  "lp_order" -> order,
  "lp_design" -> "Butterworth-RBJ-lowpass",
  "cancellation_gain" -> g,
  "delay_samples" -> d,
  "butterworth_q" -> qs,
  "sos_b0_b1_b2_a0_a1_a2" -> sos,
  "response_re_im" -> hPairs,
  "impulse_input" -> x,
  "impulse_response" -> yTime
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
