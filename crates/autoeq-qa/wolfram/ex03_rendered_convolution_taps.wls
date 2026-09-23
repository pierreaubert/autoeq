(* autoeq-qa case autoeq-qa.ex03-rendered-convolution-taps.v1 (catalogue family EX03).
   Independent realization of a half-sample fractional-delay unit
   impulse with the documented Blackman-windowed sinc kernel
   k(x) = sinc(x)*(0.42 + 0.5 Cos[Pi x/r] + 0.08 Cos[2 Pi x/r]),
   r = kernelHalf, DC-normalized, at fractional offset
   frac = shift - Floor[shift] with shift = delaySamples + pad.
   Linear convolution y = x*h and the DTFT
   H(f) = Sum_n h[n] Exp[-I 2 Pi f n/sr] are direct sums. The f32
   quantization budget 2^-24 Sum_n Abs[h[n]] follows from the IEEE
   binary32 unit roundoff, not from any implementation. Channel R
   carries an integer 5-sample delay of [1.0] (impulse at index 5).
   The Rust test realizes both channels with realize_gd_fir_delay,
   convolves/DTFTs independently, and checks the mapping plus budget. *)

Off[General::munfl];
rate = 48000.;
half = 64;
delayMs = N[0.5*1000/rate, 16];
pad = 64;
shift = N[delayMs*rate/1000, 16] + pad;
base = Floor[shift] - half;
frac = shift - Floor[shift];

sinc[x_] := If[Abs[x] < 1*^-12, N[1, 16], N[Sin[Pi x]/(Pi x), 16]];
blackman[x_, r_] := If[Abs[x] > r, 0.,
  N[0.42 + 0.5*Cos[Pi x/r] + 0.08*Cos[2 Pi x/r], 16]];
kernel[j_] := sinc[N[j - half - frac, 16]]*blackman[N[j - half - frac, 16], half];
dc = Sum[kernel[j], {j, 0, 2*half}];
taps = Table[N[kernel[j]/dc, 16], {j, 0, 2*half}];

x = {1., 2., 3., 4.};
y = ListConvolve[taps, x, {1, -1}, 0];
dtft[seq_, f_] := Sum[seq[[n + 1]]*Exp[-I N[2 Pi f n/rate, 16]], {n, 0, Length[seq] - 1}];
spots = {100., 1000., 10000.};
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};

f32bound = N[2^-24*Sum[Abs[taps[[n]]], {n, 1, Length[taps]}], 16];
rDelaySamples = 5;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ex03-rendered-convolution-taps.v1",
  "family" -> "EX03",
  "sample_rate_hz" -> rate,
  "grid" -> "tap indices from time zero; DTFT spots in Hz",
  "units" -> <|"taps" -> "linear-amplitude", "response" -> "complex-linear"|>,
  "calibration" -> "rendered convolution taps, unity DC by construction",
  "left" -> <|"delay_ms" -> delayMs, "pad_samples" -> pad,
    "base_index" -> base, "taps" -> taps|>,
  "input_x" -> x,
  "conv_y" -> (N[#, 16] & /@ y),
  "dtft_spots_hz" -> spots,
  "dtft_h_re_im" -> (pair[dtft[taps, #]] & /@ spots),
  "dtft_y_re_im" -> (pair[dtft[y, #]] & /@ spots),
  "f32_l1_bound" -> f32bound,
  "right" -> <|"delay_samples" -> rDelaySamples, "peak_index" -> rDelaySamples,
    "length" -> rDelaySamples + 1, "dc_gain" -> 1.|>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
