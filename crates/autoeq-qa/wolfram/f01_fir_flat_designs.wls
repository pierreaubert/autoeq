(* autoeq-qa case autoeq-qa.f01-fir-flat-designs.v1 (catalogue family F01).
   Independent flat-response FIR designs through the public autoeq-fir
   API contract (math-audio 664498b pipeline stages, restated here from
   the versioned algorithm contract, not transliterated):
   - Kirkeby magnitude-only correction of a flat 80 dB measurement
     against a flat 80 dB target: relative magnitude 1, regularized
     inverse rel/(rel^2+beta^2) with beta = 1/(2*10^(15/20)), clamped to
     [1/10, 10^(15/20)], cosine-free linear edge transition of the
     documented transition width, Hermitian completion, inverse DFT
     with 1/N scaling (FourierParameters -> {1,-1}), causal centering
     shift, Hann window, truncation to 64 taps.
   - Minimum-phase flat 0 dB design: unit spectrum is already
     minimum-phase, so taps are the exact unit impulse.
   - Linear-phase flat 0 dB design: centered impulse scaled by the
     Blackman window center value.
   DTFT spot checks use the direct finite sum
   H(f) = Sum_n h[n] Exp[-I 2 Pi f n/sr] with tap zero at time zero. *)

Off[General::munfl];
sr = 48000.;
nTaps = 64;
fftLen = 65536;
numBins = 32769;
step = N[sr/fftLen, 16];

(* Kirkeby desired correction spectrum on the FFT grid. *)
maxBoostDb = 15.; maxCutDb = 20.;
maxBoostLin = N[10^(maxBoostDb/20), 16];
maxCutLin = N[10^(maxCutDb/20), 16];
beta = N[1/(2*maxBoostLin), 16];
minF = 100.; maxF = 10000.;
binW = step;
bandHalf = Max[N[(maxF - minF)*0.5, 16], binW];
lowerRoom = Max[minF, binW];
width = Min[8*binW, bandHalf, lowerRoom];
relMag = 1.;
regMag = N[relMag/(relMag^2 + beta^2), 16];
limited = N[Clip[regMag, {1/maxCutLin, maxBoostLin}], 16];
trans[f_] := Which[
  f < minF, Clip[N[(f - (minF - width))/width, 16], {0., 1.}],
  f > maxF, 1. - Clip[N[(f - maxF)/width, 16], {0., 1.}],
  True, 1.];
cMag[f_] := N[1 + trans[f]*(limited - 1), 16];
cList = Table[cMag[N[i*step, 16]], {i, 0, numBins - 1}];
full = Join[cList, Reverse[Conjugate[cList[[2 ;; numBins - 1]]]]];
ir = Re[InverseFourier[full, FourierParameters -> {1, -1}]];
shift = fftLen/2;
start0 = shift - nTaps/2;
seg = Take[RotateRight[ir, shift], {start0 + 1, start0 + nTaps}];
hann = Table[N[0.5*(1 - Cos[2 Pi*(i - 1)/(nTaps - 1)]), 16], {i, 1, nTaps}];
kirkebyTaps = N[seg*hann, 16];

(* Analytic minimum/linear-phase flat designs. *)
minTaps = Table[If[i == 1, N[1., 16], N[0., 16]], {i, 1, nTaps}];
black = Table[N[0.42 - 0.5*Cos[2 Pi*(i - 1)/(nTaps - 1)]
  + 0.08*Cos[4 Pi*(i - 1)/(nTaps - 1)], 16], {i, 1, nTaps}];
center1 = nTaps/2 + 1;
linTaps = Table[If[i == center1, black[[i]], N[0., 16]], {i, 1, nTaps}];

(* Direct-sum DTFT spot checks of the golden taps. *)
spots = {0., 100., 1000., 5000., 10000., 20000.};
dtft[taps_, f_] := Sum[taps[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, nTaps - 1}];
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};
kirkebyResp = (pair[dtft[kirkebyTaps, #]] & /@ spots);
minResp = (pair[dtft[minTaps, #]] & /@ spots);
linResp = (pair[dtft[linTaps, #]] & /@ spots);

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.f01-fir-flat-designs.v1",
  "family" -> "F01",
  "sample_rate_hz" -> sr,
  "units" -> <|"taps" -> "linear-amplitude", "response" -> "complex-linear"|>,
  "calibration" -> <|"measurement_spl_db" -> 80., "target_spl_db" -> 80.,
    "flat_design_spl_db" -> 0.|>,
  "n_taps" -> nTaps,
  "fft_len" -> fftLen,
  "curve_freqs_hz" -> {20., 42., 90., 200., 430., 950., 4200., 20000.},
  "kirkeby_min_hz" -> minF,
  "kirkeby_max_hz" -> maxF,
  "kirkeby_regularized_inband" -> limited,
  "kirkeby_transition_width_hz" -> N[width, 16],
  "kirkeby_taps" -> kirkebyTaps,
  "minphase_taps" -> minTaps,
  "linear_taps" -> linTaps,
  "blackman_center" -> black[[center1]],
  "dtft_freqs_hz" -> spots,
  "kirkeby_response_re_im" -> kirkebyResp,
  "minphase_response_re_im" -> minResp,
  "linear_response_re_im" -> linResp
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
