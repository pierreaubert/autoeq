(* autoeq-qa case autoeq-qa.rw03-dsp-fir-convolution.v1 (catalogue family RW03).
   Independent delivered-chain realization with a convolution FIR: the
   serial response is gain * DTFT(taps) * delay-phase, with
   H_fir(f) = Sum_n taps[n] Exp[-I 2 Pi f n/sr] (direct sum, no FFT)
   and delay-phase Exp[-I 2 Pi f tau]. The oracle never calls
   RealizedDsp; it restates the cascade from the plugin definitions,
   so it catches double-applied delay, dropped FIR gain/mix, and
   sidecar tap truncation. *)

Off[General::munfl];
sr = 48000.;
gainDb = 6.;
taps = {1., 0.5, -0.25};
delayMs = 0.25;
freqs = {50., 100., 500., 1000., 5000., 10000., 20000.};

gain = N[10^(gainDb/20), 16];
fir[f_] := Sum[taps[[i + 1]]*Exp[-I N[2 Pi f i/sr, 16]], {i, 0, Length[taps] - 1}];
response[f_] := gain*fir[f]*Exp[-I N[2 Pi f (delayMs/1000.), 16]];

complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw03-dsp-fir-convolution.v1",
  "family" -> "RW03",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 7-point grid, Hz",
  "units" -> "linear magnitude, radians; taps dimensionless",
  "calibration" -> "tap zero at time zero; mix 1, convolution gain 0 dB",
  "gain_db" -> gainDb,
  "fir_taps" -> taps,
  "ir_file" -> "rw03_fir",
  "delay_ms" -> delayMs,
  "freqs_hz" -> freqs,
  "response_re_im" -> (complexPair[response[#]] & /@ freqs)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
