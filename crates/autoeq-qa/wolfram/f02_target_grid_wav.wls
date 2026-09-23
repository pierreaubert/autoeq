(* autoeq-qa case autoeq-qa.f02-target-grid-wav.v1 (catalogue family F02).
   Independent target-grid resolution (piecewise-linear interpolation
   on the linear frequency axis with edge hold) plus a delivered-WAV
   fixture: the taps below are the fixture of record, and their DTFT
   H(f) = Sum_n h[n] Exp[-I 2 Pi f n/sr] (tap zero at time zero) is the
   reference for samples decoded from the file the Rust test writes.
   f32 quantization is budgeted in the comparison test as
   Sum_n Abs[delta_h[n]], not absorbed into a loose tolerance. *)

Off[General::munfl];
sr = 48000.;
measGrid = {20., 100., 1000., 10000., 20000.};
tgtFreqs = {20., 500., 20000.};
tgtSpl = {80., 86., 80.};

interp[f_] := Module[{pos},
  pos = LengthWhile[tgtFreqs, # < f &];
  Which[
    pos == 0, tgtSpl[[1]],
    pos >= Length[tgtFreqs], tgtSpl[[-1]],
    True, tgtSpl[[pos]] + (tgtSpl[[pos + 1]] - tgtSpl[[pos]])*
      (f - tgtFreqs[[pos]])/(tgtFreqs[[pos + 1]] - tgtFreqs[[pos]])
  ]];
resolved = (N[interp[#], 16] & /@ measGrid);

wavTaps = {0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.05};
wavSpots = {0., 1000., 10000., 24000.};
wavResp[f_] := Sum[wavTaps[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, Length[wavTaps] - 1}];
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.f02-target-grid-wav.v1",
  "family" -> "F02",
  "sample_rate_hz" -> sr,
  "units" -> <|"spl" -> "dB", "taps" -> "linear-amplitude",
    "response" -> "complex-linear"|>,
  "calibration" -> "target-resolution-relative-dB-wav-taps-absolute",
  "measurement_grid_hz" -> measGrid,
  "target_freqs_hz" -> tgtFreqs,
  "target_spl_db" -> tgtSpl,
  "resolved_target_db" -> resolved,
  "wav_taps" -> (N[#, 16] & /@ wavTaps),
  "wav_rate_hz" -> sr,
  "wav_channels" -> 1,
  "wav_bits" -> 32,
  "wav_freqs_hz" -> wavSpots,
  "wav_response_re_im" -> (pair[wavResp[#]] & /@ wavSpots)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
