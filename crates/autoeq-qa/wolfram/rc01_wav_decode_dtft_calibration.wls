(* autoeq-qa case autoeq-qa.rc01-wav-decode-dtft-calibration.v1 (catalogue family RC01).
   Independent WAV-decode plus full-record DTFT plus declared
   calibration: integer PCM16 samples decode as int/2^15, IEEE
   float32 samples decode identically, the full record is evaluated
   with the direct DTFT H(f) = Sum_n x[n] Exp[-I 2 Pi f n/sr] (no
   window, recentering, normalization, or resampling), and the
   reported level is 20 Log10[Abs[H]] + offsetDb with the declared
   offset only. The Rust test parses minimal mono WAV bytes with the
   same documented rules and DTFTs through the shared FIR kernel. *)

Off[General::munfl];
sr = 8000.;
offsetDb = 20.;
pcm16 = {0, 8192, 16384, 24576, 32767, -32768, -16384, -8192};
norm16 = (N[#, 16] & /@ (pcm16/32768.));
floats32 = {0., 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.125};
spots = {500., 1500., 2500., 3500.};
dtft[seq_, f_] := Sum[seq[[n + 1]]*Exp[-I N[2 Pi f n/sr, 16]], {n, 0, Length[seq] - 1}];
pair[v_] := {N[Re[v], 16], N[Im[v], 16]};
cal[v_] := N[20*Log10[Abs[v]] + offsetDb, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rc01-wav-decode-dtft-calibration.v1",
  "family" -> "RC01",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit DTFT spots in Hz; full 8-sample records, no truncation",
  "units" -> <|"samples" -> "linear amplitude, full scale = 1", "level" -> "dB"|>,
  "calibration" -> "declared magnitude offset added after 20log10 of DTFT magnitude",
  "wav_bits_int" -> 16,
  "wav_channels" -> 1,
  "int_samples" -> pcm16,
  "expected_float16" -> norm16,
  "float32_samples" -> floats32,
  "magnitude_offset_db" -> offsetDb,
  "spots_hz" -> spots,
  "dtft16_re_im" -> (pair[dtft[norm16, #]] & /@ spots),
  "dtft32_re_im" -> (pair[dtft[floats32, #]] & /@ spots),
  "cal16_db" -> (cal[dtft[norm16, #]] & /@ spots),
  "cal32_db" -> (cal[dtft[floats32, #]] & /@ spots)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
