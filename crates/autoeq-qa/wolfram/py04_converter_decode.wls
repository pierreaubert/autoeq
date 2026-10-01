(* autoeq-qa case autoeq-qa.py04-converter-decode.v1 (catalogue family PY04).
   Independent evaluation of the converter axis math (utils/msop2csv.py
   decode_response): SPL = 20*log10(|H|), phase = Arg(H) in degrees, plus
   the declared 6-decimal CSV rounding of freq/spl/phase. *)

Off[General::munfl];
freq = {20., 1000., 20000.};
re = {1., 0.5, -0.25};
im = {0., 0.5, 0.4330127018922193};
mag[i_] := N[Sqrt[re[[i]]^2 + im[[i]]^2], 16];
spl[i_] := N[20*Log10[mag[i]], 16];
phase[i_] := N[ArcTan[re[[i]], im[[i]]]*180/Pi, 16];
spls = spl[#] & /@ Range[3];
phases = phase[#] & /@ Range[3];
rnd[x_] := ToString[NumberForm[N[x, 7], {Infinity, 6}, ExponentFunction -> (Null &)]];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.py04-converter-decode.v1",
  "family" -> "PY04",
  "sample_rate_hz" -> 48000.,
  "grid_hz" -> freq,
  "units" -> "SPL dB with phase degrees; CSV rounded to 6 decimals",
  "calibration" -> "SPL=20 log10 |H|, phase=Arg(H), half-even display rounding",
  "re" -> re,
  "im" -> im,
  "spl_db" -> spls,
  "phase_deg" -> phases,
  "spl_rounded" -> (rnd[#] & /@ spls),
  "phase_rounded" -> (rnd[#] & /@ phases)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
