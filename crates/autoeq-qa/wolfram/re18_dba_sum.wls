(* autoeq-qa case autoeq-qa.re18-dba-sum.v1 (catalogue family RE18).
   Independent coherent front/rear array sum: the rear array carries the
   distance-derived delay (d/c) and the DBA polarity inversion baked into
   its measured phase; the oracle is the direct complex sum. *)

Off[General::munfl];
sr = 48000.;
c = 343.;
d = 3.4;
tau = N[d/c, 16];
freqs = {20., 40., 60., 100., 150., 200.};
frontSpl = 80.;
frontPhase = 0.;
rearSpl = 76.5;
rearPhase[f_] := N[(-360. f tau + 180.) Degree, 16];
sums = Table[
   N[10^(frontSpl/20), 16] * Exp[I N[frontPhase Degree, 16]] +
    N[10^(rearSpl/20), 16] * Exp[I rearPhase[f]],
   {f, freqs}];
splOf[z_] := N[20 Log10[Max[Abs[z], 10^-12]], 16];
argOf[z_] := N[Arg[z]/Degree, 16];
rearPhases = Table[N[(-360. f N[tau, 16] + 180.), 16], {f, freqs}];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re18-dba-sum.v1",
  "family" -> "RE18",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "grid" -> "shared identical array grid (no resampling)",
  "units" -> <|"spl" -> "dB", "phase" -> "degrees", "distance" -> "m"|>,
  "calibration" -> "mic-1",
  "sound_speed_m_s" -> c,
  "array_spacing_m" -> d,
  "rear_delay_s" -> N[tau, 16],
  "front_spl_db" -> frontSpl,
  "front_phase_deg" -> frontPhase,
  "rear_spl_db" -> rearSpl,
  "rear_phase_deg" -> rearPhases,
  "combined_re_im" -> (complexPair[#] & /@ sums),
  "combined_spl_db" -> (splOf[#] & /@ sums),
  "combined_phase_deg" -> (argOf[#] & /@ sums)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
