(* autoeq-qa case autoeq-qa.re24-cardioid.v1 (catalogue family RE24).
   Independent cardioid front/rear synthesis: the rear branch takes the
   propagation delay from the driver separation (tau = sep/c) plus the
   cardioid polarity inversion, then the branches sum coherently.
   Direct sums only. *)

Off[General::munfl];
sr = 48000.;
c = 343.;
sep = 0.34;
tauMs = N[sep/c 1000., 16];
freqs = {50., 80., 120., 180., 240., 300.};
frontSpl = 80.;
frontPhase = 0.;
rearSpl = 78.;
rearPhase = -10.;
procPhase[f_] := N[rearPhase - 360. f (tauMs/1000.) + 180., 16];
sums = Table[
   N[10^(frontSpl/20), 16] * Exp[I N[frontPhase Degree, 16]] +
    N[10^(rearSpl/20), 16] * Exp[I N[procPhase[f] Degree, 16]],
   {f, freqs}];
splOf[z_] := N[20 Log10[Max[Abs[z], 10^-12]], 16];
argOf[z_] := N[Arg[z]/Degree, 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re24-cardioid.v1",
  "family" -> "RE24",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "grid" -> "shared identical front/rear grid (no resampling)",
  "units" -> <|"spl" -> "dB", "phase" -> "degrees", "delay" -> "ms", "distance" -> "m"|>,
  "calibration" -> "mic-1",
  "sound_speed_m_s" -> c,
  "separation_m" -> sep,
  "rear_delay_ms" -> tauMs,
  "front_spl_db" -> frontSpl,
  "front_phase_deg" -> frontPhase,
  "rear_spl_db" -> rearSpl,
  "rear_phase_deg" -> rearPhase,
  "processed_rear_phase_deg" -> (procPhase[#] & /@ freqs),
  "combined_re_im" -> (complexPair[#] & /@ sums),
  "combined_spl_db" -> (splOf[#] & /@ sums),
  "combined_phase_deg" -> (argOf[#] & /@ sums)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
