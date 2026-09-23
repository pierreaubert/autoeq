(* autoeq-qa case autoeq-qa.py05-heldout-contract.v1 (catalogue family PY05).
   Independent evaluation of the held-out perturbation contract
   (scripts/generate_roomeq_held_out.py generate): octave = log2(max(f,20)/20),
   magnitudeDelta = 0.32*sin(oct*(1.1+0.13*pos)+cp) + 0.11*cos(oct*2.3+pos),
   phaseDelta = 2.5*sin(oct*0.9+pos+cp), with cp = 0.7 for channel R else 0. *)

Off[General::munfl];
freqs = {20., 100., 1000., 10000.};
spl0 = {80., 81., 82., 83.};
phase0 = {0., 10., -20., 30.};
pos = 1; ch = "L";
cp = 0.;
oct[f_] := N[Log[Max[f, 20.]/20.]/Log[2.], 16];
magDelta[f_] := N[0.32*Sin[oct[f]*(1.1 + 0.13*pos) + cp] + 0.11*Cos[oct[f]*2.3 + pos], 16];
phaseDelta[f_] := N[2.5*Sin[oct[f]*0.9 + pos + cp], 16];
splOut = N[spl0[[#]] + magDelta[freqs[[#]]], 16] & /@ Range[4];
phaseOut = N[phase0[[#]] + phaseDelta[freqs[[#]]], 16] & /@ Range[4];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.py05-heldout-contract.v1",
  "family" -> "PY05",
  "sample_rate_hz" -> 48000.,
  "grid_hz" -> freqs,
  "units" -> "SPL dB and phase degrees, 4-decimal CSV rounding",
  "calibration" -> "synthetic perturbation labels; position=1 channel=L",
  "position" -> pos,
  "channel" -> ch,
  "base_spl_db" -> spl0,
  "base_phase_deg" -> phase0,
  "heldout_spl_db" -> splOut,
  "heldout_phase_deg" -> phaseOut
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
