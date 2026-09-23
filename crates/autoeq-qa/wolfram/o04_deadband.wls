(* autoeq-qa case autoeq-qa.o04-deadband.v1 (catalogue family O04).
   Independent evaluation of the audibility deadband: piecewise threshold
   T(f) = bassDb below bassMidHz, trebleDb above midTrebleHz, log-frequency
   interpolation of midDb->trebleDb between the joins; no deadband below
   schroederHz when disableBelowSchroeder is set or outside [minF, maxF];
   softening out = 0 when |e| <= T else sign(e)*(|e|-T) (residual exactly at
   +-threshold maps to zero). Compared against apply_audibility_deadband.
   Units are dB on a declared grid. *)

Off[General::munfl];
freqs = {50., 250., 300., 1000., 2000., 8000.};
bassDb = 0.25; midDb = 0.75; trebleDb = 1.0;
bassMidHz = 250.; midTrebleHz = 2000.;
disableBelow = True; schroederHz = 300.;
minF = 20.; maxF = 20000.;

thr[f_] := N[Which[
  f <= bassMidHz, bassDb,
  f >= midTrebleHz, trebleDb,
  True, midDb + (Log[f] - Log[bassMidHz])/(Log[midTrebleHz] - Log[bassMidHz])*(trebleDb - midDb)
], 16];
active[f_] := minF <= f <= maxF && !(disableBelow && f < schroederHz);
thresholds = Table[N[If[active[freqs[[i]]], thr[freqs[[i]]], 0.], 16], {i, 1, Length[freqs]}];
t300 = thr[300.]; t1000 = thr[1000.];
err = {0.1, 0.25, t300, -t1000, -2.5, -0.5};
out = Table[
  Module[{f = freqs[[i]], e = err[[i]], t = thresholds[[i]]},
    N[If[!active[f], e, If[Abs[e] <= t, 0., Sign[e]*(Abs[e] - t)]], 16]],
  {i, 1, Length[freqs]}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o04-deadband.v1",
  "family" -> "O04",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "dB residual softening on a declared grid; no sampling involved",
  "grid_hz" -> freqs,
  "units" -> "dB",
  "calibration" -> "residual magnitudes; no SPL calibration",
  "min_freq_hz" -> minF,
  "max_freq_hz" -> maxF,
  "config" -> <|
    "enabled" -> True,
    "bass_db" -> bassDb, "mid_db" -> midDb, "treble_db" -> trebleDb,
    "bass_mid_hz" -> bassMidHz, "mid_treble_hz" -> midTrebleHz,
    "disable_below_schroeder" -> disableBelow,
    "schroeder_hz" -> schroederHz|>,
  "errors_db" -> err,
  "thresholds_db" -> thresholds,
  "outputs_db" -> out
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
