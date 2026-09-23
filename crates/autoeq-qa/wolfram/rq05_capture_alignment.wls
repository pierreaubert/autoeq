(* autoeq-qa case autoeq-qa.rq05-capture-alignment.v1 (catalogue family RQ05).
   Independent prediction-vs-capture residual arithmetic: only the
   predeclared broadband gain is applied (capture - (prediction + gain)),
   peak absolute magnitude residual, log-frequency-weighted mean change,
   and unexplained output loss Max[declaredGain - meanChange, 0].
   Weights replicate the declared log-frequency cell measure
   w[i] = Log[right/left] normalized (edges reuse the endpoint), never a
   uniform mean. Timing is unassessed here (no phase evidence); the
   comparison must report it as unassessed, never zero. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 200., 400., 800., 1600., 3200., 6400.};
pred = {70., 70., 70., 70., 70., 70., 70.};
gainDb = 2.;
resid = {0.10, -0.20, 0.15, -0.05, 0.10, -0.10, 0.05};
capt = pred + gainDb + resid;

n = Length[grid];
rawW = Table[
  Module[{left = If[i == 1, grid[[1]], grid[[i - 1]]],
    right = If[i == n, grid[[n]], grid[[i + 1]]]},
   N[Log[right/left], 16]],
  {i, 1, n}];
weights = rawW/Total[rawW];
meanPred = weights . pred;
meanCapt = weights . capt;
meanChange = N[meanCapt - meanPred, 16];
worstMag = N[Max[Abs[resid]], 16];
loss = N[Max[gainDb - meanChange, 0.], 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rq05-capture-alignment.v1",
  "family" -> "RQ05",
  "wolfram_engine" -> "15.0.0",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> grid,
  "units" -> "dB SPL, Hz, ms",
  "calibration" -> "common capture calibration cal-7 on both sides",
  "prediction_db" -> pred,
  "capture_db" -> capt,
  "declared_gain_db" -> gainDb,
  "declared_delay_ms" -> 1.5,
  "residuals_db" -> N[resid, 16],
  "log_weights" -> weights,
  "worst_magnitude_db" -> worstMag,
  "mean_change_db" -> meanChange,
  "unexplained_loss_db" -> loss,
  "overlap_hz" -> {100., 6400.},
  "requested_band_hz" -> {50., 12000.},
  "expected_excluded_bands" -> 2,
  "timing_assessed" -> False,
  "bass_cuts_only_deny_boost" -> True,
  "bass_cuts_only_allow_cut" -> True,
  "bass_full_allows_boost" -> True
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
