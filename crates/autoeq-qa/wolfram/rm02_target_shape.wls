(* autoeq-qa case autoeq-qa.rm02-target-shape.v1 (catalogue family RM02).
   Independent construction of the procedural dB-domain target:
   base tilt slope*Log2[f/ref] plus a 2nd-order bass preference
   bassDb/(1+(f/fb)^2) and a 4th-order treble preference
   trebleDb/(1+(ft/f)^4) gated to f > ft/4; plus the logistic
   evidence-regime handover w = 1/(1+Exp[-Log2[f/c]/width]) and the
   effective anechoic confidence w*Clip[dc,{0,1}]. The damage-guard
   fire rule (fires iff w(band-top) >= 1/2 without validated direct
   evidence) is an exact branch check. Compared against
   build_complete_target_curve, TransitionConfig weights, and
   evaluate_damage_guard. dB-domain shapes only, no biquad realization. *)

Off[General::munfl];
freqs = {20., 50., 100., 200., 500., 1000., 2000., 5000., 8000., 10000., 20000.};
slope = -0.8; refF = 1000.;
bassDb = 3.; bassF = 200.;
trebleDb = -2.; trebleF = 8000.;
tilt[f_] := N[slope*Log[2, Max[f, 1.] / Max[refF, 1.]], 16];
bass[f_] := N[If[Abs[bassDb] > 0.001 && bassF > 0., bassDb/(1. + (f/bassF)^2), 0.], 16];
treble[f_] := N[If[Abs[trebleDb] > 0.001 && trebleF > 0. && f > trebleF*0.25,
  trebleDb/(1. + (trebleF/f)^4), 0.], 16];
tilts = tilt /@ freqs;
basses = bass /@ freqs;
trebles = treble /@ freqs;
totals = MapThread[N[#1 + #2 + #3, 16] &, {tilts, basses, trebles}];

center = 300.; width = 1.;
weight[f_] := N[1./(1. + Exp[-(Log[2, f/center]/width)]), 16];
wFreqs = {75., 150., 300., 600., 1200.};
weights = weight /@ wFreqs;
dc = 0.8;
effective = Map[N[#*Min[Max[dc, 0.], 1.], 16] &, weights];
(* Guard bands: top edge decides; RoomCurveOnly evidence fires iff w >= 1/2. *)
bandTops = {200., 300., 4000.};
bandWeights = weight /@ bandTops;
firesRoomOnly = Map[# >= 0.5 &, bandWeights];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rm02-target-shape.v1",
  "family" -> "RM02",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "procedural dB-domain target on a declared grid; no sampling involved",
  "grid_hz" -> freqs,
  "units" -> "dB SPL offset relative to the 1000 Hz reference",
  "calibration" -> "relative target shape; no SPL calibration",
  "config" -> <|
    "shape" -> "Harman", "slope_db_per_octave" -> slope,
    "reference_freq_hz" -> refF,
    "bass_shelf_db" -> bassDb, "bass_shelf_freq_hz" -> bassF,
    "treble_shelf_db" -> trebleDb, "treble_shelf_freq_hz" -> trebleF,
    "transition_center_hz" -> center, "transition_width_oct" -> width,
    "direct_confidence" -> dc|>,
  "tilt_db" -> tilts,
  "bass_db" -> basses,
  "treble_db" -> trebles,
  "total_db" -> totals,
  "weight_freqs_hz" -> wFreqs,
  "anechoic_weights" -> weights,
  "effective_confidences" -> effective,
  "guard_band_tops_hz" -> bandTops,
  "guard_band_weights" -> bandWeights,
  "guard_fires_room_curve_only" -> firesRoomOnly
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
