(* autoeq-qa case autoeq-qa.ra08-area-interp.v1 (catalogue family RA08).
   Independent continuous-area interpolation from the published IDW and
   circular-mean definitions:
   - raw IDW weight w_k = 1/(d_k + eps)^p with p = 2, eps = 1e-9,
     normalized to sum 1; a query within eps of a calibration point
     collapses to that point (one-hot);
   - SPL interpolates as the weighted dB mean;
   - phase interpolates as the weighted circular (complex) mean:
     re = sum w cos(theta), im = sum w sin(theta) (theta in radians),
     angle = Arg in degrees wrapped to [-180, 180], confidence
     R = sqrt(re^2 + im^2) in [0, 1].
   Fixtures: a 1D two-point couch line (midpoint = arithmetic mean under
   equal distance; opposed 10/100 deg phases give a 55 deg mean with
   R = sqrt(2)/4-ish partial cancellation), a 2D four-corner rectangle
   with the affine field 80 + x + 2y (center = 81.5 dB), an exact-sensor
   query (one-hot recovery), and an out-of-support query (rejected).
   Compared against ListeningArea::interpolate_with_evidence (IDW weights,
   curves, confidence) plus try_interpolate_at support rejection.
   Magnitude interpolation is checked here; no coherent-pressure claim
   is made from it. No time domain: sample rate recorded as Null. *)

Off[General::munfl];
p = 2.0; eps = 1*^-9;
deg2rad = N[Pi/180, 16];

(* --- 1D couch line --- *)
pos1 = {{0.}, {1.}};
spl1 = {{70., 71.}, {90., 91.}};
ph1 = {{10., 20.}, {100., 20.}};
qmid = {0.5};
d1 = Table[N[Sqrt[(pos1[[k, 1]] - qmid[[1]])^2], 16], {k, 1, 2}];
raw1 = Table[N[1/(d1[[k]] + eps)^p, 16], {k, 1, 2}];
w1 = N[raw1/Total[raw1], 16];
splMid = Table[N[Sum[w1[[k]]*spl1[[k, b]], {k, 1, 2}], 16], {b, 1, 2}];
circ[b_] := Module[{re, im},
   re = N[Sum[w1[[k]]*Cos[ph1[[k, b]]*deg2rad], {k, 1, 2}], 16];
   im = N[Sum[w1[[k]]*Sin[ph1[[k, b]]*deg2rad], {k, 1, 2}], 16];
   {N[Mod[N[ArcTan[re, im]/deg2rad, 16] + 180, 360] - 180, 16],
    N[Sqrt[re^2 + im^2], 16]}];
phMid = {circ[1][[1]], circ[2][[1]]};
confMid = {circ[1][[2]], circ[2][[2]]};

(* --- 2D rectangle, affine field 80 + x + 2y, zero phase --- *)
pos2 = {{0., 0.}, {1., 0.}, {0., 1.}, {1., 1.}};
spl2 = {{80.}, {81.}, {82.}, {83.}};
qc = {0.5, 0.5};
d2 = Table[N[Sqrt[(pos2[[k, 1]] - qc[[1]])^2 + (pos2[[k, 2]] - qc[[2]])^2], 16], {k, 1, 4}];
raw2 = Table[N[1/(d2[[k]] + eps)^p, 16], {k, 1, 4}];
w2 = N[raw2/Total[raw2], 16];
splCenter = N[Sum[w2[[k]]*spl2[[k, 1]], {k, 1, 4}], 16];

(* --- exact-sensor query recovers the calibration point --- *)
qexact = {1., 0.};
dex = Table[N[Sqrt[(pos2[[k, 1]] - qexact[[1]])^2 + (pos2[[k, 2]] - qexact[[2]])^2], 16], {k, 1, 4}];
wExact = Table[If[dex[[k]] <= eps, 1., 0.], {k, 1, 4}];
splExact = N[Sum[wExact[[k]]*spl2[[k, 1]], {k, 1, 4}], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra08-area-interp.v1",
  "family" -> "RA08",
  "sample_rate_hz" -> Null,
  "grid_hz" -> {100., 200.},
  "grid_units" -> "Hz",
  "sample_units" -> "dB SPL; phase degrees",
  "calibration" -> "analytic plant; synthetic 1D/2D fields, no measurement",
  "idw_power" -> p,
  "idw_epsilon" -> eps,
  "positions_1d" -> pos1,
  "spl_1d_db" -> spl1,
  "phase_1d_deg" -> ph1,
  "query_mid_1d" -> qmid,
  "weights_mid_1d" -> w1,
  "spl_mid_1d_db" -> splMid,
  "phase_mid_1d_deg" -> phMid,
  "confidence_mid_1d" -> confMid,
  "positions_2d" -> pos2,
  "spl_2d_db" -> spl2,
  "query_center_2d" -> qc,
  "weights_center_2d" -> w2,
  "spl_center_2d_db" -> splCenter,
  "query_exact_2d" -> qexact,
  "weights_exact_2d" -> wExact,
  "spl_exact_2d_db" -> splExact,
  "query_outside_1d" -> {2.0},
  "bbox_1d" -> {{0., 1.}}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
