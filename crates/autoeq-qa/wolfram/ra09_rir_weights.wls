(* autoeq-qa case autoeq-qa.ra09-rir-weights.v1 (catalogue family RA09).
   Independent RIR-prototype weighting from the published definitions:
   - Euclidean distances from the reference position;
   - source-axis angles via normalized dot products (mic coincident with
     the source falls back to 0.0);
   - distance weights: Uniform 1.0, InverseSquare 1/max(d,1e-6)^2,
     Gaussian Exp[-d^2/(2 sigma^2)];
   - rigid-sphere directivity: ka = 2 pi f r/c, c = 343 m/s,
     directionality = ka^2/(1+ka^2), off-axis = (1+cos th)/2,
     w = (1-directionality) + off-axis*directionality;
   - per-frequency column normalization of the [mic x freq] matrix;
   - magnitude-only power prototype: 10 log10(sum_i w P_i),
     P_i = 10^(L_i/10);
   - arrival-energy mixture for the in-band bin: per-event energies
     10^((E-max)/10), directional sums with measured arrival angles,
     raw = dw*directional/energy, renormalized; out-of-band bins keep
     geometric weights; measured_direction_bins counts mixture bins.
   Fixtures: (A) uniform/omnidirectional 2-mic power mean from 80/86 dB;
   (B) Gaussian + frequency-dependent spherical-head 2-mic geometry;
   (C) 4-mic compact capture with one direct + one reflection arrival
   (1000 Hz inside the [300, 1200] Hz band, 100/10000 Hz outside).
   Compared against build_weighted_prototype / with_capture, including
   the magnitude-only contract (phase None). Prototype magnitude must
   never be read as measured coherent phase. No time domain: sample
   rate recorded as Null. *)

Off[General::munfl];
c = 343.0;

(* ---------- Part A: uniform / omni power mean ---------- *)
protoA = N[10*Log10[(10^(80/10) + 10^(86/10))/2], 16];

(* ---------- Part B: Gaussian + spherical head ---------- *)
refB = {0., 0., 0.}; srcB = {0., 2., 0.};
micsB = {{0.1, 0., 0.}, {1., 0., 0.}};
freqsB = {100., 1000., 10000.};
distB = Table[N[Sqrt[Total[(micsB[[i]] - refB)^2]], 16], {i, 1, 2}];
axisB = N[(refB - srcB)/Sqrt[Total[(refB - srcB)^2]], 16];
angB = Table[
   Module[{v = N[micsB[[i]] - srcB, 16], n},
    n = Sqrt[Total[v^2]];
    If[n < 1*^-9, 0., N[ArcCos[Clip[Total[axisB*(v/n)], {-1., 1.}]], 16]]],
   {i, 1, 2}];
sigma = 0.5; radius = 0.0875;
dwB = Table[N[Exp[-distB[[i]]^2/(2*sigma^2)], 16], {i, 1, 2}];
dirW[f_, th_] := Module[{ka, dd, off},
   ka = N[2*Pi*f*radius/c, 16]; dd = ka^2/(1 + ka^2);
   off = 0.5*(1 + Cos[th]); N[(1 - dd) + off*dd, 16]];
rawB = Table[dwB[[i]]*dirW[freqsB[[j]], angB[[i]]], {i, 1, 2}, {j, 1, 3}];
wB = Table[
   Module[{col = rawB[[All, j]], s = Total[rawB[[All, j]]]}, N[col/s, 16]],
   {j, 1, 3}];
splB = {{80., 80., 80.}, {86., 86., 86.}};
protoB = Table[
   N[10*Log10[Sum[wB[[j, i]]*10^(splB[[i, j]]/10), {i, 1, 2}]], 16],
   {j, 1, 3}];

(* ---------- Part C: arrival-energy mixture ---------- *)
refC = {0., 0., 0.}; srcC = {1., 0., 0.};
micsC = {{0., 0., 0.}, {0.06, 0., 0.}, {0., 0.06, 0.}, {0., 0., 0.06}};
freqsC = {100., 1000., 10000.};
axisC = N[(refC - srcC)/Sqrt[Total[(refC - srcC)^2]], 16];
angC = Table[
   Module[{v = N[micsC[[i]] - srcC, 16], n},
    n = Sqrt[Total[v^2]];
    If[n < 1*^-9, 0., N[ArcCos[Clip[Total[axisC*(v/n)], {-1., 1.}]], 16]]],
   {i, 1, 4}];
(* Geometric baseline: uniform distance weights times the
   frequency-dependent spherical-head directivity at the geometric
   angles, renormalized per bin. Mics 2/3 sit slightly off-axis, so
   this is only near-uniform; compute it, never assume it. *)
geoRaw = Table[dirW[freqsC[[j]], angC[[i]]], {i, 1, 4}, {j, 1, 3}];
geoW = Table[
   Module[{col = geoRaw[[All, j]], s = Total[geoRaw[[All, j]]]}, N[col/s, 16]],
   {j, 1, 3}];
(* In-band mixture at 1000 Hz: direct arrival angle 0, reflection Pi. *)
dirMeas = {0., N[Pi, 16]};
eDirect = {0., 0., 0., 0.}; eRefl = {-20., 0., -20., 0.};
mixW = Table[
   Module[{mx = Max[eDirect[[i]], eRefl[[i]]], es, ds},
    es = 10^((eDirect[[i]] - mx)/10) + 10^((eRefl[[i]] - mx)/10);
    ds = 10^((eDirect[[i]] - mx)/10)*dirW[1000., dirMeas[[1]]] +
      10^((eRefl[[i]] - mx)/10)*dirW[1000., dirMeas[[2]]];
    N[ds/es, 16]], {i, 1, 4}];
mixTot = Total[mixW];
mixWn = N[mixW/mixTot, 16];
wC = {geoW[[1]], mixWn, geoW[[3]]};
levelsC = {0., 10., 0., 10.};
protoC = Table[
   N[10*Log10[Sum[wC[[j, i]]*10^(levelsC[[i]]/10), {i, 1, 4}]], 16],
   {j, 1, 3}];
measBins = {1, 1, 1, 1};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra09-rir-weights.v1",
  "family" -> "RA09",
  "sample_rate_hz" -> Null,
  "grid_hz" -> freqsC,
  "grid_units" -> "Hz",
  "sample_units" -> "dB SPL; weights dimensionless",
  "calibration" -> "analytic plant; synthetic geometry and levels, no measurement",
  "sound_speed_m_s" -> c,
  "proto_uniform_omni_db" -> protoA,
  "geomB_reference_m" -> refB,
  "geomB_source_m" -> srcB,
  "geomB_mics_m" -> micsB,
  "geomB_freqs_hz" -> freqsB,
  "geomB_distances_m" -> distB,
  "geomB_angles_rad" -> angB,
  "geomB_gauss_sigma_m" -> sigma,
  "geomB_head_radius_m" -> radius,
  "geomB_distance_weights" -> dwB,
  "geomB_weights" -> Transpose[wB],
  "geomB_spl_db" -> splB,
  "geomB_prototype_db" -> protoB,
  "mix_reference_m" -> refC,
  "mix_source_m" -> srcC,
  "mix_mics_m" -> micsC,
  "mix_freqs_hz" -> freqsC,
  "mix_geometric_angles_rad" -> angC,
  "mix_geometric_weights" -> geoW,
  "mix_arrival_angles_rad" -> dirMeas,
  "mix_direct_energy_db" -> eDirect,
  "mix_reflection_energy_db" -> eRefl,
  "mix_band_hz" -> {300., 1200.},
  "mix_weights_raw" -> mixW,
  "mix_weights" -> mixWn,
  "mix_weights_by_bin" -> wC,
  "mix_levels_db" -> levelsC,
  "mix_prototype_db" -> protoC,
  "mix_measured_direction_bins" -> measBins
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
