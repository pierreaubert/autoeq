(* autoeq-qa case autoeq-qa.c07-quality-geometry.v1 (catalogue family C07).
   Independent measurement-confidence and direct-sound geometry summary.
   Medians/minima follow the published order-statistic definitions
   (sort, then middle element for odd n); seat variance is the
   per-bin population standard deviation (divide by seat count N, then
   square root) averaged/maximized across bins; the reflection-free
   interval is (rR - rD)/c; the gate bound is cycles/gate; the timing
   bound is the 9-degree phase condition f <= (9/360)/(residual_s),
   i.e. f <= 25000/residual_us. Compared against
   autoeq_core::measurement_quality, capture_provenance and
   direct_sound. Grid/units/calibration are recorded below. *)

Off[General::munfl];
freqs = {20., 100., 1000.};
spl = {80., 81., 79.};
noiseFloor = {40., 42., 41.};
coherence = {0.95, 0.98, 0.94};

snr = spl - noiseFloor;
sortedSnr = Sort[snr];
medianSnr = N[sortedSnr[[2]], 16];
sortedCoh = Sort[coherence];
minCoh = N[sortedCoh[[1]], 16];
medianCoh = N[sortedCoh[[2]], 16];
quality = If[medianCoh >= 0.9 && medianSnr >= 20., "good", "not-good"];
depthScale = If[quality === "good", 1., -1.];

seats = {{80., 81., 79.}, {82., 79., 81.}, {80., 83., 77.}};
binStd[vals_] := Module[{m = N[Total[vals]/Length[vals], 16]},
  N[Sqrt[Total[(# - m)^2 & /@ vals]/Length[vals]], 16]];
perBinStd = Table[binStd[seats[[All, b]]], {b, 1, 3}];
meanVar = N[Total[perBinStd]/Length[perBinStd], 16];
maxVar = N[Max[perBinStd], 16];

soundSpeed = 343.;
directM = 2.;
reflectionM = 3.43;
intervalS = N[(reflectionM - directM)/soundSpeed, 16];
gateS = 0.002;
cycles = 2.;
lowerHz = N[cycles/gateS, 16];
gateFits = gateS <= intervalS;

residualUs = 40.;
maxCoherentHz = N[25000./residualUs, 16];
phaseDegAtBound = N[360. maxCoherentHz*residualUs*1.*10^-6, 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c07-quality-geometry.v1",
  "family" -> "C07",
  "sample_rate_hz" -> Null,
  "freqs_hz" -> freqs,
  "units" -> <|"spl" -> "dB", "coherence" -> "gamma-squared-unitless",
    "snr" -> "dB", "seat_variance" -> "dB",
    "interval" -> "s", "lower_bound" -> "Hz",
    "residual" -> "us", "phase" -> "deg"|>,
  "calibration" -> <|"noise_floor_db" -> noiseFloor,
    "sound_speed_m_s" -> soundSpeed|>,
  "spl_db" -> spl,
  "coherence_vals" -> coherence,
  "snr_db" -> (N[#, 16] & /@ snr),
  "median_snr_db" -> medianSnr,
  "min_coherence" -> minCoh,
  "median_coherence" -> medianCoh,
  "quality" -> quality,
  "correction_depth_scale" -> depthScale,
  "seats_spl_db" -> seats,
  "per_bin_std_db" -> (N[#, 16] & /@ perBinStd),
  "mean_seat_variance_db" -> meanVar,
  "max_seat_variance_db" -> maxVar,
  "direct_path_m" -> directM,
  "first_reflection_path_m" -> reflectionM,
  "gate_s" -> gateS,
  "cycles_required" -> cycles,
  "reflection_free_interval_s" -> intervalS,
  "valid_lower_bound_hz" -> lowerHz,
  "gate_fits_interval" -> gateFits,
  "residual_uncertainty_us" -> residualUs,
  "max_coherent_hz" -> maxCoherentHz,
  "phase_uncertainty_deg_at_bound" -> phaseDegAtBound
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
