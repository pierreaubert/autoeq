(* autoeq-qa case autoeq-qa.ra10-evidence-stats.v1 (catalogue family RA10).
   Independent evidence statistics from the published definitions:
   - band repeatability / seat spread: worst-bin max-min range over the
     bins of each band (repeats and seats use the same mechanics but are
     deliberately distinct types; spatial difference is not noise);
   - timing-to-phase: 360 * f * dt degrees; non-physical inputs give
     no value (recorded as Null);
   - band support: per-band minima of coherence and of SPL-minus-noise
     SNR against the v1 policy (min_coherence 0.7, min_snr 10 dB);
     a band with no covered bins, or a curve with no coherence/noise
     evidence, is Unknown (never good quality); support fractions are
     exact counts over the three covered bands;
   - direct/reflected geometry: reflection-free interval
     (rR - rD)/c, valid lower bound cycles/gate, per-frequency gate
     labels (ReflectionFree at/above the bound, else
     ReflectionContaminated), and the quasi-anechoic verdict for a
     gated direct-sound capture with two off-axis directions and a
     requested band inside the validated window (DetailEligible,
     Supported phase source, no reason codes).
   Compared against compute_band_repeatability / compute_band_seat_spread /
   phase_uncertainty_deg / assess_band_support / reflection_free_interval_s /
   valid_lower_bound_hz / gate_label_for / validate_quasi_anechoic.
   Decision labels and reason codes are compared exactly. *)

Off[General::munfl];
grid = {20., 40., 80., 160., 320., 640., 1280., 2560., 5120., 10240.};
r0 = Table[80., {10}];
r1 = 80 + {0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5};
r2 = 80 + {-0.2, 0.3, -0.4, 0.2, -0.1, 0.4, -0.3, 0.1, -0.2, 0.3};
reps = {r0, r1, r2};
bands = {{20., 200.}, {200., 2000.}, {2000., 12000.}};
bandBins[lo_, hi_] := Flatten[Position[grid, f_ /; lo <= f <= hi]];
repSpread = Table[
   Module[{bins = bandBins[bands[[b, 1]], bands[[b, 2]]]},
    N[Max[Table[
       Max[reps[[1, i]], reps[[2, i]], reps[[3, i]]] -
        Min[reps[[1, i]], reps[[2, i]], reps[[3, i]]], {i, bins}]], 16]],
   {b, 1, 3}];
(* Seats are exactly 80/82/79 dB: spread 3 dB in every band. *)
seatSpread = {3., 3., 3.};

dt = 0.0005;
phAt[f_] := N[360*f*dt, 16];
ph100 = phAt[100.]; ph1000 = phAt[1000.];

coh = {0.95, 0.95, 0.95, 0.4, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95};
noise = {55., 55., 55., 55., 55., 55., 55., 55., 75., 55.};
splSup = Table[80., {10}];
snr = Table[N[splSup[[i]] - noise[[i]], 16], {i, 1, 10}];
bandMin[b_] := Module[{bins = bandBins[bands[[b, 1]], bands[[b, 2]]]},
   {N[Min[coh[[bins]]], 16], N[Min[snr[[bins]]], 16]}];
minCoh = Table[bandMin[b][[1]], {b, 1, 3}];
minSnr = Table[bandMin[b][[2]], {b, 1, 3}];
support = {"restricted", "supported", "restricted"};
fracSupported = N[1/3, 16]; fracRestricted = N[2/3, 16];

directM = 1.0; reflM = 2.0; csnd = 343.0;
interval = N[(reflM - directM)/csnd, 16];
gate = 0.002; cycles = 2.0;
lower = N[cycles/gate, 16];
upper = 8000.;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra10-evidence-stats.v1",
  "family" -> "RA10",
  "sample_rate_hz" -> Null,
  "grid_hz" -> grid,
  "grid_units" -> "Hz",
  "sample_units" -> "dB SPL; seconds; degrees",
  "calibration" -> "analytic plant; synthetic repeats/seats/capture facts, no measurement",
  "repeat_curves_db" -> reps,
  "seat_levels_db" -> {80., 82., 79.},
  "bands_hz" -> bands,
  "repeat_spread_db" -> repSpread,
  "seat_spread_db" -> seatSpread,
  "timing_s" -> dt,
  "phase_at_100hz_deg" -> ph100,
  "phase_at_1000hz_deg" -> ph1000,
  "phase_invalid_freq" -> Null,
  "phase_invalid_timing" -> Null,
  "support_coherence" -> coh,
  "support_noise_db" -> noise,
  "support_spl_db" -> splSup,
  "policy_min_coherence" -> 0.7,
  "policy_min_snr_db" -> 10.0,
  "band_min_coherence" -> minCoh,
  "band_min_snr_db" -> minSnr,
  "band_support" -> support,
  "fraction_supported" -> fracSupported,
  "fraction_restricted" -> fracRestricted,
  "gap_band_hz" -> {30000., 40000.},
  "gap_support" -> "unknown",
  "direct_path_m" -> directM,
  "reflection_path_m" -> reflM,
  "sound_speed_m_s" -> csnd,
  "reflection_free_interval_s" -> interval,
  "gate_s" -> gate,
  "cycles_for_valid_band" -> cycles,
  "valid_lower_hz" -> lower,
  "valid_upper_hz" -> upper,
  "gate_label_above" -> "reflection_free",
  "gate_label_below" -> "reflection_contaminated",
  "gate_label_nobound" -> "reflection_contaminated",
  "quasi_record_id" -> "rec-1",
  "quasi_angles_deg" -> {0., 20., -30.},
  "quasi_requested_band_hz" -> {1200., 8000.},
  "quasi_detail" -> "detail_eligible",
  "quasi_phase_source" -> "supported",
  "quasi_tonal_permitted" -> True,
  "quasi_reasons" -> {}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
