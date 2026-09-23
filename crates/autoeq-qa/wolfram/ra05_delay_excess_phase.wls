(* autoeq-qa case autoeq-qa.ra05-delay-excess-phase.v1 (catalogue family RA05).
   Independent delay-removal and excess-phase arithmetic on two known
   systems over a 256-point log grid 40-18000 Hz at 48 kHz: (a) a pure
   delay of 2.3 ms on flat 0 dB magnitude, where the bulk delay is the
   closed-form least-squares slope tau = -slope/360 of unwrapped phase
   (degrees) versus frequency and every residual/smoothed excess quantity
   is identically zero; (b) a first-order all-pass with corner 200 Hz,
   arg = -2 atan(f/f0), whose analytic excess group delay is
   1000/(pi f0 (1 + (f/f0)^2)) ms. Compared against
   roomeq_analysis::excess_phase::{assess_excess_phase, propose_correction}.
   Magnitude/min-phase agreement (flat/zero here) is separated from the
   excess-phase claims. Phases in degrees, delays in seconds, GD in ms. *)

Off[General::munfl];
sr = 48000.;
n = 256; fLo = 40.; fHi = 18000.;
freqs = N[Table[fLo*(fHi/fLo)^(i/(n - 1)), {i, 0, n - 1}], 16];
tau = 0.0023;
phaseDelay = N[-360.*freqs*tau, 16];
magDb = Table[0., {n}];
snrDb = Table[40., {n}];

mx = N[Mean[freqs], 16]; my = N[Mean[phaseDelay], 16];
slope = N[Total[(freqs - mx)*(phaseDelay - my)]/Total[(freqs - mx)^2], 16];
tauHat = N[-slope/360., 16];

f0 = 200.;
phaseAP = N[-2.*ArcTan[#/f0]*180./Pi & /@ freqs, 16];
gdAt[f_] := N[1000./(Pi*f0*(1 + (f/f0)^2)), 16];
near40 = First[Ordering[Abs[freqs - 40.], 1]];
gd40 = gdAt[freqs[[near40]]];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ra05-delay-excess-phase.v1",
  "family" -> "RA05",
  "sample_rate_hz" -> sr,
  "grid_hz" -> freqs,
  "grid_units" -> "Hz",
  "magnitude_db" -> magDb,
  "magnitude_units" -> "dB",
  "snr_db" -> snrDb,
  "snr_units" -> "dB",
  "calibration" -> "synthetic flat 0 dB, exact linear/all-pass phase, 40 dB uniform SNR",
  "phase_units" -> "degrees",
  "delay_tau_s" -> tau,
  "phase_delay_deg" -> phaseDelay,
  "expected_bulk_delay_s" -> tauHat,
  "expected_residual_deg" -> 0.,
  "expected_excess_gd_ms" -> 0.,
  "expected_correction_phase_rad" -> 0.,
  "allpass_corner_hz" -> f0,
  "phase_allpass_deg" -> phaseAP,
  "allpass_probe_hz" -> freqs[[near40]],
  "expected_allpass_gd_ms" -> gd40,
  "analysis_band_hz" -> {fLo, fHi},
  "snr_floor_db" -> 10.,
  "min_valid_fraction" -> 0.8,
  "smooth_narrow_oct" -> N[1/6, 16],
  "smooth_wide_oct" -> 0.5,
  "consistency_tol_ms" -> 0.5,
  "taper_oct" -> 0.5,
  "dip_depth_db" -> 6.,
  "correction_max_taps" -> 512.,
  "correction_max_latency_ms" -> 4.
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
