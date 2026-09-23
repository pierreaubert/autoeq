(* autoeq-qa case autoeq-qa.re12-gd-coherence.v1 (catalogue family RE12).
   Independent evaluation of controlled-channel group delay, the
   coherence summary, and the phase-confidence gates from closed forms
   only. Two controlled channels share one grid: flat 80 dB (coherence
   0.95) and flat 76 dB inverted (coherence 0.92), both with zero
   intrinsic phase, each realizing a pure 2 ms delay. The coherent sum
   is Hsum = (A0 - A1) Exp[-j w tau], so by the published group-delay
   definition GD = -d phi/d omega the reference GD is exactly 2 ms at
   every in-band bin (the finite-difference replay agrees because the
   phase is exactly linear; no cancellation guard triggers since
   |A0 - A1|/(A0 + A1) >> 1e-3). The coherence summary is the plain
   arithmetic mean 0.935 across every in-band bin of every channel, and
   the Trustworthy verdict follows from the documented gate order
   (nonempty curves, valid band, phase present, coherence present,
   threshold 0.9, no noise floors to check). Degraded reasons
   (no_curves, no_phase_data, coherence_below_threshold), the crossover
   advisory split (unknown vs bad evidence), and derive_band
   (max(min, fc/4), 2 fc) are declared from their documented rules.
   Weighted-median targeting and bootstrap enumeration have no
   independently evaluable public contract at this fixture scale and
   are not claimed here. *)

Off[General::munfl];
freqs = N[{40., 50., 60., 80., 100., 120.}, 16];
band = {40., 120.};
delayMs = 2.;
spls = {80., 76.};
cohs = {0.95, 0.92};
sumGd = Table[N[delayMs, 16], {6}];
meanCoh = N[Mean[Flatten[Table[cohs, {6}]]], 16];
deriveIn = {20., 80.};
deriveOut = {N[Max[deriveIn[[1]], deriveIn[[2]]*0.25], 16],
  N[deriveIn[[2]]*2., 16]};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re12-gd-coherence.v1",
  "family" -> "RE12",
  "sample_rate_hz" -> 48000.,
  "grid" -> "explicit 6-point grid 40..120 Hz shared by both channels",
  "units" -> "Hz, dB SPL, ms, coherence gamma^2",
  "calibration" -> "absolute channel levels; flat channels with zero intrinsic phase",
  "formula" -> "Hsum = (A0 - A1) Exp[-j w tau]; GD = -dphi/dw = tau = 2 ms",
  "band_hz" -> band,
  "freqs_hz" -> freqs,
  "channel_spl_db" -> spls,
  "channel_coherence" -> cohs,
  "channel_delay_ms" -> {delayMs, delayMs},
  "channel_inverted" -> {False, True},
  "per_channel_delay_ms" -> {delayMs, delayMs},
  "sum_gd_reference_ms" -> sumGd,
  "mean_coherence" -> meanCoh,
  "coherence_threshold" -> 0.9,
  "verdict" -> "trustworthy",
  "degraded_empty" -> "no_curves",
  "degraded_no_phase" -> "no_phase_data",
  "degraded_low_coherence" -> "coherence_below_threshold",
  "advisories_unknown" -> {"crossover_phase_coherence_unverified",
    "crossover_phase_snr_unverified"},
  "derive_band_in" -> deriveIn,
  "derive_band_out" -> deriveOut
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
