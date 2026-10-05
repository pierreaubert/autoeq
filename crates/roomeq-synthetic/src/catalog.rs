//! G3 fixture catalogue for QA: constructor × fixture-ID × behavior × rates.
//!
//! | Constructor | Fixture ID | Planted behavior | Sample rate |
//! | --- | --- | --- | --- |
//! | `timing::clock_drift_fixture` | F01 | 50 ppm affine drift over 20 s = 1 ms; noisy markers; raw offsets preserved | 48 000 Hz (IRs) |
//! | `timing::uncertainty_band_fixture` | F06 input | Band-local noise + coherence drop, good rest of spectrum | rate-independent curves |
//! | `timing::coverage_gap_fixture` | F05 input | Disjoint gap stays NaN/unsupported, never interpolated | rate-independent curves |
//! | `timing::spatial_magnitude_only_sample` | F07 input | Magnitude only: no phase, coherence or sensitivity | rate-independent curves |
//! | `timing::calibration_unknown_sample` | F15 input | Relative spectrum, calibration stays unknown | rate-independent curves |
//! | `spatial::shared_bass_fixture` | F03 | Equal coherent sum +6.020599913 dB; opposite polarity cancels (no finite score) | analytic (no rate) |
//! | `spatial::common_eq_seat_pair` | F04 | Common EQ preserves the seat-to-seat ratio | 48 000 Hz biquad rendering |
//! | `spatial::worse_seat_counterexample` | F08 | Better seat mean with a regressed held-out seat | analytic (no rate) |
//! | `spatial::overlapping_removals_fixture` | F09 | Overlapping removals combine super-additively vs frozen full chain | rate-independent grid |
//! | `spatial::bass_only_candidate_with_upper_fault` | F14 | Bass fixed, +5 dB upper-band fault retained in full-band error | 48 000 Hz biquad rendering |
//! | `spatial::modal_cut_reference` | S2 modal | −9 dB Q=4 minimum-phase cut at 60 Hz with transfer + IR reference | 48 000 Hz |
//! | `spatial::moving_dip_fixture` | S2 dip | Deep null whose center moves across seats | 48 000 Hz biquad rendering |
//! | `spatial::narrow_peak_fixture` | S2 peak | Repeatable narrow +9 dB resonance at a fixed center | 48 000 Hz biquad rendering |
//! | `stimulus::equal_energy_signal_a/b` | F12 control | Equal-energy, disjoint spectra; no equivalence claim | 48 000 Hz |
//! | `stimulus::tilt_signal` | S3 tilt | 1/n harmonic-complex tilt | 48 000 Hz |
//! | `stimulus::resonance_signal` | S3 resonance | Sustained 75 Hz excitation with raised-cosine edges | 48 000 Hz |
//! | `stimulus::transient_signal` | S3 transient | Impulse + decaying 2 kHz burst | 48 000 Hz |
//! | `stimulus::am_sweep_signal` | S3 AM | 440 Hz carrier, 2 → 20 Hz AM-rate sweep | 48 000 Hz |
//! | `stimulus::beats_signal` | S3 beats | 440 + 443 Hz beating pair | 48 000 Hz |
//! | `stimulus::missing_fundamental_signal` | S3 MF | 200–500 Hz harmonics, 100 Hz absent | 48 000 Hz |
//! | `stimulus::output_loss_pair` | F11 | Constant 6 dB output loss + separately normalized display view | rate-independent curves |
//!
//! Fixture IDs name the analytic control from the master plan; QA decides
//! which rows become release gates. IDs owned by other lanes (F02 timing
//! uncertainty, F07/F10/F13/F15 policy verdicts, F12 preference) are not
//! claimed here — this crate supplies inputs and ground truth, never scores.

/// One row of the G3 fixture catalogue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureRecord {
    /// Fully qualified constructor path.
    pub constructor: &'static str,
    /// Master-plan fixture ID (or lane-local control ID).
    pub fixture_id: &'static str,
    /// Planted behavior in one line.
    pub behavior: &'static str,
    /// Documented sample rate, or `None` for rate-independent curve fixtures.
    pub sample_rate_hz: Option<f64>,
}

/// The full constructor × fixture-ID × behavior × rates table for QA.
pub fn fixture_catalog() -> Vec<FixtureRecord> {
    let curves = None;
    let audio = Some(48_000.0);
    vec![
        FixtureRecord {
            constructor: "roomeq_synthetic::timing::clock_drift_fixture",
            fixture_id: "F01",
            behavior: "50 ppm affine drift over 20 s accumulates 1 ms; noisy markers; raw offsets and recentered IRs exposed",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::timing::uncertainty_band_fixture",
            fixture_id: "F06 input",
            behavior: "band-local noise and coherence drop; rest of spectrum bit-identical to ground truth",
            sample_rate_hz: curves,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::timing::coverage_gap_fixture",
            fixture_id: "F05 input",
            behavior: "disjoint coverage gap stays NaN/unsupported; constructor never interpolates across it",
            sample_rate_hz: curves,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::timing::spatial_magnitude_only_sample",
            fixture_id: "F07 input",
            behavior: "spatial magnitude capture with no stationary phase, coherence or sensitivity attached",
            sample_rate_hz: curves,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::timing::calibration_unknown_sample",
            fixture_id: "F15 input",
            behavior: "relative spectrum only; calibration stays unknown with no fabricated sensitivity",
            sample_rate_hz: curves,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::shared_bass_fixture",
            fixture_id: "F03",
            behavior: "equal coherent sources sum to +6.020599913 dB; opposite polarity cancels with no finite score",
            sample_rate_hz: None,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::common_eq_seat_pair",
            fixture_id: "F04",
            behavior: "one common EQ on two seats preserves their relative response difference",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::worse_seat_counterexample",
            fixture_id: "F08",
            behavior: "candidate improves the seat mean while regressing the held-out seat",
            sample_rate_hz: None,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::overlapping_removals_fixture",
            fixture_id: "F09",
            behavior: "two overlapping removals combine super-additively against the frozen full chain",
            sample_rate_hz: curves,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::bass_only_candidate_with_upper_fault",
            fixture_id: "F14",
            behavior: "bass-only fix plus unrelated +5 dB upper-band damage retained in full-band evaluation",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::modal_cut_reference",
            fixture_id: "S2 modal",
            behavior: "matched -9 dB Q=4 minimum-phase modal cut with transfer and IR reference",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::moving_dip_fixture",
            fixture_id: "S2 dip",
            behavior: "deep null whose center moves across seats; label identifies construction",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::spatial::narrow_peak_fixture",
            fixture_id: "S2 peak",
            behavior: "repeatable narrow +9 dB resonance at a fixed center; label identifies construction",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::equal_energy_signal_a/b",
            fixture_id: "F12 control",
            behavior: "equal-energy pair with disjoint spectra; physical energy only, no equivalence claim",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::tilt_signal",
            fixture_id: "S3 tilt",
            behavior: "harmonic-complex spectral tilt control",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::resonance_signal",
            fixture_id: "S3 resonance",
            behavior: "sustained 75 Hz resonance excitation with raised-cosine edges",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::transient_signal",
            fixture_id: "S3 transient",
            behavior: "impulse plus decaying 2 kHz transient burst",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::am_sweep_signal",
            fixture_id: "S3 AM",
            behavior: "440 Hz carrier with 2 Hz to 20 Hz AM-rate sweep",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::beats_signal",
            fixture_id: "S3 beats",
            behavior: "440 Hz + 443 Hz beating-tone pair",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::missing_fundamental_signal",
            fixture_id: "S3 MF",
            behavior: "200-500 Hz harmonic complex with the 100 Hz fundamental absent",
            sample_rate_hz: audio,
        },
        FixtureRecord {
            constructor: "roomeq_synthetic::stimulus::output_loss_pair",
            fixture_id: "F11",
            behavior: "constant 6 dB output loss with a separately normalized display view",
            sample_rate_hz: curves,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_catalog_covers_lane_fixtures() {
        let catalog = fixture_catalog();
        assert!(
            catalog.len() >= 20,
            "catalog must list every lane constructor"
        );
        for record in &catalog {
            assert!(!record.constructor.is_empty());
            assert!(!record.fixture_id.is_empty());
            assert!(!record.behavior.is_empty());
        }
        let ids: Vec<&&str> = catalog.iter().map(|r| &r.fixture_id).collect();
        for required in ["F01", "F03", "F04", "F08", "F09", "F11", "F14"] {
            assert!(
                ids.iter().any(|id| ***id == *required),
                "catalog must cover {required}"
            );
        }
        // Constructors are unique; audio fixtures document 48 kHz.
        let mut ctors: Vec<&&str> = catalog.iter().map(|r| &r.constructor).collect();
        ctors.sort_unstable();
        ctors.dedup();
        assert_eq!(ctors.len(), catalog.len());
        assert!(
            catalog
                .iter()
                .filter(|r| r.sample_rate_hz == Some(48_000.0))
                .count()
                >= 10
        );
    }
}
