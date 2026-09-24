//! Post-realization authorization for multi-segment measurement support.
//!
//! When a channel declares disjoint usable bands, correction and scoring
//! consume the union of measured segments: gap bins are absent from the
//! usable curve, so the optimizer never fits them and range-gated scores
//! never consume them. This module replays the realized correction to check
//! the two remaining boundaries:
//!
//! - every reported PEQ center lies inside a declared segment (exact
//!   containment — a filter parameterized by gap frequencies corrects
//!   nothing measured, so the channel is refused);
//! - the realized PEQ+FIR correction transfer inside each coverage gap is
//!   evaluated and reported as observed evidence. No inaudibility threshold
//!   gates it: per-segment scores below gate acceptance instead;
//! - flatness is scored per segment so a regressing segment cannot hide
//!   behind union improvement.
//!
//! Single-band and unrestricted channels return `None`: their behavior is
//! unchanged.

use autoeq_core::{AutoeqError, Curve, Result, response};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::PreparedChannelInput;
use crate::channel_target::flatness_score_in_range;

/// Synthetic evaluation points per gap when the raw grid holds fewer than
/// two bins strictly inside that gap. The transfer is analytic, so these
/// points are evaluation frequencies, not fabricated measurements.
const SYNTHETIC_GAP_POINTS: usize = 8;

/// Flatness evidence for one declared usable segment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentScore {
    /// Declared usable segment in Hz.
    pub band_hz: [f64; 2],
    /// Measured bins retained in this segment.
    pub bins: usize,
    /// Flatness of the pre-EQ response over this segment.
    pub pre_score: f64,
    /// Flatness of the realized post-EQ response over this segment.
    pub post_score: f64,
}

/// Post-realization authorization report for multi-segment support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentSupportReport {
    /// Declared usable segments in Hz.
    pub bands_hz: Vec<[f64; 2]>,
    /// Per-segment flatness evidence, in declared order.
    pub segments: Vec<SegmentScore>,
    /// Maximum |correction| in dB of the realized PEQ+FIR correction
    /// transfer evaluated at gap frequencies. Observed evidence for the
    /// report; acceptance is gated by per-segment scores, not by an
    /// invented inaudibility threshold.
    pub gap_leakage_db_max: f64,
    /// Gap frequencies at which the transfer was evaluated.
    pub gap_points: usize,
}

/// Check realized multi-segment authorization for one processed channel.
///
/// Returns `None` when fewer than two usable segments are declared. Fails
/// when a PEQ center lies in an unmeasured gap, when an authorized segment
/// cannot be scored, or when the gap transfer is non-finite.
///
/// Each segment is scored over its full declared band, independent of the
/// configured correction band: measured support stays reportable even where
/// the operator chose not to correct. `raw_pre_eq_curve` supplies the
/// evaluation grid (its gap bins are frequency points only; gap SPL values
/// are never consumed).
#[allow(clippy::too_many_arguments)]
pub fn assess_segment_support(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    raw_pre_eq_curve: &Curve,
    raw_post_eq_curve: &Curve,
    filters: &[Biquad],
    fir_coeffs: Option<&[f64]>,
    sample_rate: f64,
) -> Result<Option<SegmentSupportReport>> {
    let bands = prepared.valid_bands_hz();
    if bands.len() < 2 {
        return Ok(None);
    }
    refuse_gap_centered_filters(channel_name, prepared, filters)?;
    let mut segments = Vec::with_capacity(bands.len());
    for band in bands {
        segments.push(segment_score(
            channel_name,
            raw_pre_eq_curve,
            raw_post_eq_curve,
            *band,
        )?);
    }
    let (gap_leakage_db_max, gap_points) = gap_leakage(
        channel_name,
        prepared,
        raw_pre_eq_curve,
        filters,
        fir_coeffs,
        sample_rate,
    )?;
    Ok(Some(SegmentSupportReport {
        bands_hz: bands.to_vec(),
        segments,
        gap_leakage_db_max,
        gap_points,
    }))
}

/// Refuse correction filters parameterized by unmeasured gap frequencies.
fn refuse_gap_centered_filters(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    filters: &[Biquad],
) -> Result<()> {
    for filter in filters {
        if prepared.supports_frequency(filter.freq) {
            continue;
        }
        let gap = prepared
            .gap_intervals_hz()
            .into_iter()
            .find(|[low, high]| filter.freq > *low && filter.freq < *high);
        return Err(AutoeqError::OptimizationFailed {
            message: match gap {
                Some([low, high]) => format!(
                    "channel '{channel_name}': correction filter centered at {:.1} Hz lies in the unmeasured coverage gap {:.1}-{:.1} Hz; disjoint support authorizes correction only inside declared segments",
                    filter.freq, low, high,
                ),
                None => format!(
                    "channel '{channel_name}': correction filter centered at {:.1} Hz lies outside declared usable support",
                    filter.freq,
                ),
            },
        });
    }
    Ok(())
}

/// Score one authorized segment on measured bins only.
fn segment_score(
    channel_name: &str,
    raw_pre_eq_curve: &Curve,
    raw_post_eq_curve: &Curve,
    band: [f64; 2],
) -> Result<SegmentScore> {
    let [low, high] = band;
    let pre = raw_pre_eq_curve
        .select_frequency_band(band)
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!(
                "channel '{channel_name}': cannot score segment {low:.1}-{high:.1} Hz: {error}"
            ),
        })?;
    let post = raw_post_eq_curve
        .select_frequency_band(band)
        .map_err(|error| AutoeqError::OptimizationFailed {
            message: format!(
                "channel '{channel_name}': cannot score realized segment {low:.1}-{high:.1} Hz: {error}"
            ),
        })?;
    for (label, curve) in [("pre-EQ", &pre), ("post-EQ", &post)] {
        if !curve.spl.iter().all(|value| value.is_finite()) {
            return Err(AutoeqError::OptimizationFailed {
                message: format!(
                    "channel '{channel_name}': non-finite {label} response in segment {low:.1}-{high:.1} Hz"
                ),
            });
        }
    }
    Ok(SegmentScore {
        band_hz: band,
        bins: pre.freq.len(),
        pre_score: flatness_score_in_range(&pre, low, high),
        post_score: flatness_score_in_range(&post, low, high),
    })
}

/// Evaluate the realized PEQ+FIR correction transfer at gap frequencies.
fn gap_leakage(
    channel_name: &str,
    prepared: &PreparedChannelInput,
    raw_pre_eq_curve: &Curve,
    filters: &[Biquad],
    fir_coeffs: Option<&[f64]>,
    sample_rate: f64,
) -> Result<(f64, usize)> {
    let mut points: Vec<f64> = Vec::new();
    for [low, high] in prepared.gap_intervals_hz() {
        let mut inside: Vec<f64> = raw_pre_eq_curve
            .freq
            .iter()
            .copied()
            .filter(|f| *f > low && *f < high)
            .collect();
        if inside.len() < 2 {
            inside = (0..SYNTHETIC_GAP_POINTS)
                .map(|i| {
                    let t = (i + 1) as f64 / (SYNTHETIC_GAP_POINTS + 1) as f64;
                    low * (high / low).powf(t)
                })
                .collect();
        }
        points.extend(inside);
    }
    if points.is_empty() {
        return Ok((0.0, 0));
    }
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(AutoeqError::OptimizationFailed {
            message: format!(
                "channel '{channel_name}': cannot evaluate gap transfer at a non-finite sample rate"
            ),
        });
    }
    let freqs = Array1::from_vec(points);
    let peq = response::compute_peq_complex_response(filters, &freqs, sample_rate);
    let fir = fir_coeffs
        .map(|coeffs| response::compute_fir_complex_response(coeffs, &freqs, sample_rate));
    let mut leakage_db_max = 0.0_f64;
    for (index, peq_bin) in peq.iter().enumerate() {
        let mut bin = *peq_bin;
        if let Some(fir) = fir.as_ref() {
            bin *= fir[index];
        }
        let magnitude = bin.norm();
        if !magnitude.is_finite() || magnitude <= 0.0 {
            return Err(AutoeqError::OptimizationFailed {
                message: format!(
                    "channel '{channel_name}': non-finite realized correction transfer in an unmeasured coverage gap"
                ),
            });
        }
        leakage_db_max = leakage_db_max.max((20.0 * magnitude.log10()).abs());
    }
    if !leakage_db_max.is_finite() {
        return Err(AutoeqError::OptimizationFailed {
            message: format!("channel '{channel_name}': non-finite gap leakage observation"),
        });
    }
    Ok((leakage_db_max, freqs.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PreparedChannelMeasurements;

    fn curve() -> Curve {
        Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1_000.0, 8_000.0, 20_000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0, 99.0, 79.0, 78.0]),
            ..Curve::default()
        }
    }

    fn prepared() -> PreparedChannelInput {
        let curve = curve();
        PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve],
            false,
        ))
        .with_valid_bands_hz(&[[20.0, 100.0], [8_000.0, 20_000.0]])
        .unwrap()
    }

    fn peaking(freq: f64) -> Biquad {
        Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            freq,
            48_000.0,
            1.0,
            3.0,
        )
    }

    #[test]
    fn unrestricted_and_single_band_support_need_no_report() {
        let curve = curve();
        let plain = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve.clone()],
            false,
        ));
        assert!(
            assess_segment_support("left", &plain, &curve, &curve, &[], None, 48_000.0)
                .unwrap()
                .is_none()
        );
        let single = PreparedChannelInput::from_measurements(PreparedChannelMeasurements::new(
            curve.clone(),
            vec![curve.clone()],
            false,
        ))
        .with_valid_band_hz([20.0, 20_000.0])
        .unwrap();
        assert!(
            assess_segment_support("left", &single, &curve, &curve, &[], None, 48_000.0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn gap_centered_filter_refuses_the_channel() {
        let curve = curve();
        let error = assess_segment_support(
            "left",
            &prepared(),
            &curve,
            &curve,
            &[peaking(1_000.0)],
            None,
            48_000.0,
        )
        .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("1,000") || message.contains("1000"),
            "{message}"
        );
        assert!(message.contains("unmeasured coverage gap"), "{message}");
    }

    #[test]
    fn contained_filters_score_per_segment_and_observe_leakage() {
        let pre = curve();
        let mut post = curve();
        // A 1 dB correction inside each segment only.
        post.spl = Array1::from_vec(vec![79.0, 80.0, 99.0, 78.0, 77.0]);
        let report = assess_segment_support(
            "left",
            &prepared(),
            &pre,
            &post,
            &[peaking(50.0), peaking(10_000.0)],
            None,
            48_000.0,
        )
        .unwrap()
        .expect("multi-segment support reports");
        assert_eq!(report.bands_hz, vec![[20.0, 100.0], [8_000.0, 20_000.0]]);
        assert_eq!(report.segments.len(), 2);
        assert_eq!(report.segments[0].bins, 2);
        assert_eq!(report.segments[1].bins, 2);
        for segment in &report.segments {
            assert!(segment.pre_score.is_finite(), "{segment:?}");
            assert!(segment.post_score.is_finite(), "{segment:?}");
        }
        // One raw gap bin cannot sample the transfer shape, so the
        // deterministic synthetic log-spaced gap grid evaluates it; the two
        // contained peaking filters leak a finite, observed amount there.
        assert_eq!(report.gap_points, SYNTHETIC_GAP_POINTS);
        assert!(report.gap_leakage_db_max.is_finite());
        assert!(report.gap_leakage_db_max >= 0.0);
    }

    #[test]
    fn raw_gap_bins_are_the_evaluation_grid_when_present() {
        let pre = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 500.0, 2_000.0, 8_000.0, 20_000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0, 82.0, 83.0, 79.0, 78.0]),
            ..Curve::default()
        };
        let report = assess_segment_support(
            "left",
            &prepared(),
            &pre,
            &pre,
            &[peaking(50.0)],
            None,
            48_000.0,
        )
        .unwrap()
        .expect("multi-segment support reports");
        // Both raw gap bins evaluate the transfer; no synthetic points.
        assert_eq!(report.gap_points, 2);
        assert!(report.gap_leakage_db_max.is_finite());
    }

    #[test]
    fn empty_gap_grid_falls_back_to_synthetic_points() {
        let pre = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 8_000.0, 20_000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0, 79.0, 78.0]),
            ..Curve::default()
        };
        let report = assess_segment_support("left", &prepared(), &pre, &pre, &[], None, 48_000.0)
            .unwrap()
            .expect("multi-segment support reports");
        assert_eq!(report.gap_points, SYNTHETIC_GAP_POINTS);
        assert_eq!(report.gap_leakage_db_max, 0.0);
    }
}
