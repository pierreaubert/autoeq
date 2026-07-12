//! Measurement-confidence summaries shared by RoomEQ and QA.

use autoeq_core::Curve;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementQuality {
    Good,
    Degraded,
    Poor,
    Unusable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MeasurementQualityReport {
    pub quality: MeasurementQuality,
    pub min_coherence: Option<f64>,
    pub median_coherence: Option<f64>,
    pub median_snr_db: Option<f64>,
    pub mean_seat_variance_db: Option<f64>,
    pub max_seat_variance_db: Option<f64>,
    pub correction_depth_scale: f64,
    pub advisories: Vec<String>,
}

impl MeasurementQualityReport {
    fn unusable(advisory: impl Into<String>) -> Self {
        Self {
            quality: MeasurementQuality::Unusable,
            min_coherence: None,
            median_coherence: None,
            median_snr_db: None,
            mean_seat_variance_db: None,
            max_seat_variance_db: None,
            correction_depth_scale: 0.0,
            advisories: vec![advisory.into()],
        }
    }
}

pub fn assess_measurement_quality(curve: &Curve) -> MeasurementQualityReport {
    if let Err(error) = curve.validate("measurement quality") {
        return MeasurementQualityReport::unusable(error.to_string());
    }

    let coherence = curve.coherence.as_ref().map(|values| {
        let finite: Vec<f64> = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        (
            finite.iter().copied().fold(f64::INFINITY, f64::min),
            median(finite),
        )
    });
    let median_snr_db = curve.noise_floor_db.as_ref().and_then(|noise| {
        (noise.len() == curve.spl.len()).then(|| {
            median(
                curve
                    .spl
                    .iter()
                    .zip(noise)
                    .map(|(spl, floor)| spl - floor)
                    .filter(|value| value.is_finite())
                    .collect(),
            )
        })
    });
    let min_coherence = coherence
        .map(|value| value.0)
        .filter(|value| value.is_finite());
    let median_coherence = coherence
        .map(|value| value.1)
        .filter(|value| value.is_finite());

    let mut advisories = Vec::new();
    let quality = match (median_coherence, median_snr_db) {
        (Some(coherence), Some(snr)) if coherence >= 0.9 && snr >= 20.0 => MeasurementQuality::Good,
        (Some(coherence), Some(snr)) if coherence >= 0.7 && snr >= 10.0 => {
            advisories.push("measurement_confidence_degraded".to_string());
            MeasurementQuality::Degraded
        }
        (Some(_), Some(_)) => {
            advisories.push("measurement_confidence_poor".to_string());
            MeasurementQuality::Poor
        }
        _ => {
            advisories.push("measurement_confidence_metadata_missing".to_string());
            MeasurementQuality::Degraded
        }
    };
    let correction_depth_scale = match quality {
        MeasurementQuality::Good => 1.0,
        MeasurementQuality::Degraded => 0.75,
        MeasurementQuality::Poor => 0.35,
        MeasurementQuality::Unusable => 0.0,
    };

    MeasurementQualityReport {
        quality,
        min_coherence,
        median_coherence,
        median_snr_db,
        mean_seat_variance_db: None,
        max_seat_variance_db: None,
        correction_depth_scale,
        advisories,
    }
}

pub fn assess_multiple_measurement_quality(curves: &[Curve]) -> MeasurementQualityReport {
    let Some(first) = curves.first() else {
        return MeasurementQualityReport::unusable("no_measurements");
    };
    if curves.iter().any(|curve| {
        curve.validate("multi-seat measurement").is_err()
            || curve.freq.len() != first.freq.len()
            || curve
                .freq
                .iter()
                .zip(&first.freq)
                .any(|(left, right)| (left - right).abs() > 1e-9)
    }) {
        return MeasurementQualityReport::unusable("mismatched_measurement_grids");
    }

    let variances: Vec<f64> = (0..first.spl.len())
        .map(|index| {
            let values: Vec<f64> = curves.iter().map(|curve| curve.spl[index]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            (values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64)
                .sqrt()
        })
        .collect();
    let mean_variance = variances.iter().sum::<f64>() / variances.len().max(1) as f64;
    let max_variance = variances.iter().copied().fold(0.0_f64, f64::max);
    let mut report = assess_measurement_quality(first);
    report.mean_seat_variance_db = Some(mean_variance);
    report.max_seat_variance_db = Some(max_variance);
    if max_variance > 6.0 {
        report.quality = report.quality.max(MeasurementQuality::Poor);
        report.correction_depth_scale = report.correction_depth_scale.min(0.35);
        report.advisories.push("high_spatial_variance".to_string());
    } else if max_variance > 3.0 {
        report.quality = report.quality.max(MeasurementQuality::Degraded);
        report.correction_depth_scale = report.correction_depth_scale.min(0.75);
        report
            .advisories
            .push("elevated_spatial_variance".to_string());
    }
    report
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn curve(spl: Vec<f64>) -> Curve {
        Curve {
            freq: Array1::from(vec![20.0, 100.0, 1000.0]),
            spl: Array1::from(spl),
            ..Default::default()
        }
    }

    #[test]
    fn coherent_high_snr_measurement_is_good() {
        let mut input = curve(vec![80.0, 81.0, 79.0]);
        input.coherence = Some(Array1::from(vec![0.95, 0.98, 0.94]));
        input.noise_floor_db = Some(Array1::from(vec![40.0, 42.0, 41.0]));
        let report = assess_measurement_quality(&input);
        assert_eq!(report.quality, MeasurementQuality::Good);
        assert_eq!(report.correction_depth_scale, 1.0);
    }

    #[test]
    fn low_confidence_and_high_variance_limit_correction() {
        let mut left = curve(vec![70.0, 85.0, 70.0]);
        left.coherence = Some(Array1::from(vec![0.4, 0.5, 0.45]));
        left.noise_floor_db = Some(Array1::from(vec![65.0, 80.0, 65.0]));
        let right = curve(vec![85.0, 70.0, 85.0]);
        let report = assess_multiple_measurement_quality(&[left, right]);
        assert_eq!(report.quality, MeasurementQuality::Poor);
        assert!(report.max_seat_variance_db.unwrap() > 6.0);
        assert_eq!(report.correction_depth_scale, 0.35);
    }

    #[test]
    fn mismatched_grids_are_unusable() {
        let left = curve(vec![80.0, 80.0, 80.0]);
        let mut right = curve(vec![80.0, 80.0, 80.0]);
        right.freq[1] = 110.0;
        assert_eq!(
            assess_multiple_measurement_quality(&[left, right]).quality,
            MeasurementQuality::Unusable
        );
    }
}
