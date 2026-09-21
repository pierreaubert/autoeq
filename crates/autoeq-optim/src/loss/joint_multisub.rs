//! Joint transfer-matrix objective for multi-subwoofer optimization.
//!
//! The transfer matrix `H[s,k](f)` maps every independently controlled sub `k`
//! to every relevant seat `s`. The joint loss scalarizes three components
//! over the combined per-seat responses: seat-to-seat variation, unnormalized
//! usable output and required drive, and target error. Shared EQ is applied
//! afterwards to the residual common response only; it cannot change relative
//! seat responses (see `F04` fixture in `reviews/plan-20260921.md`).
//!
//! All output terms use absolute levels. Normalized views must never hide
//! output loss (`F11`).

// Rust guideline compliant 2026-02-21

use super::multisub::array_output_penalty;

/// Scalarization weights for [`joint_multisub_loss`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointSubWeights {
    /// Weight on seat-to-seat variation.
    pub variation: f64,
    /// Weight on unnormalized output and drive cost.
    pub output_drive: f64,
    /// Weight on target error of the seat responses.
    pub target_error: f64,
}

impl JointSubWeights {
    /// Build weights for the joint objective.
    ///
    /// All weights must be finite and non-negative, and at least one must be
    /// positive so the objective can discriminate candidates.
    ///
    /// # Examples
    ///
    /// ```
    /// use autoeq_optim::loss::JointSubWeights;
    /// let weights = JointSubWeights::new(1.0, 1.0, 1.0).unwrap();
    /// assert_eq!(weights.variation, 1.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error string when any weight is nonfinite or negative, or
    /// when every weight is zero.
    pub fn new(variation: f64, output_drive: f64, target_error: f64) -> Result<Self, String> {
        for (name, value) in [
            ("variation", variation),
            ("output_drive", output_drive),
            ("target_error", target_error),
        ] {
            if !value.is_finite() {
                return Err(format!("joint multi-sub weight `{name}` must be finite"));
            }
            if value < 0.0 {
                return Err(format!(
                    "joint multi-sub weight `{name}` must be non-negative"
                ));
            }
        }
        if variation == 0.0 && output_drive == 0.0 && target_error == 0.0 {
            return Err("at least one joint multi-sub weight must be positive".to_string());
        }
        Ok(Self {
            variation,
            output_drive,
            target_error,
        })
    }
}

impl Default for JointSubWeights {
    /// Unit weights: no component is silenced by default.
    ///
    /// A single-component optimum (for example lowest variance alone) must
    /// never be selected on a partial criterion; the default keeps every
    /// term in the scalarization.
    fn default() -> Self {
        Self {
            variation: 1.0,
            output_drive: 1.0,
            target_error: 1.0,
        }
    }
}

/// Decomposed joint loss for one candidate sub array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointSubComponents {
    /// Mean across bins of seat-to-seat level variance (dB^2).
    pub variation: f64,
    /// Unnormalized output and drive penalty (absolute dB cost).
    pub output_drive: f64,
    /// Mean squared target error averaged over seats (dB^2).
    pub target_error: f64,
    /// Weighted scalar total selected by the optimizer.
    pub total: f64,
}

/// Evaluate the joint multi-sub objective.
///
/// `seat_levels` holds one absolute (unnormalized) combined level vector per
/// seat, all on the same frequency grid. `power_reference` holds the absolute
/// measured power-sum reference per bin, and `target_db` the desired absolute
/// response per bin. Every slice must share one length and hold finite values.
///
/// # Examples
///
/// ```
/// use autoeq_optim::loss::{JointSubWeights, joint_multisub_loss};
/// let seats = vec![vec![80.0, 81.0], vec![80.0, 81.0]];
/// let reference = vec![80.0, 81.0];
/// let target = vec![80.0, 81.0];
/// let components =
///     joint_multisub_loss(&seats, &reference, &target, &JointSubWeights::default())
///         .unwrap();
/// assert!(components.total.is_finite());
/// ```
///
/// # Errors
///
/// Returns an error string when inputs are empty, ragged, length-mismatched,
/// or nonfinite.
///
/// # Panics
///
/// Never panics on validated inputs; all indexing is bounds-checked by slice
/// iteration.
pub fn joint_multisub_loss(
    seat_levels: &[Vec<f64>],
    power_reference: &[f64],
    target_db: &[f64],
    weights: &JointSubWeights,
) -> Result<JointSubComponents, String> {
    if seat_levels.is_empty() {
        return Err("joint multi-sub objective needs at least one seat".to_string());
    }
    let bins = power_reference.len();
    if bins == 0 || target_db.len() != bins {
        return Err("reference and target must share a non-empty length".to_string());
    }
    for (seat, levels) in seat_levels.iter().enumerate() {
        if levels.len() != bins {
            return Err(format!(
                "seat {seat} has {} bins, expected {bins}",
                levels.len()
            ));
        }
        if levels.iter().any(|v| !v.is_finite()) {
            return Err(format!("seat {seat} levels must be finite"));
        }
    }
    if power_reference.iter().any(|v| !v.is_finite()) || target_db.iter().any(|v| !v.is_finite()) {
        return Err("reference and target levels must be finite".to_string());
    }

    let seats = seat_levels.len() as f64;
    let mut variation_sum = 0.0;
    let mut target_sum = 0.0;
    for bin in 0..bins {
        let mean = seat_levels.iter().map(|levels| levels[bin]).sum::<f64>() / seats;
        variation_sum += seat_levels
            .iter()
            .map(|levels| (levels[bin] - mean).powi(2))
            .sum::<f64>()
            / seats;
        target_sum += seat_levels
            .iter()
            .map(|levels| (levels[bin] - target_db[bin]).powi(2))
            .sum::<f64>()
            / seats;
    }
    let bins_f = bins as f64;
    let variation = variation_sum / bins_f;
    let target_error = target_sum / bins_f;

    // Output and drive stay unnormalized: the seat-average absolute response
    // is compared against the absolute power-sum reference, so a candidate
    // that only turns everything down cannot hide behind renormalization.
    let average: Vec<f64> = (0..bins)
        .map(|bin| seat_levels.iter().map(|levels| levels[bin]).sum::<f64>() / seats)
        .collect();
    let output_drive = array_output_penalty(&average, power_reference);

    let total = weights.variation * variation
        + weights.output_drive * output_drive
        + weights.target_error * target_error;
    Ok(JointSubComponents {
        variation,
        output_drive,
        target_error,
        total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_weights() -> JointSubWeights {
        JointSubWeights::default()
    }

    #[test]
    fn joint_weights_reject_nonfinite_negative_and_all_zero() {
        assert!(JointSubWeights::new(f64::NAN, 1.0, 1.0).is_err());
        assert!(JointSubWeights::new(1.0, -0.5, 1.0).is_err());
        assert!(JointSubWeights::new(0.0, 0.0, 0.0).is_err());
        assert!(JointSubWeights::new(0.0, 1.0, 0.0).is_ok());
    }

    #[test]
    fn joint_loss_rejects_empty_ragged_and_nonfinite_inputs() {
        let weights = unit_weights();
        assert!(joint_multisub_loss(&[], &[80.0], &[80.0], &weights).is_err());
        assert!(
            joint_multisub_loss(&[vec![80.0, 81.0]], &[80.0], &[80.0, 81.0], &weights).is_err()
        );
        assert!(
            joint_multisub_loss(
                &[vec![80.0, f64::INFINITY]],
                &[80.0, 81.0],
                &[80.0, 81.0],
                &weights
            )
            .is_err()
        );
    }

    #[test]
    fn joint_identical_on_target_seats_score_zero() {
        let seats = vec![vec![80.0, 81.0, 82.0], vec![80.0, 81.0, 82.0]];
        let reference = vec![80.0, 81.0, 82.0];
        let target = vec![80.0, 81.0, 82.0];
        let components = joint_multisub_loss(&seats, &reference, &target, &unit_weights()).unwrap();
        assert_eq!(components.variation, 0.0);
        assert_eq!(components.target_error, 0.0);
        assert_eq!(components.output_drive, 0.0);
        assert_eq!(components.total, 0.0);
    }

    /// Lowest variance alone is not the winner: a perfectly consistent but
    /// deeply attenuated array must lose to a slightly varied array that
    /// preserves usable output.
    #[test]
    fn joint_lowest_variance_alone_is_not_winner() {
        let reference = vec![80.0, 80.0, 80.0];
        let target = vec![80.0, 80.0, 80.0];
        let quiet = vec![vec![68.0, 68.0, 68.0], vec![68.0, 68.0, 68.0]];
        let lively = vec![vec![80.0, 80.5, 79.5], vec![79.5, 80.0, 80.5]];
        let quiet_loss = joint_multisub_loss(&quiet, &reference, &target, &unit_weights()).unwrap();
        let lively_loss =
            joint_multisub_loss(&lively, &reference, &target, &unit_weights()).unwrap();
        assert_eq!(quiet_loss.variation, 0.0);
        assert!(lively_loss.variation > 0.0);
        assert!(
            quiet_loss.total > lively_loss.total,
            "zero-variance candidate must not win on variation alone"
        );
    }

    /// Normalized graphs cannot hide output loss: two candidates with the
    /// same shape but different absolute levels score differently.
    #[test]
    fn joint_unnormalized_output_loss_stays_visible() {
        let reference = vec![80.0, 81.0, 82.0];
        let target = vec![80.0, 81.0, 82.0];
        let full = vec![vec![80.0, 81.0, 82.0], vec![80.0, 81.0, 82.0]];
        let turned_down = vec![vec![74.0, 75.0, 76.0], vec![74.0, 75.0, 76.0]];
        let full_loss = joint_multisub_loss(&full, &reference, &target, &unit_weights()).unwrap();
        let down_loss =
            joint_multisub_loss(&turned_down, &reference, &target, &unit_weights()).unwrap();
        assert_eq!(full_loss.variation, down_loss.variation);
        assert!(
            down_loss.output_drive > full_loss.output_drive,
            "common attenuation must stay visible in the output term"
        );
        assert!(down_loss.total > full_loss.total);
    }
}
