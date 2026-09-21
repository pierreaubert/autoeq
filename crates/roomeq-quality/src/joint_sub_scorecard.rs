//! Joint multi-sub scorecard for acceptance reporting.
//!
//! Scores the optimized sub array per seat and as a group: per-seat target
//! error, seat-to-seat variation, unnormalized usable output and required
//! drive, and an explicit prime-seat versus group trade-off note. Output
//! levels are absolute; normalization must never hide output loss (`F11` in
//! `reviews/plan-20260921.md`).

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// Per-seat outcome of the joint sub array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeatSubScore {
    /// Seat index in the transfer matrix.
    pub seat: usize,
    /// Root-mean-square target error over the evaluation band, in dB.
    pub target_error_db: f64,
    /// Mean absolute level before array controls, in dB.
    pub pre_level_db: f64,
    /// Mean absolute level after array controls, in dB.
    pub post_level_db: f64,
    /// Absolute level change (`post - pre`), in dB. Negative means lost output.
    pub output_delta_db: f64,
}

/// Unnormalized group output summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupOutputScore {
    /// Mean absolute pre level across seats, in dB.
    pub pre_level_db: f64,
    /// Mean absolute post level across seats, in dB.
    pub post_level_db: f64,
    /// Absolute group level change, in dB.
    pub delta_db: f64,
    /// Worst per-seat level change, in dB.
    pub worst_seat_delta_db: f64,
}

/// Joint multi-sub scorecard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSubScorecard {
    /// One score per seat, retained in seat order.
    pub per_seat: Vec<SeatSubScore>,
    /// Seat-to-seat level variation after controls, in dB (max-min of post means).
    pub variation_db: f64,
    /// Absolute group output summary.
    pub output: GroupOutputScore,
    /// Mean target error across seats, in dB.
    pub mean_target_error_db: f64,
    /// Worst-seat target error, in dB.
    pub worst_seat_target_error_db: f64,
    /// Honest prime-seat versus group trade-off disclosure.
    pub tradeoff_note: String,
}

/// Evaluate the joint multi-sub scorecard.
///
/// `pre_levels` and `post_levels` hold absolute (unnormalized) per-seat level
/// vectors on one shared grid; `target_db` is the absolute target per bin.
/// Only bins within `[min_freq, max_freq]` participate. `prime_seat` selects
/// the reference seat named in the trade-off note.
///
/// # Examples
///
/// ```
/// use roomeq_quality::evaluate_joint_sub_scorecard;
/// let card = evaluate_joint_sub_scorecard(
///     &[vec![80.0, 81.0]],
///     &[vec![80.0, 81.0]],
///     &[80.0, 81.0],
///     &[20.0, 200.0],
///     20.0,
///     200.0,
///     0,
/// )
/// .unwrap();
/// assert_eq!(card.per_seat.len(), 1);
/// ```
///
/// # Errors
///
/// Returns an error string when seats are missing or ragged, the grid is
/// inconsistent, the band selects no bins, or `prime_seat` is out of range.
pub fn evaluate_joint_sub_scorecard(
    pre_levels: &[Vec<f64>],
    post_levels: &[Vec<f64>],
    target_db: &[f64],
    freqs: &[f64],
    min_freq: f64,
    max_freq: f64,
    prime_seat: usize,
) -> Result<JointSubScorecard, String> {
    if pre_levels.is_empty() || pre_levels.len() != post_levels.len() {
        return Err("scorecard needs matching non-empty pre/post seats".to_string());
    }
    if freqs.len() != target_db.len() || freqs.is_empty() {
        return Err("scorecard needs a non-empty grid matching the target".to_string());
    }
    if prime_seat >= pre_levels.len() {
        return Err("prime seat index out of range".to_string());
    }
    let bins: Vec<usize> = freqs
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency >= min_freq && **frequency <= max_freq)
        .map(|(bin, _)| bin)
        .collect();
    if bins.is_empty() {
        return Err("scorecard band selects no bins".to_string());
    }
    for (seat, (pre, post)) in pre_levels.iter().zip(post_levels.iter()).enumerate() {
        if pre.len() != freqs.len() || post.len() != freqs.len() {
            return Err(format!("seat {seat} levels do not match the grid"));
        }
        if pre.iter().chain(post.iter()).any(|v| !v.is_finite()) {
            return Err(format!("seat {seat} levels must be finite"));
        }
    }

    let band_mean =
        |levels: &[f64]| bins.iter().map(|bin| levels[*bin]).sum::<f64>() / bins.len() as f64;
    let band_rmse = |levels: &[f64]| {
        (bins
            .iter()
            .map(|bin| (levels[*bin] - target_db[*bin]).powi(2))
            .sum::<f64>()
            / bins.len() as f64)
            .sqrt()
    };

    let per_seat: Vec<SeatSubScore> = pre_levels
        .iter()
        .zip(post_levels.iter())
        .enumerate()
        .map(|(seat, (pre, post))| {
            let pre_level = band_mean(pre);
            let post_level = band_mean(post);
            SeatSubScore {
                seat,
                target_error_db: band_rmse(post),
                pre_level_db: pre_level,
                post_level_db: post_level,
                output_delta_db: post_level - pre_level,
            }
        })
        .collect();

    let post_means: Vec<f64> = per_seat.iter().map(|score| score.post_level_db).collect();
    let variation_db = post_means.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
        - post_means.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let pre_group =
        per_seat.iter().map(|score| score.pre_level_db).sum::<f64>() / per_seat.len() as f64;
    let post_group = post_means.iter().sum::<f64>() / post_means.len() as f64;
    let worst_seat_delta_db = per_seat
        .iter()
        .map(|score| score.output_delta_db)
        .fold(f64::INFINITY, f64::min);
    let mean_target_error_db = per_seat
        .iter()
        .map(|score| score.target_error_db)
        .sum::<f64>()
        / per_seat.len() as f64;
    let worst_seat_target_error_db = per_seat
        .iter()
        .map(|score| score.target_error_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let prime = &per_seat[prime_seat];
    let tradeoff_note = format!(
        "prime seat {} target error {:.2} dB with output change {:+.2} dB; group mean target error {:.2} dB, worst seat {:.2} dB, worst output change {:+.2} dB",
        prime.seat,
        prime.target_error_db,
        prime.output_delta_db,
        mean_target_error_db,
        worst_seat_target_error_db,
        worst_seat_delta_db,
    );

    Ok(JointSubScorecard {
        per_seat,
        variation_db,
        output: GroupOutputScore {
            pre_level_db: pre_group,
            post_level_db: post_group,
            delta_db: post_group - pre_group,
            worst_seat_delta_db,
        },
        mean_target_error_db,
        worst_seat_target_error_db,
        tradeoff_note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<f64> {
        vec![30.0, 60.0, 120.0]
    }

    #[test]
    fn joint_scorecard_validates_inputs() {
        let grid = grid();
        assert!(evaluate_joint_sub_scorecard(&[], &[], &[80.0], &grid, 20.0, 200.0, 0).is_err());
        assert!(
            evaluate_joint_sub_scorecard(
                &[vec![80.0, 81.0]],
                &[vec![80.0, 81.0]],
                &[80.0],
                &[30.0],
                20.0,
                200.0,
                0
            )
            .is_err()
        );
        assert!(
            evaluate_joint_sub_scorecard(
                &[vec![80.0, 81.0, 82.0]],
                &[vec![80.0, 81.0, 82.0]],
                &[80.0, 81.0, 82.0],
                &grid,
                20.0,
                200.0,
                3
            )
            .is_err()
        );
        assert!(
            evaluate_joint_sub_scorecard(
                &[vec![80.0, 81.0, 82.0]],
                &[vec![80.0, 81.0, 82.0]],
                &[80.0, 81.0, 82.0],
                &grid,
                500.0,
                600.0,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn joint_scorecard_retains_per_seat_results() {
        let grid = grid();
        let card = evaluate_joint_sub_scorecard(
            &[vec![80.0, 80.0, 80.0], vec![78.0, 78.0, 78.0]],
            &[vec![82.0, 82.0, 82.0], vec![79.0, 79.0, 79.0]],
            &[82.0, 82.0, 82.0],
            &grid,
            20.0,
            200.0,
            0,
        )
        .unwrap();
        assert_eq!(card.per_seat.len(), 2);
        assert_eq!(card.per_seat[0].seat, 0);
        assert_eq!(card.per_seat[1].seat, 1);
        assert!((card.per_seat[0].output_delta_db - 2.0).abs() < 1e-12);
        assert!((card.per_seat[1].output_delta_db - 1.0).abs() < 1e-12);
        assert!((card.variation_db - 3.0).abs() < 1e-12);
        assert!((card.mean_target_error_db - 1.5).abs() < 1e-12);
    }

    /// Unnormalized output loss stays reported even when every seat keeps
    /// its normalized shape (F11).
    #[test]
    fn joint_scorecard_reports_absolute_output_loss() {
        let grid = grid();
        let card = evaluate_joint_sub_scorecard(
            &[vec![80.0, 81.0, 82.0], vec![80.0, 81.0, 82.0]],
            &[vec![74.0, 75.0, 76.0], vec![74.0, 75.0, 76.0]],
            &[80.0, 81.0, 82.0],
            &grid,
            20.0,
            200.0,
            0,
        )
        .unwrap();
        assert!((card.output.delta_db + 6.0).abs() < 1e-9);
        assert!((card.output.worst_seat_delta_db + 6.0).abs() < 1e-9);
        assert!(card.output.post_level_db < card.output.pre_level_db);
    }

    /// Prime-seat versus group trade-offs are disclosed, never averaged away.
    #[test]
    fn joint_scorecard_discloses_prime_versus_group_tradeoff() {
        let grid = grid();
        let card = evaluate_joint_sub_scorecard(
            &[vec![80.0, 80.0, 80.0], vec![80.0, 80.0, 80.0]],
            &[vec![82.0, 82.0, 82.0], vec![70.0, 70.0, 70.0]],
            &[82.0, 82.0, 82.0],
            &grid,
            20.0,
            200.0,
            0,
        )
        .unwrap();
        assert!(card.per_seat[1].output_delta_db < -9.0);
        assert!(card.worst_seat_target_error_db > card.mean_target_error_db);
        assert!(card.tradeoff_note.contains("worst seat"));
        assert!(card.tradeoff_note.contains("prime seat 0"));
    }
}
