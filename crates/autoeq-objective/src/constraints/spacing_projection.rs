//! Minimum-distance projection for filter centers in log-frequency space.

use crate::PeqModel;
use crate::param_utils;
use std::collections::HashMap;
use std::time::{Duration as WallDuration, Instant};

#[derive(Clone, Copy)]
struct Block {
    start: usize,
    end: usize,
    sum: f64,
    lower: f64,
    upper: f64,
    value: f64,
}

fn next_up(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value == 0.0 {
        f64::from_bits((1_u64 << 63) | 1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn octave_distance(lower_log10: f64, upper_log10: f64) -> f64 {
    (10.0_f64.powf(upper_log10) / 10.0_f64.powf(lower_log10)).log2()
}

/// Return a bounded log-frequency vector with the required pairwise spacing.
///
/// Filter groups stay in their original vector slots. Only center frequencies
/// may change. For a fixed frequency order the isotonic projection minimizes
/// squared movement; when that order is infeasible, another order may be used.
/// A greedy earliest placement backstops pinned isotonic repairs before the
/// order is abandoned. Bounds with equal endpoints remain fixed.
///
/// # Errors
///
/// Returns an error for malformed inputs or infeasible frequency bounds.
pub fn project_min_spacing(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    model: PeqModel,
    min_spacing_oct: f64,
) -> Result<Vec<f64>, String> {
    if x.len() != lower.len() || x.len() != upper.len() {
        return Err(String::from(
            "spacing projection: parameter and bound lengths differ",
        ));
    }
    let width = param_utils::params_per_filter(model);
    if !x.len().is_multiple_of(width) || !min_spacing_oct.is_finite() || min_spacing_oct < 0.0 {
        return Err(String::from(
            "spacing projection: invalid vector or spacing",
        ));
    }
    let count = x.len() / width;
    let freq_offset = if width == 3 { 0 } else { 1 };
    let mut indices: Vec<usize> = (0..count).map(|i| i * width + freq_offset).collect();
    if indices.iter().any(|&i| {
        !x[i].is_finite()
            || !lower[i].is_finite()
            || !upper[i].is_finite()
            || lower[i] > upper[i]
            || x[i] < lower[i]
            || x[i] > upper[i]
    }) {
        return Err(String::from(
            "spacing projection: frequency outside finite bounds",
        ));
    }
    if count < 2 || min_spacing_oct == 0.0 {
        return Ok(x.to_vec());
    }
    indices.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    if super::min_spacing::viol_spacing_from_xs(x, model, min_spacing_oct) == 0.0 {
        return Ok(x.to_vec());
    }
    match project_in_order(x, lower, upper, model, min_spacing_oct, &indices) {
        Ok(repaired) => Ok(repaired),
        Err(_) => {
            // Deadline order is inexpensive for large, mostly unconstrained
            // layouts. The sparse exact search below remains the fallback.
            let mut deadline_order = indices.clone();
            deadline_order.sort_by(|&a, &b| upper[a].total_cmp(&upper[b]).then(a.cmp(&b)));
            if let Ok(repaired) =
                project_in_order(x, lower, upper, model, min_spacing_oct, &deadline_order)
            {
                return Ok(repaired);
            }
            let alternate = feasible_order(&indices, lower, upper, min_spacing_oct)?;
            project_in_order(x, lower, upper, model, min_spacing_oct, &alternate)
        }
    }
}

fn project_in_order_pava(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    model: PeqModel,
    min_spacing_oct: f64,
    indices: &[usize],
) -> Result<Vec<f64>, String> {
    let count = indices.len();
    let minimum_gap = min_spacing_oct * std::f64::consts::LOG10_2;

    // Subtracting rank * gap converts spacing to nondecreasing isotonic order.
    // Use the exact requested gap; an added margin would reject otherwise
    // feasible candidates with tight or fixed frequency bounds.
    let gap = minimum_gap;
    let mut blocks: Vec<Block> = Vec::with_capacity(count);
    for (rank, &index) in indices.iter().enumerate() {
        let shift = rank as f64 * gap;
        let lower_bound = lower[index] - shift;
        let upper_bound = upper[index] - shift;
        let value = (x[index] - shift).clamp(lower_bound, upper_bound);
        blocks.push(Block {
            start: rank,
            end: rank + 1,
            sum: x[index] - shift,
            lower: lower_bound,
            upper: upper_bound,
            value,
        });
        while blocks.len() >= 2 {
            let len = blocks.len();
            if blocks[len - 2].value <= blocks[len - 1].value {
                break;
            }
            let right = blocks.pop().expect("two blocks exist");
            let left = blocks.pop().expect("two blocks exist");
            let lower_bound = left.lower.max(right.lower);
            let upper_bound = left.upper.min(right.upper);
            if lower_bound > upper_bound {
                return Err(String::from(
                    "spacing projection: frequency bounds are infeasible",
                ));
            }
            let sum = left.sum + right.sum;
            let size = right.end - left.start;
            blocks.push(Block {
                start: left.start,
                end: right.end,
                sum,
                lower: lower_bound,
                upper: upper_bound,
                value: (sum / size as f64).clamp(lower_bound, upper_bound),
            });
        }
    }
    let mut repaired = x.to_vec();
    for block in blocks {
        for (rank, &index) in indices.iter().enumerate().take(block.end).skip(block.start) {
            // Subtracting and adding the rank shift can move a fixed center
            // by one ULP. Restore the exact per-filter box before checking
            // decoded spacing, so fixed filters remain bit-identical.
            repaired[index] = (block.value + rank as f64 * gap).clamp(lower[index], upper[index]);
        }
    }

    // The production constraint decodes log-frequency before comparing
    // octaves. Correct only representational rounding, one adjacent float
    // at a time, without changing the requested minimum spacing.
    for _ in 0..(16 * count) {
        let violating = indices
            .windows(2)
            .find(|pair| octave_distance(repaired[pair[0]], repaired[pair[1]]) < min_spacing_oct);
        let Some(pair) = violating else {
            break;
        };
        let left = pair[0];
        let right = pair[1];
        let raised = next_up(repaired[right]);
        if raised <= upper[right] {
            repaired[right] = raised;
            continue;
        }
        let lowered = next_down(repaired[left]);
        if lowered >= lower[left] {
            repaired[left] = lowered;
            continue;
        }
        return Err(String::from(
            "spacing projection: decoded frequencies cannot meet spacing within bounds",
        ));
    }
    if indices
        .iter()
        .any(|&i| repaired[i] < lower[i] || repaired[i] > upper[i])
        || super::min_spacing::viol_spacing_from_xs(&repaired, model, min_spacing_oct) > 0.0
    {
        return Err(String::from(
            "spacing projection: rounded result is infeasible",
        ));
    }
    Ok(repaired)
}

/// Verify a repaired placement against the exact production constraint.
fn placement_is_feasible(
    repaired: &[f64],
    lower: &[f64],
    upper: &[f64],
    indices: &[usize],
    model: PeqModel,
    min_spacing_oct: f64,
) -> bool {
    indices
        .iter()
        .all(|&i| repaired[i] >= lower[i] && repaired[i] <= upper[i])
        && super::min_spacing::viol_spacing_from_xs(repaired, model, min_spacing_oct) == 0.0
}

/// Earliest feasible placement for a fixed filter order.
///
/// PAVA minimizes movement but can pin blocks against box edges where the
/// ULP nudger gives up. Earliest placement is complete for difference
/// constraints: if any placement satisfies this order and the boxes, the
/// earliest one does. Decoded octave distance (the production constraint)
/// decides every gap, never the log10 approximation.
fn greedy_placement_in_order(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    indices: &[usize],
    min_spacing_oct: f64,
) -> Result<Vec<f64>, String> {
    /// ULP nudges bridging log10-gap/decoded-octave rounding per pair.
    /// Mathematical gaps need only a few; exhaustion means unreachable.
    const MAX_NUDGE: usize = 1024;
    let gap = min_spacing_oct * std::f64::consts::LOG10_2;
    let mut repaired = x.to_vec();
    let mut prev: Option<f64> = None;
    for &index in indices {
        let value = match prev {
            None => lower[index],
            Some(p) => {
                let mut v = (p + gap).max(lower[index]);
                let mut nudges = 0;
                while octave_distance(p, v) < min_spacing_oct {
                    v = next_up(v);
                    nudges += 1;
                    if nudges > MAX_NUDGE || v > upper[index] {
                        return Err(String::from(
                            "spacing projection: greedy repair cannot meet spacing within bounds",
                        ));
                    }
                }
                v
            }
        };
        if value > upper[index] {
            return Err(String::from(
                "spacing projection: greedy repair cannot meet spacing within bounds",
            ));
        }
        repaired[index] = value;
        prev = Some(value);
    }
    Ok(repaired)
}

fn project_in_order(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    model: PeqModel,
    min_spacing_oct: f64,
    indices: &[usize],
) -> Result<Vec<f64>, String> {
    project_in_order_pava(x, lower, upper, model, min_spacing_oct, indices).or_else(|original| {
        greedy_placement_in_order(x, lower, upper, indices, min_spacing_oct)
            .ok()
            .filter(|candidate| {
                placement_is_feasible(candidate, lower, upper, indices, model, min_spacing_oct)
            })
            .ok_or(original)
    })
}

/// Reports whether a projector error proves the filter count infeasible.
///
/// Definite errors let the adaptive search back off to fewer filters.
/// Search-budget ("feasibility undetermined") and malformed-input errors
/// stay strict.
pub fn is_definite_spacing_infeasibility(reason: &str) -> bool {
    const DEFINITE: [&str; 4] = [
        "spacing projection: frequency bounds are infeasible",
        "spacing projection: no ordering satisfies the frequency bounds",
        "spacing projection: decoded frequencies cannot meet spacing within bounds",
        "spacing projection: rounded result is infeasible",
    ];
    DEFINITE.iter().any(|suffix| reason.contains(suffix))
}

/// Find a feasible order of the existing filter identities when their current
/// frequency order cannot satisfy the individual frequency boxes. Each state
/// retains the earliest reachable last center for a subset, which dominates
/// all later centers for the same subset. No filter type, Q, or gain is moved.
fn feasible_order(
    preferred: &[usize],
    lower: &[f64],
    upper: &[f64],
    min_spacing_oct: f64,
) -> Result<Vec<usize>, String> {
    // The exact subset search is only reached after both inexpensive orders
    // fail. Bound time and memory independently of filter count. Exhaustion
    // is an undetermined search, never a claim that the boxes are infeasible.
    const MAX_STATES: usize = 1_000_000;
    const MAX_SEARCH_TIME: WallDuration = WallDuration::from_secs(5);
    let started = Instant::now();
    let count = preferred.len();
    let gap = min_spacing_oct * std::f64::consts::LOG10_2;
    #[derive(Clone)]
    struct State {
        last: f64,
        prior: Vec<u64>,
        chosen: usize,
    }
    let words = count.div_ceil(u64::BITS as usize);
    let empty = vec![0_u64; words];
    let mut states: HashMap<Vec<u64>, State> = HashMap::new();
    states.insert(
        empty.clone(),
        State {
            last: f64::NEG_INFINITY,
            prior: empty.clone(),
            chosen: 0,
        },
    );
    let mut frontier = vec![empty];
    for rank in 0..count {
        if started.elapsed() >= MAX_SEARCH_TIME {
            return Err(String::from(
                "spacing projection: alternate-order search time limit reached; feasibility undetermined",
            ));
        }
        let mut next_frontier = Vec::new();
        for mask in frontier {
            if started.elapsed() >= MAX_SEARCH_TIME {
                return Err(String::from(
                    "spacing projection: alternate-order search time limit reached; feasibility undetermined",
                ));
            }
            let last = states[&mask].last;
            for (choice, &index) in preferred.iter().enumerate() {
                if mask[choice / 64] & (1_u64 << (choice % 64)) != 0 {
                    continue;
                }
                let next = if rank == 0 {
                    lower[index]
                } else {
                    lower[index].max(last + gap)
                };
                if next > upper[index] {
                    continue;
                }
                let mut next_mask = mask.clone();
                next_mask[choice / 64] |= 1_u64 << (choice % 64);
                // Even if every remaining center can be placed at its earliest
                // possible position, its deadline must leave enough room for
                // all earlier deadlines. This is necessary for any completion,
                // independent of the eventual ordering or lower bounds.
                let mut remaining_deadlines: Vec<f64> = preferred
                    .iter()
                    .enumerate()
                    .filter_map(|(remaining_choice, &remaining_index)| {
                        (next_mask[remaining_choice / 64] & (1_u64 << (remaining_choice % 64)) == 0)
                            .then_some(upper[remaining_index])
                    })
                    .collect();
                remaining_deadlines.sort_by(f64::total_cmp);
                // Repeated addition uses the same operation as the actual
                // placement. Multiplying the gap by the offset can round up
                // and incorrectly discard a tight, representable solution.
                let mut earliest = next;
                if remaining_deadlines.iter().any(|&deadline| {
                    earliest += gap;
                    deadline < earliest
                }) {
                    continue;
                }
                match states.get_mut(&next_mask) {
                    Some(old) if next < old.last => {
                        *old = State {
                            last: next,
                            prior: mask.clone(),
                            chosen: choice,
                        };
                    }
                    Some(_) => {}
                    None => {
                        if states.len() >= MAX_STATES {
                            return Err(String::from(
                                "spacing projection: alternate-order search state limit reached; feasibility undetermined",
                            ));
                        }
                        next_frontier.push(next_mask.clone());
                        states.insert(
                            next_mask,
                            State {
                                last: next,
                                prior: mask.clone(),
                                chosen: choice,
                            },
                        );
                    }
                }
            }
        }
        frontier = next_frontier;
    }
    let mut full = vec![u64::MAX; words];
    let remainder = count % 64;
    if !count.is_multiple_of(64) {
        full[words - 1] = (1_u64 << remainder) - 1;
    }
    if !states.contains_key(&full) {
        return Err(String::from(
            "spacing projection: no ordering satisfies the frequency bounds",
        ));
    }
    let mut order = Vec::with_capacity(count);
    let mut mask = full;
    while mask != vec![0; words] {
        let state = &states[&mask];
        order.push(preferred[state.chosen]);
        mask = state.prior.clone();
    }
    order.reverse();
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_close_peaks_without_moving_q_or_gain() {
        let x = [2.0, 1.2, 3.0, 2.05, 2.0, -2.0];
        let lower = [1.0, 1.0, -6.0, 1.0, 1.0, -6.0];
        let upper = [3.0, 6.0, 6.0, 3.0, 6.0, 6.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap();
        assert_eq!(repaired[1], x[1]);
        assert_eq!(repaired[2], x[2]);
        assert_eq!(repaired[4], x[4]);
        assert_eq!(repaired[5], x[5]);
        assert!(repaired[0] < repaired[3]);
        assert!((repaired[3] - repaired[0]) / std::f64::consts::LOG10_2 >= 0.2);
    }

    #[test]
    fn respects_fixed_hp_and_lp_frequency_bounds() {
        let x = [1.7, 1.0, 0.0, 1.71, 1.0, 2.0, 3.8, 1.0, 0.0];
        let lower = [1.7, 1.0, 0.0, 1.0, 1.0, -6.0, 3.8, 1.0, 0.0];
        let upper = [1.7, 1.5, 0.0, 3.0, 6.0, 6.0, 3.8, 1.5, 0.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::HpPkLp, 0.2).unwrap();
        assert_eq!(repaired[0], x[0]);
        assert_eq!(repaired[6], x[6]);
        assert!(repaired[3] > x[3]);
    }

    #[test]
    fn permits_movable_hp_and_lp_centers() {
        let x = [1.7, 1.0, 0.0, 1.71, 1.0, 2.0, 1.72, 1.0, 0.0];
        let lower = [1.0, 1.0, 0.0, 1.0, 1.0, -6.0, 1.0, 1.0, 0.0];
        let upper = [3.0, 1.5, 0.0, 3.0, 6.0, 6.0, 3.0, 1.5, 0.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::HpPkLp, 0.2).unwrap();
        assert_ne!(repaired[0], x[0]);
        assert_ne!(repaired[6], x[6]);
    }

    #[test]
    fn rejects_infeasible_frequency_boxes() {
        let x = [2.0, 1.0, 1.0, 2.01, 1.0, 1.0];
        let lower = [2.0, 1.0, 1.0, 2.01, 1.0, 1.0];
        assert!(project_min_spacing(&x, &lower, &lower, PeqModel::Pk, 0.2).is_err());
    }

    #[test]
    fn finds_feasible_order_without_swapping_filter_attributes() {
        let x = [2.0, 1.0, 0.0, 2.06, 2.0, 1.0, 2.07, 3.0, -1.0];
        let lower = [2.0, 1.0, 0.0, 2.06, 2.0, 1.0, 2.07, 3.0, -1.0];
        let upper = [2.0, 1.0, 0.0, 2.3, 2.0, 1.0, 2.07, 3.0, -1.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap();
        assert!(repaired[3] > repaired[6]);
        assert_eq!(repaired[4], x[4]);
        assert_eq!(repaired[5], x[5]);
        assert_eq!(repaired[7], x[7]);
        assert_eq!(repaired[8], x[8]);
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::Pk, 0.2),
            0.0
        );
    }

    #[test]
    fn alternate_order_remains_available_above_twenty_filters() {
        let count = 25;
        let mut x = Vec::with_capacity(3 * count);
        let mut lower = Vec::with_capacity(3 * count);
        let mut upper = Vec::with_capacity(3 * count);
        for slot in 0..count {
            let (center, high) = match slot {
                0 => (2.0, 2.0),
                1 => (2.06, 2.3),
                2 => (2.07, 2.07),
                _ => {
                    let fixed = 2.25 + (slot - 3) as f64 * 0.07;
                    (fixed, fixed)
                }
            };
            x.extend_from_slice(&[center, 1.0, slot as f64]);
            lower.extend_from_slice(&[center, 1.0, slot as f64]);
            upper.extend_from_slice(&[high, 1.0, slot as f64]);
        }
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap();
        assert!(repaired[3] > repaired[6]);
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::Pk, 0.2),
            0.0
        );
        for slot in 0..count {
            assert_eq!(
                &repaired[slot * 3 + 1..slot * 3 + 3],
                &x[slot * 3 + 1..slot * 3 + 3]
            );
            if slot != 1 {
                assert_eq!(repaired[slot * 3].to_bits(), x[slot * 3].to_bits());
            }
        }
    }

    #[test]
    fn sparse_order_search_supports_multiple_mask_words() {
        let count = 65;
        let mut preferred = Vec::with_capacity(count);
        let mut bounds = Vec::with_capacity(3 * count);
        for slot in 0..count {
            let center = 1.0 + slot as f64 * 0.07;
            preferred.push(3 * slot);
            bounds.extend_from_slice(&[center, 1.0, 0.0]);
        }
        assert_eq!(
            feasible_order(&preferred, &bounds, &bounds, 0.2).unwrap(),
            preferred
        );
    }

    #[test]
    fn deadline_pruning_rejects_an_order_that_strands_fixed_centers() {
        let gap = 0.2 * std::f64::consts::LOG10_2;
        let preferred = [0, 3, 6];
        let lower = [0.0, 1.0, 0.0, gap, 1.0, 0.0, 2.0 * gap, 1.0, 0.0];
        let upper = lower;
        // Choosing the final fixed center first leaves two earlier deadlines
        // impossible, but the complete feasible order must still be found.
        assert_eq!(
            feasible_order(&preferred, &lower, &upper, 0.2).unwrap(),
            preferred
        );
    }

    #[test]
    fn tight_fixed_deadlines_keep_a_feasible_alternate_order() {
        let gap = 0.2 * std::f64::consts::LOG10_2;
        let first = 0.1_f64;
        let second = first + gap;
        let third = second + gap;
        let preferred = [0, 3, 6];
        let bounds = [first, 1.0, 0.0, third, 1.0, 0.0, second, 1.0, 0.0];
        assert_eq!(
            feasible_order(&preferred, &bounds, &bounds, 0.2).unwrap(),
            [0, 6, 3]
        );
    }

    #[test]
    fn infeasible_fixed_deadlines_report_infeasibility_not_search_exhaustion() {
        let preferred = [0, 3, 6];
        let bounds = [0.0, 1.0, 0.0, 0.01, 1.0, 0.0, 0.02, 1.0, 0.0];
        assert_eq!(
            feasible_order(&preferred, &bounds, &bounds, 0.2).unwrap_err(),
            "spacing projection: no ordering satisfies the frequency bounds"
        );
    }

    #[test]
    fn repairs_selected_dt1990pro_subthreshold_spacing() {
        let x = [
            2.2615865034879357,
            0.6054922755266307,
            1.4382566575761995,
            2.3217717980826973,
            1.2097280830807673,
            -3.2338506444654134,
        ];
        let lower = [
            1.6865971391405554,
            0.6,
            -18.0,
            2.072164282617129,
            0.6,
            -18.0,
        ];
        let upper = [2.4577314260937038, 6.0, 6.0, 2.8432985695702775, 6.0, 6.0];
        assert!(super::super::min_spacing::viol_spacing_from_xs(&x, PeqModel::Pk, 0.2) > 0.0);
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap();
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::Pk, 0.2),
            0.0
        );
        assert_eq!(&repaired[1..3], &x[1..3]);
        assert_eq!(&repaired[4..6], &x[4..6]);
        for (index, value) in repaired.iter().enumerate() {
            assert!(*value >= lower[index] && *value <= upper[index]);
        }
    }

    #[test]
    fn leaves_feasible_vector_bit_identical() {
        let x = [2.0, 1.0, 1.0, 3.0, 2.0, -1.0];
        let lower = [1.0, 1.0, -6.0, 1.0, 1.0, -6.0];
        let upper = [3.0, 6.0, 6.0, 4.0, 6.0, 6.0];
        assert_eq!(
            project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap(),
            x
        );
    }

    #[test]
    fn tight_feasible_bounds_move_only_violating_middle_filter() {
        let gap = 0.2 * std::f64::consts::LOG10_2;
        let first = 2.0;
        let last = first + 2.0 * gap + 1e-10;
        let x = [first, 1.0, 0.0, first + 0.5 * gap, 1.0, 2.0, last, 1.0, 0.0];
        let lower = [first, 1.0, 0.0, first, 1.0, -6.0, last, 1.0, 0.0];
        let upper = [first, 1.0, 0.0, last, 6.0, 6.0, last, 1.0, 0.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::HpPkLp, 0.2).unwrap();
        assert_eq!(repaired[0], first);
        assert_eq!(repaired[6], last);
        assert!(repaired[3] > x[3]);
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::HpPkLp, 0.2),
            0.0
        );
    }

    #[test]
    fn unsorted_vector_keeps_filter_attributes_in_original_slots() {
        let x = [2.1, 1.0, 2.0, 2.0, 2.0, -3.0, 2.02, 3.0, 4.0];
        let lower = [1.0, 1.0, -6.0, 1.0, 1.0, -6.0, 1.0, 1.0, -6.0];
        let upper = [3.0, 6.0, 6.0, 3.0, 6.0, 6.0, 3.0, 6.0, 6.0];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap();
        for i in 0..3 {
            assert_eq!(repaired[i * 3 + 1], x[i * 3 + 1]);
            assert_eq!(repaired[i * 3 + 2], x[i * 3 + 2]);
        }
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::Pk, 0.2),
            0.0
        );
    }

    #[test]
    fn pinned_tight_band_falls_back_to_greedy_repair() {
        // Captured 2.0_ascilab3/iir winner: seven Pk filters over a 1.36
        // octave active band with tiled boxes. The upper five centers sit
        // just under the 0.2 octave gap; PAVA pins against the box edges
        // and the ULP nudger gives up ("rounded result is infeasible"),
        // yet earliest placement fits.
        let x = [
            1.304294888438453,
            6.0,
            -10.93621444538024,
            1.4027466616706856,
            6.0,
            -9.270147769349748,
            1.4755909374237028,
            0.5000037930734992,
            4.499996845137993,
            1.534826970249966,
            5.999999921326473,
            -22.49999782293844,
            1.5940611777102593,
            2.5200244553975817,
            -17.16296105560578,
            1.653295385414744,
            0.5000030254136156,
            4.499997626723097,
            1.7125295931516948,
            3.375784040147605,
            4.5,
        ];
        let lower = [
            1.3040940100155531,
            0.5,
            -22.5,
            1.3040940100155531,
            0.5,
            -22.5,
            1.3624419504977845,
            0.5,
            -22.5,
            1.4207898909800158,
            0.5,
            -22.5,
            1.4791378314622474,
            0.5,
            -22.5,
            1.5374857719444788,
            0.5,
            -22.5,
            1.5958337124267101,
            0.5,
            -22.5,
        ];
        let upper = [
            1.3624419504977845,
            6.0,
            4.5,
            1.4207898909800158,
            6.0,
            4.5,
            1.4791378314622472,
            6.0,
            4.5,
            1.5374857719444786,
            6.0,
            4.5,
            1.5958337124267101,
            6.0,
            4.5,
            1.6541816529089415,
            6.0,
            4.5,
            1.7125295933911728,
            6.0,
            4.5,
        ];
        let repaired = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2)
            .expect("tight but feasible band must repair");
        assert_eq!(
            super::super::min_spacing::viol_spacing_from_xs(&repaired, PeqModel::Pk, 0.2),
            0.0
        );
        for i in 0..7 {
            assert!(
                repaired[i * 3] >= lower[i * 3] && repaired[i * 3] <= upper[i * 3],
                "center {i} out of bounds"
            );
            assert_eq!(repaired[i * 3 + 1], x[i * 3 + 1]);
            assert_eq!(repaired[i * 3 + 2], x[i * 3 + 2]);
        }
    }

    #[test]
    fn greedy_repair_refuses_truly_infeasible_boxes() {
        // Two filters whose boxes cannot host the 0.2 octave gap.
        let x = [1.0, 1.0, 0.0, 1.01, 1.0, 0.0];
        let lower = [1.0, 1.0, -6.0, 1.0, 1.0, -6.0];
        let upper = [1.02, 6.0, 6.0, 1.02, 6.0, 6.0];
        let error = project_min_spacing(&x, &lower, &upper, PeqModel::Pk, 0.2).unwrap_err();
        assert!(
            is_definite_spacing_infeasibility(&error),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn definite_infeasibility_ignores_undetermined_and_malformed() {
        assert!(is_definite_spacing_infeasibility(
            "spacing projection: rounded result is infeasible"
        ));
        assert!(is_definite_spacing_infeasibility(
            "spacing projection: no ordering satisfies the frequency bounds"
        ));
        assert!(!is_definite_spacing_infeasibility(
            "spacing projection: alternate-order search time limit reached; feasibility undetermined"
        ));
        assert!(!is_definite_spacing_infeasibility(
            "spacing projection: frequency outside finite bounds"
        ));
        assert!(!is_definite_spacing_infeasibility(
            "spacing repair refused: within_bounds=true, spacing=0, ceiling=1, min_gain=0"
        ));
    }
}
