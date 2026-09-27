use super::*;
use roomeq_model::{SummationSupportEvidence, UpperBandAcousticBound};

/// Maximum error allowed when replacing an unknown small branch with zero.
/// The nominal phase is the retained sum's phase, never extrapolated sub phase.
const MAX_OMISSION_ERROR_DB: f64 = 0.1;

/// A declared negligible tail has no retained complex reference when a
/// standalone LFE/subwoofer input is replayed above its capture endpoint.
/// Keep that case bounded instead of producing an infinite relative ratio.
const MAX_ABSOLUTE_OMITTED_AMPLITUDE: f64 = 1.0e-5;

pub(super) struct Branch {
    pub output: String,
    pub measured: Curve,
    pub upper: Option<(Curve, UpperBandAcousticBound)>,
}

#[derive(Debug)]
pub(super) struct Summed {
    pub curve: Curve,
    pub support: Vec<SummationSupportEvidence>,
}

/// Summed response and uncertainty evidence for one realized seat.
impl Summed {
    pub fn uncertainty_db(&self) -> f64 {
        self.support
            .iter()
            .map(|s| s.max_magnitude_uncertainty_db)
            .fold(0.0, f64::max)
    }
}

/// Least-squares rolloff in dB per octave over the last measured
/// half-octave, plus the fitted level at the endpoint. Returns `None` when
/// fewer than two points cover the window, the fit is degenerate, or the
/// tail is rising: a rising edge cannot bound unmeasured output, so the
/// caller keeps the flat peak-hold assumption. The junction level never
/// understates the last measured bin.
fn measured_rolloff_db_per_oct(
    freq: &ndarray::Array1<f64>,
    spl: &ndarray::Array1<f64>,
    endpoint: f64,
) -> Option<(f64, f64)> {
    let points: Vec<(f64, f64)> = freq
        .iter()
        .zip(spl.iter())
        .filter(|(frequency, level)| {
            **frequency >= endpoint / std::f64::consts::SQRT_2
                && frequency.is_finite()
                && level.is_finite()
        })
        .map(|(frequency, level)| (frequency.log2(), *level))
        .collect();
    if points.len() < 2 {
        return None;
    }
    let count = points.len() as f64;
    let (sum_x, sum_y) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x, sy + y));
    let (sum_xx, sum_xy) = points
        .iter()
        .fold((0.0, 0.0), |(sxx, sxy), (x, y)| {
            (sxx + x * x, sxy + x * y)
        });
    let denominator = count * sum_xx - sum_x * sum_x;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return None;
    }
    let slope = (count * sum_xy - sum_x * sum_y) / denominator;
    if !slope.is_finite() || slope > 0.0 {
        return None;
    }
    let intercept = (sum_y - slope * sum_x) / count;
    let fitted = intercept + slope * endpoint.log2();
    let last = *spl.last().unwrap();
    if !fitted.is_finite() || !last.is_finite() {
        return None;
    }
    Some((slope, fitted.max(last)))
}

/// Apply identical electrical processing to measured transfer and an explicit
/// acoustic magnitude bound. The bound supplies no measured or invented phase.
pub(super) fn process_branch(
    output: &str,
    raw: &Curve,
    declarations: &HashMap<String, Vec<UpperBandAcousticBound>>,
    partition: &str,
    seat: usize,
    grid: &ndarray::Array1<f64>,
    subwoofer_low_pass_hz: Option<f64>,
    mut process: impl FnMut(&Curve) -> Result<Curve>,
) -> Result<Branch> {
    raw.validate("bounded replay capture")?;
    let mut native_grid = grid.to_vec();
    native_grid.extend(raw.freq.iter().copied());
    native_grid.sort_by(f64::total_cmp);
    native_grid.dedup();
    let grid = &ndarray::Array1::from(native_grid);
    let matches: Vec<_> = declarations
        .get(output)
        .into_iter()
        .flatten()
        .filter(|b| b.partition == partition && b.seat_index == seat)
        .collect();
    if matches.len() > 1 {
        return Err(invalid(format!(
            "duplicate upper-band bounds for '{output}' {partition} seat {seat}"
        )));
    }
    let endpoint = *raw
        .freq
        .last()
        .ok_or_else(|| invalid("empty bounded capture"))?;
    // For subwoofers measured through the crossover region, bound the
    // unmeasured stopband from the last measured half-octave. A falling
    // measured tail continues to fall at its fitted rolloff; anything else
    // (rising tail, fewer than two points) keeps the previous flat
    // peak-hold assumption. This records measured magnitude behavior, never
    // extrapolated phase. Requiring an octave beyond the low-pass preserves
    // crossover evidence; the actual DSP and omission budget must still make
    // the tail negligible.
    let inferred = subwoofer_low_pass_hz
        .filter(|cutoff| cutoff.is_finite() && *cutoff > 0.0 && endpoint >= 2.0 * cutoff)
        .filter(|_| grid[grid.len() - 1] > endpoint)
        .map(|_| {
            let rolloff = measured_rolloff_db_per_oct(&raw.freq, &raw.spl, endpoint);
            let (max_spl_db, rolloff_db_per_oct, evidence_id) = match rolloff {
                Some((slope, endpoint_level)) => (
                    endpoint_level,
                    Some(slope),
                    "measured_subwoofer_stopband_rolloff",
                ),
                None => (
                    raw.freq
                        .iter()
                        .zip(&raw.spl)
                        .filter(|(frequency, _)| {
                            **frequency >= endpoint / std::f64::consts::SQRT_2
                        })
                        .map(|(_, spl)| *spl)
                        .fold(f64::NEG_INFINITY, f64::max),
                    None,
                    "assumed_subwoofer_stopband_below_measured_tail",
                ),
            };
            UpperBandAcousticBound {
                partition: partition.into(),
                seat_index: seat,
                band_hz: [endpoint, grid[grid.len() - 1]],
                max_spl_db,
                rolloff_db_per_oct,
                evidence_id: evidence_id.into(),
            }
        });
    // An explicit declaration always takes precedence over the assumption.
    let bound = matches.first().copied().or(inferred.as_ref());
    let upper = if let Some(bound) = bound {
        let [low, high] = bound.band_hz;
        if !low.is_finite()
            || !high.is_finite()
            || low <= 0.0
            || high <= low
            || low > endpoint
            || high <= endpoint
            || !bound.max_spl_db.is_finite()
            || bound
                .rolloff_db_per_oct
                .is_some_and(|slope| !slope.is_finite() || slope > 0.0)
            || bound.evidence_id.trim().is_empty()
        {
            return Err(invalid(format!(
                "invalid upper-band acoustic bound for '{output}'"
            )));
        }
        if raw.freq.iter().zip(&raw.spl).any(|(f, spl)| {
            *f >= low && *spl > bound.level_at_hz(*f) + 1e-9
        }) {
            return Err(invalid(format!(
                "upper-band acoustic bound contradicts measured '{output}' levels"
            )));
        }
        let bound_curve = Curve {
            freq: grid.clone(),
            spl: grid.mapv(|frequency| bound.level_at_hz(frequency)),
            phase: None,
            ..Default::default()
        };
        Some((process(&bound_curve)?, (*bound).clone()))
    } else {
        None
    };
    // Interpolate acoustic captures first, then evaluate known electrical DSP
    // on the receiving grid. Interpolating a sparse already-delayed phase can
    // alias multiple turns and misrepresent even a known pure delay.
    let measured_grid: ndarray::Array1<f64> = grid
        .iter()
        .copied()
        .filter(|f| *f >= raw.freq[0] && *f <= *raw.freq.last().unwrap())
        .collect();
    let aligned = autoeq_measurements::read::interpolate_log_space(&measured_grid, raw);
    Ok(Branch {
        output: output.into(),
        measured: process(&aligned)?,
        upper,
    })
}

pub(super) fn sum_branches(branches: &[Branch], low_limit: f64, high_limit: f64) -> Result<Summed> {
    if branches.is_empty() {
        return Err(invalid("no final-seat branches"));
    }
    if branches.len() == 1 {
        return Ok(Summed {
            curve: branches[0].measured.clone(),
            support: vec![],
        });
    }
    for branch in branches {
        branch.measured.validate("final-seat branch")?;
        if !roomeq_engine::topology::curve_has_usable_phase(&branch.measured) {
            return Err(invalid(
                "final-seat coherent replay needs phase for every physical branch",
            ));
        }
    }
    let measured_low = branches
        .iter()
        .map(|b| b.measured.freq[0])
        .fold(f64::INFINITY, f64::min);
    // The test-only convenience wrapper uses zero as "native support";
    // production replay always supplies a positive configured lower bound.
    let low = (low_limit.is_finite() && low_limit > 0.0)
        .then_some(low_limit)
        .unwrap_or(measured_low);
    // A missing low-frequency branch cannot redefine the assessment band.
    // Upper-band omission bounds do not establish a lower-band acoustic bound.
    for branch in branches {
        // A capture may start one native measurement bin above the requested
        // lower edge (for example 20.1416 Hz for a nominal 20 Hz sweep).
        // This is a grid-edge condition, not evidence that the branch can be
        // extrapolated below its support. Hold the measured edge below it and
        // retain the requested observation band in the scorecard. Limit the
        // allowance both to one native bin and 1% of the edge so a genuinely
        // truncated capture still fails closed.
        let edge = branch.measured.freq[0];
        let native_bin = branch
            .measured
            .freq
            .get(1)
            .map(|next| (next - edge).abs())
            .unwrap_or(0.0);
        let endpoint_tolerance = (1.05 * native_bin).min(0.01 * edge.max(low));
        if edge - low > endpoint_tolerance {
            return Err(invalid(format!(
                "insufficient summation evidence: '{}' is unmeasured below {} Hz; requested sum starts at {low} Hz",
                branch.output, branch.measured.freq[0],
            )));
        }
    }
    // Do not shrink the requested observation band to the common measured
    // endpoint. A short branch is valid only when an explicit upper-band
    // acoustic bound was declared; otherwise the loop below reports honest
    // insufficient evidence.
    let measured_high = branches
        .iter()
        .map(|b| *b.measured.freq.last().unwrap())
        .fold(0.0, f64::max);
    // Likewise, an infinite upper limit means native support for the small
    // curve-summing helper. Real replay passes a finite observation limit.
    let high = (high_limit.is_finite() && high_limit > 0.0)
        .then_some(high_limit)
        .unwrap_or(measured_high);
    let mut grid: Vec<_> = branches
        .iter()
        .flat_map(|b| b.measured.freq.iter().copied())
        .filter(|f| *f >= low && *f <= high)
        .collect();
    grid.push(low);
    grid.push(high);
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    if grid.len() < 3 || high <= low {
        return Err(invalid("insufficient support for final-seat branch sum"));
    }
    let grid = ndarray::Array1::from(grid);
    let aligned: Vec<_> = branches
        .iter()
        .map(|b| autoeq_core::interpolate_log_space_hold_edges(&grid, &b.measured))
        .collect();
    let upper: Vec<_> = branches
        .iter()
        .map(|b| {
            b.upper
                .as_ref()
                .map(|(c, _)| autoeq_core::interpolate_log_space_hold_edges(&grid, c))
        })
        .collect();
    let mut evidence: Vec<Option<SummationSupportEvidence>> = vec![None; branches.len()];
    let mut spl = Vec::with_capacity(grid.len());
    let mut phase = Vec::with_capacity(grid.len());
    for (i, frequency) in grid.iter().enumerate() {
        let mut retained = num_complex::Complex64::new(0.0, 0.0);
        let mut omitted = 0.0;
        let mut retained_indices = Vec::new();
        let mut omitted_indices = Vec::new();
        for (index, branch) in branches.iter().enumerate() {
            let endpoint = *branch.measured.freq.last().unwrap();
            // Grid-quantization allowance: a round assessment edge (200 Hz)
            // routinely lands a fraction of a measurement bin past a dense
            // capture's last bin (199.951 Hz). Genuine coverage gaps are
            // octaves, never fractions of a bin, so anything beyond this
            // allowance still requires an explicit upper-band declaration.
            let covered =
                *frequency <= endpoint || *frequency - endpoint <= 1e-3 * endpoint.max(1.0);
            if covered {
                retained += num_complex::Complex64::from_polar(
                    10.0_f64.powf(aligned[index].spl[i] / 20.0),
                    aligned[index].phase.as_ref().unwrap()[i].to_radians(),
                );
                retained_indices.push(index);
            } else {
                let Some((_, declaration)) = &branch.upper else {
                    return Err(invalid(format!(
                        "insufficient summation evidence: '{}' unmeasured above {endpoint} Hz; requested assessment extends to {high} Hz",
                        branch.output
                    )));
                };
                if *frequency > declaration.band_hz[1] {
                    return Err(invalid(format!(
                        "insufficient summation evidence: '{}' bound ends at {} Hz, requested {high} Hz",
                        branch.output, declaration.band_hz[1]
                    )));
                }
                omitted += 10.0_f64.powf(upper[index].as_ref().unwrap().spl[i] / 20.0);
                omitted_indices.push(index);
            }
        }
        if !omitted_indices.is_empty() {
            let retained_norm = retained.norm();
            let ratio = if retained_norm > f64::EPSILON {
                omitted / retained_norm
            } else if retained_indices.is_empty() && omitted <= MAX_ABSOLUTE_OMITTED_AMPLITUDE {
                // A standalone LFE/subwoofer input can have no retained
                // complex reference above its endpoint.  The explicit
                // stop-band bound makes its contribution negligible in
                // absolute amplitude, so zero is the conservative replay
                // value and no relative ratio can be formed.
                0.0
            } else {
                f64::INFINITY
            };
            let error_db = -20.0 * (1.0 - ratio).log10();
            if !ratio.is_finite()
                || ratio >= 1.0
                || !error_db.is_finite()
                || error_db > MAX_OMISSION_ERROR_DB
            {
                return Err(invalid(format!(
                    "insufficient summation evidence at {frequency} Hz: omitted amplitude ratio {ratio:.6} exceeds {MAX_OMISSION_ERROR_DB} dB uncertainty budget"
                )));
            }
            for index in omitted_indices {
                let branch = &branches[index];
                let entry = evidence[index].get_or_insert_with(|| SummationSupportEvidence {
                    physical_output: branch.output.clone(),
                    measured_band_hz: [
                        branch.measured.freq[0],
                        *branch.measured.freq.last().unwrap(),
                    ],
                    omitted_band_hz: [*branch.measured.freq.last().unwrap(), high],
                    acoustic_bound: branch.upper.as_ref().unwrap().1.clone(),
                    max_sum_omitted_amplitude_ratio: 0.0,
                    max_magnitude_uncertainty_db: 0.0,
                    max_phase_uncertainty_deg: 0.0,
                });
                entry.max_sum_omitted_amplitude_ratio =
                    entry.max_sum_omitted_amplitude_ratio.max(ratio);
                entry.max_magnitude_uncertainty_db =
                    entry.max_magnitude_uncertainty_db.max(error_db);
                entry.max_phase_uncertainty_deg = entry
                    .max_phase_uncertainty_deg
                    .max(ratio.asin().to_degrees());
            }
        }
        spl.push(20.0 * retained.norm().max(1e-12).log10());
        phase.push(retained.arg().to_degrees());
    }
    Ok(Summed {
        curve: Curve {
            freq: grid,
            spl: spl.into(),
            phase: Some(phase.into()),
            ..Default::default()
        },
        support: evidence.into_iter().flatten().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch(output: &str, last_hz: f64) -> Branch {
        let freq = ndarray::Array1::logspace(10.0, 10.0_f64.log10(), last_hz.log10(), 32);
        let len = freq.len();
        Branch {
            output: output.into(),
            measured: Curve {
                freq,
                spl: ndarray::Array1::from_elem(len, 70.0),
                phase: Some(ndarray::Array1::zeros(len)),
                ..Default::default()
            },
            upper: None,
        }
    }

    #[test]
    fn one_lower_edge_bin_is_held_without_extrapolation() {
        let make = |output: &str| Branch {
            output: output.into(),
            measured: Curve {
                freq: ndarray::Array1::from_vec(vec![20.14, 20.28, 40.0, 80.0]),
                spl: ndarray::Array1::from_vec(vec![70.0, 69.0, 68.0, 67.0]),
                phase: Some(ndarray::Array1::zeros(4)),
                ..Default::default()
            },
            upper: None,
        };
        let summed = sum_branches(&[make("a"), make("b")], 20.0, 80.0)
            .expect("one native lower-edge bin should be accepted");
        let first = summed.curve.spl[0];
        assert!(
            (first - 76.0206).abs() < 0.01,
            "held edge should sum the measured 70 dB values, got {first}"
        );
    }

    #[test]
    fn genuine_lower_gap_still_fails_closed() {
        let make = |output: &str| Branch {
            output: output.into(),
            measured: Curve {
                freq: ndarray::Array1::from_vec(vec![25.0, 30.0, 40.0, 80.0]),
                spl: ndarray::Array1::from_elem(4, 70.0),
                phase: Some(ndarray::Array1::zeros(4)),
                ..Default::default()
            },
            upper: None,
        };
        let error = sum_branches(&[make("a"), make("b")], 20.0, 80.0)
            .expect_err("a genuinely truncated lower band must not be scored");
        assert!(error.to_string().contains("unmeasured below"));
    }

    #[test]
    fn sub_bin_past_assessment_edge_is_covered() {
        // REW linear tails end at 199.951172 Hz against a round 200 Hz
        // assessment edge: a fraction of a measurement bin, not missing
        // evidence (measured 2.2_sigberg3 left_sub).
        let branches = [
            branch("left_sub", 199.951172),
            branch("left_main", 20_000.0),
        ];
        let summed = sum_branches(&branches, 20.0, 200.0)
            .expect("sub-bin endpoint gap must not fail summation");
        assert!(summed.support.is_empty());
        let top = *summed.curve.freq.last().unwrap();
        assert!((top - 200.0).abs() < 1e-9, "unexpected top {top}");
    }

    #[test]
    fn short_support_without_bound_is_not_scored() {
        let branches = [branch("sub_a", 199.951172), branch("sub_b", 200.0)];
        // Common measured support must not silently redefine the requested
        // observation band. Missing upper support is handled by
        // `sum_branches` through an explicit bound or an error.
        let error = sum_branches(&branches, 20.0, 16_000.0)
            .expect_err("short branches without a declared upper bound must not be scored");
        assert!(
            error
                .to_string()
                .contains("insufficient summation evidence")
        );
    }

    #[test]
    fn genuine_upper_gap_still_needs_declaration() {
        let branches = [branch("left_sub", 150.0), branch("left_main", 20_000.0)];
        let error = match sum_branches(&branches, 20.0, 200.0) {
            Ok(_) => panic!("50 Hz coverage gap must not sum silently"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("insufficient summation evidence"), "{error}");
    }

    #[test]
    fn low_pass_alone_cannot_establish_short_subwoofer_acoustic_support() {
        let grid = ndarray::Array1::from_vec(vec![20.0, 100.0, 199.951172, 1_000.0, 16_000.0]);
        let sub = Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 199.951172]),
            spl: ndarray::Array1::from_vec(vec![32.0, 32.0, 32.0]),
            phase: Some(ndarray::Array1::zeros(3)),
            ..Default::default()
        };
        let main = Curve {
            freq: grid.clone(),
            spl: ndarray::Array1::from_elem(grid.len(), 80.0),
            phase: Some(ndarray::Array1::zeros(grid.len())),
            ..Default::default()
        };
        let declarations = HashMap::new();
        let sub = process_branch(
            "subwoofer_1",
            &sub,
            &declarations,
            "training",
            0,
            &grid,
            None,
            |curve| {
                // Model the deployed low-pass attenuation on the inferred
                // bound.  The measured path itself stays unchanged.
                let mut processed = curve.clone();
                if processed.phase.is_none() {
                    processed.phase = Some(ndarray::Array1::zeros(processed.freq.len()));
                }
                processed.spl = processed
                    .freq
                    .mapv(|frequency| if frequency > 200.0 { -120.0 } else { 32.0 });
                Ok(processed)
            },
        )
        .expect("measured branch processing remains valid");
        let main = Branch {
            output: "main".into(),
            measured: main,
            upper: None,
        };
        assert!(
            sub.upper.is_none(),
            "low-pass metadata is not acoustic evidence"
        );
        let error = sum_branches(&[sub, main], 20.0, 16_000.0)
            .expect_err("an unmeasured acoustic tail must remain unassessed");
        assert!(
            error
                .to_string()
                .contains("insufficient summation evidence")
        );
    }

    #[test]
    fn subwoofer_tail_assumption_requires_crossover_coverage_and_negligible_output() {
        let grid = ndarray::Array1::from_vec(vec![20.0, 100.0, 250.0, 1000.0, 16000.0]);
        let raw = Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 250.0]),
            spl: ndarray::Array1::from_elem(3, 70.0),
            phase: Some(ndarray::Array1::zeros(3)),
            ..Default::default()
        };
        let main = || Branch {
            output: "main".into(),
            measured: Curve {
                freq: grid.clone(),
                spl: ndarray::Array1::from_elem(grid.len(), 80.0),
                phase: Some(ndarray::Array1::zeros(grid.len())),
                ..Default::default()
            },
            upper: None,
        };
        for (cutoff, attenuation, accepted) in [
            (Some(80.0), 60.0, true),
            (None, 60.0, false),
            (Some(150.0), 60.0, false),
            (Some(80.0), 0.0, false),
        ] {
            let branch = process_branch(
                "sub",
                &raw,
                &HashMap::new(),
                "training",
                0,
                &grid,
                cutoff,
                |curve| {
                    let mut processed = curve.clone();
                    for (frequency, spl) in processed.freq.iter().zip(&mut processed.spl) {
                        if *frequency > 250.0 {
                            *spl -= attenuation;
                        }
                    }
                    Ok(processed)
                },
            )
            .unwrap();
            let summed = sum_branches(&[main(), branch], 20.0, 16000.0);
            assert_eq!(
                summed.is_ok(),
                accepted,
                "cutoff={cutoff:?}, cut={attenuation}"
            );
            if let Ok(summed) = summed {
                assert_eq!(*summed.curve.freq.last().unwrap(), 16000.0);
                assert!(summed.uncertainty_db() < MAX_OMISSION_ERROR_DB);
                assert_eq!(summed.support.len(), 1);
            }
        }
    }

    #[test]
    fn negligible_standalone_subwoofer_tail_has_no_relative_denominator() {
        let grid = ndarray::Array1::from_vec(vec![20.0, 100.0, 200.0, 1_000.0, 16_000.0]);
        let branches = (0..2)
            .map(|index| Branch {
                output: format!("subwoofer_{}", index + 1),
                measured: Curve {
                    freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 200.0]),
                    spl: ndarray::Array1::from_elem(3, 32.0),
                    phase: Some(ndarray::Array1::zeros(3)),
                    ..Default::default()
                },
                upper: Some((
                    Curve {
                        freq: grid.clone(),
                        spl: ndarray::Array1::from_elem(grid.len(), -120.0),
                        phase: None,
                        ..Default::default()
                    },
                    UpperBandAcousticBound {
                        partition: "training".into(),
                        seat_index: 0,
                        band_hz: [200.0, 16_000.0],
                        max_spl_db: -120.0,
                        rolloff_db_per_oct: None,
                        evidence_id: format!("subwoofer-stop-band-{index}"),
                    },
                )),
            })
            .collect::<Vec<_>>();
        let summed = sum_branches(&branches, 20.0, 16_000.0)
            .expect("negligible omitted subwoofer tails should not form an infinite ratio");
        assert_eq!(summed.support.len(), 2);
        assert!(
            summed
                .support
                .iter()
                .all(|support| support.max_magnitude_uncertainty_db == 0.0)
        );
        assert!(summed.curve.spl.iter().skip(3).all(|level| *level < -200.0));
    }

    #[test]
    fn inferred_subwoofer_bound_follows_measured_rolloff() {
        // Mirror of 2.2_sigberg1 FIR: sub endpoint 199.951172 Hz with a
        // -24 dB/oct measured tail, +4.835 dB branch gain, 72.11 Hz LR24
        // low-pass. Flat peak-hold puts 41.6 dB of omitted bound against a
        // 76 dB seat sum at 201.554078 Hz (ratio 0.0168, over the 0.1 dB
        // budget); the fitted rolloff puts 11.7 dB there (ratio 0.0006).
        let grid = ndarray::Array1::from_vec(vec![
            20.0,
            100.0,
            199.951172,
            201.554078,
            1000.0,
            16_000.0,
        ]);
        let freq =
            ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 199.951172_f64.log10(), 64);
        let spl = freq.mapv(|frequency| {
            if frequency <= 72.11 {
                78.0
            } else {
                78.0 - 24.0 * (frequency / 72.11).log2()
            }
        });
        let raw = Curve {
            freq: freq.clone(),
            spl,
            phase: Some(ndarray::Array1::zeros(freq.len())),
            ..Default::default()
        };
        let branch = process_branch(
            "sub",
            &raw,
            &HashMap::new(),
            "training",
            0,
            &grid,
            Some(72.11),
            |curve| {
                let mut processed = curve.clone();
                for (frequency, level) in processed.freq.iter().zip(&mut processed.spl) {
                    *level += 4.835;
                    if *frequency > 72.11 {
                        *level -= 24.0 * (*frequency / 72.11).log2();
                    }
                }
                Ok(processed)
            },
        )
        .unwrap();
        let (curve, declaration) = branch
            .upper
            .as_ref()
            .expect("a falling tail must infer a bound");
        assert_eq!(
            declaration.evidence_id,
            "measured_subwoofer_stopband_rolloff"
        );
        let slope = declaration
            .rolloff_db_per_oct
            .expect("rolloff must be recorded");
        assert!(
            (slope + 24.0).abs() < 0.5,
            "unexpected fitted slope {slope}"
        );
        let top = *curve.spl.last().unwrap();
        assert!(
            top < declaration.max_spl_db - 100.0,
            "bound curve must decline, ends at {top}"
        );
        let main = Branch {
            output: "main".into(),
            measured: Curve {
                freq: grid.clone(),
                spl: ndarray::Array1::from_elem(grid.len(), 76.0),
                phase: Some(ndarray::Array1::zeros(grid.len())),
                ..Default::default()
            },
            upper: None,
        };
        let summed = sum_branches(&[main, branch], 20.0, 16_000.0)
            .expect("the fitted rolloff must satisfy the omission budget");
        assert_eq!(summed.support.len(), 1);
        assert!(summed.uncertainty_db() < MAX_OMISSION_ERROR_DB);
        assert!(summed.uncertainty_db() > 0.0);
    }

    #[test]
    fn rising_tail_keeps_flat_peak_hold() {
        let raw = Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 150.0, 199.951172]),
            spl: ndarray::Array1::from_vec(vec![50.0, 52.0, 55.0, 60.0]),
            phase: Some(ndarray::Array1::zeros(4)),
            ..Default::default()
        };
        let grid =
            ndarray::Array1::from_vec(vec![20.0, 100.0, 199.951172, 1000.0, 16_000.0]);
        let branch = process_branch(
            "sub",
            &raw,
            &HashMap::new(),
            "training",
            0,
            &grid,
            Some(72.11),
            |curve| Ok(curve.clone()),
        )
        .unwrap();
        let (_, declaration) = branch
            .upper
            .as_ref()
            .expect("peak hold must still infer a bound");
        assert_eq!(declaration.rolloff_db_per_oct, None);
        assert_eq!(
            declaration.evidence_id,
            "assumed_subwoofer_stopband_below_measured_tail"
        );
        assert_eq!(declaration.max_spl_db, 60.0);
    }

    #[test]
    fn explicit_rising_rolloff_is_rejected() {
        let raw = Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 250.0]),
            spl: ndarray::Array1::from_vec(vec![50.0, 50.0, 50.0]),
            phase: Some(ndarray::Array1::zeros(3)),
            ..Default::default()
        };
        let grid =
            ndarray::Array1::from_vec(vec![20.0, 100.0, 250.0, 1000.0, 16_000.0]);
        let declarations = HashMap::from([(
            "sub".to_owned(),
            vec![UpperBandAcousticBound {
                partition: "training".into(),
                seat_index: 0,
                band_hz: [200.0, 16_000.0],
                max_spl_db: 70.0,
                rolloff_db_per_oct: Some(3.0),
                evidence_id: "rising-tail".into(),
            }],
        )]);
        let error = process_branch(
            "sub",
            &raw,
            &declarations,
            "training",
            0,
            &grid,
            None,
            |curve| Ok(curve.clone()),
        )
        .err()
        .expect("a rising tail can never bound unmeasured output");
        assert!(
            error.to_string().contains("invalid upper-band acoustic bound"),
            "{error}"
        );
    }

    #[test]
    fn declining_bound_is_checked_at_frequency() {
        let raw = Curve {
            freq: ndarray::Array1::from_vec(vec![20.0, 100.0, 200.0, 250.0]),
            spl: ndarray::Array1::from_vec(vec![50.0, 50.0, 70.0, 68.0]),
            phase: Some(ndarray::Array1::zeros(4)),
            ..Default::default()
        };
        let grid =
            ndarray::Array1::from_vec(vec![20.0, 100.0, 250.0, 1000.0, 16_000.0]);
        // At 250 Hz the declining bound allows 70 - 12*log2(250/200) = 66.1
        // dB, so a 68 dB measured point contradicts it.
        let declarations = HashMap::from([(
            "sub".to_owned(),
            vec![UpperBandAcousticBound {
                partition: "training".into(),
                seat_index: 0,
                band_hz: [200.0, 16_000.0],
                max_spl_db: 70.0,
                rolloff_db_per_oct: Some(-12.0),
                evidence_id: "declining-tail".into(),
            }],
        )]);
        let error = process_branch(
            "sub",
            &raw,
            &declarations,
            "training",
            0,
            &grid,
            None,
            |curve| Ok(curve.clone()),
        )
        .err()
        .expect("measured levels above the declining bound must fail");
        assert!(
            error.to_string().contains("contradicts measured"),
            "{error}"
        );
    }
}
