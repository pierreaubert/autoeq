use super::*;
use roomeq_model::{SummationSupportEvidence, UpperBandAcousticBound};

/// Maximum error allowed when replacing an unknown small branch with zero.
/// The nominal phase is the retained sum's phase, never extrapolated sub phase.
const MAX_OMISSION_ERROR_DB: f64 = 0.1;

pub(super) struct Branch {
    pub output: String,
    pub measured: Curve,
    pub upper: Option<(Curve, UpperBandAcousticBound)>,
}

pub(super) struct Summed {
    pub curve: Curve,
    pub support: Vec<SummationSupportEvidence>,
}

impl Summed {
    pub fn uncertainty_db(&self) -> f64 {
        self.support
            .iter()
            .map(|s| s.max_magnitude_uncertainty_db)
            .fold(0.0, f64::max)
    }
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
    let upper = if let Some(bound) = matches.first() {
        let [low, high] = bound.band_hz;
        let endpoint = *raw
            .freq
            .last()
            .ok_or_else(|| invalid("empty bounded capture"))?;
        if !low.is_finite()
            || !high.is_finite()
            || low <= 0.0
            || high <= low
            || low > endpoint
            || high <= endpoint
            || !bound.max_spl_db.is_finite()
            || bound.evidence_id.trim().is_empty()
        {
            return Err(invalid(format!(
                "invalid upper-band acoustic bound for '{output}'"
            )));
        }
        if raw
            .freq
            .iter()
            .zip(&raw.spl)
            .any(|(f, spl)| *f >= low && *spl > bound.max_spl_db + 1e-9)
        {
            return Err(invalid(format!(
                "upper-band acoustic bound contradicts measured '{output}' levels"
            )));
        }
        let bound_curve = Curve {
            freq: grid.clone(),
            spl: ndarray::Array1::from_elem(grid.len(), bound.max_spl_db),
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
    let low = branches
        .iter()
        .map(|b| b.measured.freq[0])
        .fold(f64::INFINITY, f64::min)
        .max(low_limit);
    // A missing low-frequency branch cannot redefine the assessment band.
    // Upper-band omission bounds do not establish a lower-band acoustic bound.
    for branch in branches {
        // Decimal CSV endpoints and logspace endpoints can differ by an ULP
        // (e.g. 20 versus 20.000000000000004 Hz). This is numeric tolerance,
        // not an acoustic extrapolation allowance.
        let endpoint_tolerance = 8.0 * f64::EPSILON * branch.measured.freq[0].max(low);
        if branch.measured.freq[0] - low > endpoint_tolerance {
            return Err(invalid(format!(
                "insufficient summation evidence: '{}' is unmeasured below {} Hz; requested sum starts at {low} Hz",
                branch.output, branch.measured.freq[0],
            )));
        }
    }
    let high = branches
        .iter()
        .map(|b| *b.measured.freq.last().unwrap())
        .fold(0.0, f64::max)
        .min(high_limit);
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
        .map(|b| autoeq_measurements::read::interpolate_log_space(&grid, &b.measured))
        .collect();
    let upper: Vec<_> = branches
        .iter()
        .map(|b| {
            b.upper
                .as_ref()
                .map(|(c, _)| autoeq_measurements::read::interpolate_log_space(&grid, c))
        })
        .collect();
    let mut evidence: Vec<Option<SummationSupportEvidence>> = vec![None; branches.len()];
    let mut spl = Vec::with_capacity(grid.len());
    let mut phase = Vec::with_capacity(grid.len());
    for (i, frequency) in grid.iter().enumerate() {
        let mut retained = num_complex::Complex64::new(0.0, 0.0);
        let mut omitted = 0.0;
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
            let ratio = omitted / retained.norm();
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
    fn genuine_upper_gap_still_needs_declaration() {
        let branches = [branch("left_sub", 150.0), branch("left_main", 20_000.0)];
        let error = match sum_branches(&branches, 20.0, 200.0) {
            Ok(_) => panic!("50 Hz coverage gap must not sum silently"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("insufficient summation evidence"),
            "{error}"
        );
    }
}
