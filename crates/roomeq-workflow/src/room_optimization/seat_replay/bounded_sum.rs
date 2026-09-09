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
            if *frequency <= endpoint {
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
