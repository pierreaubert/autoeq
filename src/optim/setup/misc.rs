use crate::param_utils::PeqLayout;

/// Clamp the per-filter gain **upper** bound to 0 dB for any filter
/// whose maximum allowed center frequency is at or below the Schroeder
/// frequency.
///
/// Below Schroeder the room is modal: peaks caused by constructive
/// interference between the direct wave and its reflections *can* be
/// cut by EQ (reducing the input at f directly reduces the SPL at f),
/// but nulls caused by destructive interference *cannot* be filled by
/// EQ boost — the cancellation happens after the EQ, so adding more
/// input energy just raises the direct wave and its anti-phase
/// reflection by the same ratio and the null stays. Worse, the boost
/// burns amplifier headroom and excites woofer excursion for no
/// audible benefit.
///
/// Letting the DE optimizer place boost filters anywhere in the modal
/// region therefore wastes filter slots on physically impossible
/// corrections. This function enforces "below Schroeder is cuts-only"
/// as a hard constraint on the optimizer's parameter bounds.
///
/// Filters whose allowed frequency range *straddles* Schroeder (i.e.
/// their upper frequency bound is above `schroeder_hz`) keep their
/// original symmetric bounds — those filters can still be positioned
/// in the above-Schroeder part of their range where boosts are
/// physically meaningful.
pub fn restrict_boost_above_schroeder(
    upper_bounds: &mut [f64],
    params: &crate::OptimParams,
    schroeder_hz: f64,
) {
    if schroeder_hz <= 0.0 {
        return;
    }
    let model = params.peq_model;
    let ppf = crate::param_utils::params_per_filter(model);
    let log_schroeder = schroeder_hz.log10();

    let layout = model.layout();
    for i in 0..params.num_filters {
        let offset = i * ppf;
        let freq_idx = offset + layout.freq_idx;
        let gain_idx = offset + layout.gain_idx;
        if freq_idx >= upper_bounds.len() || gain_idx >= upper_bounds.len() {
            continue;
        }
        // If the filter's highest possible center frequency is at or
        // below Schroeder, every placement of this filter lives inside
        // the modal region — constrain it to cuts only.
        if upper_bounds[freq_idx] <= log_schroeder && upper_bounds[gain_idx] > 0.0 {
            upper_bounds[gain_idx] = 0.0;
        }
    }
}

/// Build an initial guess vector for each filter.
pub fn initial_guess(
    params: &crate::OptimParams,
    lower_bounds: &[f64],
    upper_bounds: &[f64],
) -> Vec<f64> {
    let model = params.peq_model;
    let ppf = crate::param_utils::params_per_filter(model);
    let mut x = vec![];

    for i in 0..params.num_filters {
        let group = model.initial_guess_filter(
            i,
            &lower_bounds[i * ppf..(i + 1) * ppf],
            &upper_bounds[i * ppf..(i + 1) * ppf],
            params.min_db,
            params.max_freq,
        );
        x.extend_from_slice(&group);
    }
    x
}

pub(super) fn resolves_to_backend(name: &str, canonical: &str) -> bool {
    super::super::registry::resolve(name)
        .map(|backend| backend.name().eq_ignore_ascii_case(canonical))
        .unwrap_or(false)
}

/// Generate initial guess for multi-subwoofer optimization.
///
/// Returns a vector of zeros for all gain and delay parameters,
/// representing no gain adjustment and no delay for each driver.
///
/// # Arguments
///
/// * `n_drivers` - Number of subwoofers
///
/// # Returns
///
/// Vector of `n_drivers * 2` zeros (gains followed by delays).
pub fn multisub_initial_guess(n_drivers: usize) -> Vec<f64> {
    vec![0.0; n_drivers * 2]
}
