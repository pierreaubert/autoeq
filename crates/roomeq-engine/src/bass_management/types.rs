use crate::Curve;
use crate::error::{AutoeqError, Result};
use roomeq_model::{CrossoverConfig, RoomConfig, SubwooferCrossoverRef};

/// Information about an individual subwoofer driver from multi-sub preprocessing
#[derive(Clone)]
pub struct SubDriverInfo {
    /// Driver name (e.g., "subs_1", "Front Sub")
    pub name: String,
    /// Gain in dB from MSO/DBA optimization
    pub gain: f64,
    /// Delay in ms from MSO/DBA optimization
    pub delay: f64,
    /// Whether this driver is polarity-inverted
    pub inverted: bool,
    /// Initial measurement curve for this driver
    pub processing: Option<SubDriverProcessing>,
    pub initial_curve: Option<Curve>,
}

#[derive(Clone)]
pub struct SubDriverProcessing {
    pub plugins: Vec<roomeq_model::PluginConfigWrapper>,
    /// Measured response after per-driver filtering, before gain/delay/polarity.
    pub curve: Curve,
}

/// Result of subwoofer preprocessing
pub struct SubPreprocessResult {
    /// Per-seat combined responses for shared EQ, distinct from routing's
    /// representative complex response. Controls are already applied once.
    pub shared_eq_seats: Option<Vec<Curve>>,
    /// Dedicated spatial/global sub EQ already ran before routed integration.
    pub common_eq_complete: bool,
    pub advisories: Vec<String>,
    pub optimizer_evidence: Vec<autoeq_optim::optim::OptimizerRunEvidence>,
    /// Combined curve (for crossover optimization and shared post-EQ)
    pub combined_curve: Curve,
    /// Per-driver info (None for single sub)
    pub drivers: Option<Vec<SubDriverInfo>>,
}

#[derive(Debug, Clone)]
pub(super) struct GroupCrossoverPlan {
    pub(super) crossover_type: String,
    pub(super) frequency_hz: f64,
    pub(super) configured_hz: f64,
    pub(super) frequency_range: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub(super) struct BassManagementJointGroupInput {
    pub(super) group_id: String,
    pub(super) roles: Vec<String>,
    pub(super) plan: GroupCrossoverPlan,
    pub(super) virtual_main: Curve,
    pub(super) phase_available: bool,
    pub(super) advisories: Vec<String>,
}

/// Crossover plan for one physical subwoofer under a per-sub crossover list.
///
/// Entry `i` of [`SubwooferCrossoverRef::PerSub`] applies to sub `i` in
/// driver order. Each entry resolves to its own crossover key, hence its own
/// selectable low-pass range.
#[derive(Debug, Clone)]
pub(super) struct PerSubCrossoverPlan {
    pub(super) sub_index: usize,
    pub(super) crossover_key: String,
    pub(super) crossover_type: String,
    pub(super) configured_hz: f64,
    pub(super) frequency_range: Option<(f64, f64)>,
}

impl PerSubCrossoverPlan {
    /// Selectable low-pass bounds: the configured range, or the fixed
    /// frequency as a point bound. Always ordered `(min, max)`.
    pub(super) fn selectable_bounds_hz(&self) -> (f64, f64) {
        let (minimum, maximum) = self
            .frequency_range
            .unwrap_or((self.configured_hz, self.configured_hz));
        (minimum.min(maximum), minimum.max(maximum))
    }
}

/// Positional per-sub crossover keys when `system.subwoofers.crossover` is a
/// list with at least two entries; empty for the legacy single-string form
/// (including a one-element list), a missing section, or any other shape.
/// An empty return always means "keep the single-frequency group behavior
/// bit-identical".
pub(super) fn per_sub_crossover_keys(config: &RoomConfig) -> Vec<String> {
    match config
        .system
        .as_ref()
        .and_then(|system| system.subwoofers.as_ref())
        .and_then(|subwoofers| subwoofers.crossover.as_ref())
    {
        Some(SubwooferCrossoverRef::PerSub(keys)) if keys.len() >= 2 => keys.clone(),
        _ => Vec::new(),
    }
}

/// Resolve the per-sub crossover plans when `system.subwoofers.crossover`
/// is a positional list with at least two entries.
///
/// Returns `None` for the legacy single-string form (including a one-element
/// list, which behaves exactly like the shared form), for a missing
/// subwoofer/crossover section, and for unresolvable keys. `None` always
/// means "keep the single-frequency group behavior bit-identical".
/// Validation owns arity/existence errors; this is a best-effort resolver.
pub(super) fn per_sub_crossover_plans(config: &RoomConfig) -> Option<Vec<PerSubCrossoverPlan>> {
    let keys = per_sub_crossover_keys(config);
    if keys.is_empty() {
        return None;
    }
    let crossovers = config.crossovers.as_ref()?;
    keys.iter()
        .enumerate()
        .map(|(sub_index, key)| {
            let selected = crossovers.get(key)?;
            let configured_hz = selected.frequency.or_else(|| {
                selected
                    .frequency_range
                    .map(|(minimum, maximum)| (minimum.max(1.0) * maximum.max(1.0)).sqrt())
            })?;
            if !configured_hz.is_finite() || configured_hz <= 0.0 {
                return None;
            }
            Some(PerSubCrossoverPlan {
                sub_index,
                crossover_key: key.clone(),
                crossover_type: selected.crossover_type.clone(),
                configured_hz,
                frequency_range: selected.frequency_range,
            })
        })
        .collect()
}

pub(super) fn group_crossover_plan(
    config: &RoomConfig,
    fallback: &CrossoverConfig,
    group_id: &str,
) -> Result<GroupCrossoverPlan> {
    let selected = config
        .system
        .as_ref()
        .and_then(|system| system.bass_management.as_ref())
        .and_then(|bm| bm.group_crossovers.get(group_id))
        .and_then(|key| {
            config
                .crossovers
                .as_ref()
                .and_then(|crossovers| crossovers.get(key))
        })
        .unwrap_or(fallback);

    let (min_hz, max_hz, configured_hz) = if let Some(freq) = selected.frequency {
        (freq, freq, freq)
    } else if let Some((min, max)) = selected.frequency_range {
        (min, max, (min.max(1.0) * max.max(1.0)).sqrt())
    } else {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "Bass-management crossover for group '{group_id}' requires 'frequency' or 'frequency_range'"
            ),
        });
    };

    Ok(GroupCrossoverPlan {
        crossover_type: selected.crossover_type.clone(),
        frequency_hz: configured_hz,
        configured_hz,
        frequency_range: (min_hz != max_hz).then_some((min_hz, max_hz)),
    })
}
