use super::default::default_all_channel_multiseat_enabled;
use super::default::default_all_channel_multiseat_strategy;
use super::default::default_max_deviation_db;
use super::default::default_multiseat_global_eq;
use super::default::default_multiseat_per_sub_peq;
use super::default::default_primary_seat_weight;
use super::multi_seat_search_config::MultiSeatSearchConfig;
use super::types::ContinuousListeningAreaConfig;
use super::types::MultiMeasurementStrategy;
use super::types::MultiSeatStrategy;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Multi-seat optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MultiSeatConfig {
    /// Enable multi-seat optimization
    #[serde(default)]
    pub enabled: bool,
    /// Optimization strategy
    #[serde(default)]
    pub strategy: MultiSeatStrategy,
    /// Index of primary seat (0-based, used with PrimaryWithConstraints strategy)
    #[serde(default)]
    pub primary_seat: usize,
    /// Maximum allowed deviation at non-primary seats (dB)
    #[serde(default = "default_max_deviation_db")]
    pub max_deviation_db: f64,
    /// Enable per-sub polarity search for MSO.
    #[serde(default)]
    pub optimize_polarity: bool,
    /// Number of per-sub all-pass filters allowed during MSO.
    #[serde(default)]
    pub allpass_filters_per_sub: usize,
    /// Optimize a per-subwoofer PEQ from that sub's measurements across all seats
    /// before the gain/delay/polarity/all-pass MSO pass.
    #[serde(default = "default_multiseat_per_sub_peq")]
    pub per_sub_peq: bool,
    /// Optimize a shared EQ on the post-MSO combined response across all seats.
    #[serde(default = "default_multiseat_global_eq")]
    pub global_eq: bool,
    /// Enable all-channel multi-seat correction for non-sub home-cinema channels.
    #[serde(default = "default_all_channel_multiseat_enabled")]
    pub all_channel_enabled: bool,
    /// Strategy used when deriving per-channel multi-measurement correction.
    #[serde(default = "default_all_channel_multiseat_strategy")]
    pub all_channel_strategy: MultiMeasurementStrategy,
    /// Optional seat weights for all-channel multi-seat correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_weights: Option<Vec<f64>>,
    /// Relative primary-seat weight used with PrimaryWithConstraints.
    #[serde(default = "default_primary_seat_weight")]
    pub primary_seat_weight: f64,
    /// Continuous listening-area prior. Required (and only consulted) when
    /// `strategy = ContinuousArea`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_area: Option<ContinuousListeningAreaConfig>,
    /// Stable per-seat identifiers, parallel to
    /// `continuous_area.seat_positions` (and to the discrete measurement
    /// seat order for the other strategies). See [`SeatIdentityMap`].
    /// `None` keeps the legacy positional (index-order) correspondence under
    /// the documented strict-ordering contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seat_identity: Option<SeatIdentityMap>,
    /// Stage-specific search contract (seed, evaluation/time budgets, and
    /// resource caps). `None` preserves the historical engine behavior
    /// (fixed internal seed and budgets); see [`MultiSeatSearchConfig`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<MultiSeatSearchConfig>,
}

/// Stable seat identifiers closing the index/parallel-array correspondence gap.
///
/// `primary_seat` is a positional index while seat positions form a parallel
/// array: loaders only check equal counts, not semantic correspondence, so a
/// swapped order on one sub/source silently combines different physical
/// positions. One unique, non-empty ID per seat lets consumers
/// (`roomeq-workflow`, `roomeq-engine`) join every source's per-seat
/// measurements to IDs instead of relying on index order, and echo the IDs in
/// output reports so the correspondence is auditable.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SeatIdentityMap {
    /// One ID per seat, parallel to the seat order. Length must equal the
    /// seat count; every entry must be non-empty and unique.
    pub ids: Vec<String>,
}

impl SeatIdentityMap {
    /// Structural validation against a seat count. Returns one error string
    /// per violation (length mismatch, empty ID, duplicate ID); empty means
    /// the map is usable for `num_seats` seats.
    pub fn validate(&self, num_seats: usize) -> Vec<String> {
        let mut errors = Vec::new();
        if self.ids.len() != num_seats {
            errors.push(format!(
                "multi_seat.seat_identity.ids length {} must equal seat count {}",
                self.ids.len(),
                num_seats
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for (i, id) in self.ids.iter().enumerate() {
            if id.is_empty() {
                errors.push(format!(
                    "multi_seat.seat_identity.ids[{}] must be non-empty",
                    i
                ));
            } else if !seen.insert(id.as_str()) {
                errors.push(format!(
                    "multi_seat.seat_identity.ids[{}] duplicates seat ID '{}'",
                    i, id
                ));
            }
        }
        errors
    }

    /// Canonical IDs for `num_seats` seats: explicit IDs with positional
    /// `seat-{i}` fallback for missing/empty slots (validation reports those
    /// separately via [`SeatIdentityMap::validate`]).
    pub fn effective_ids(&self, num_seats: usize) -> Vec<String> {
        (0..num_seats)
            .map(|i| {
                self.ids
                    .get(i)
                    .filter(|id| !id.is_empty())
                    .cloned()
                    .unwrap_or_else(|| ContinuousListeningAreaConfig::fallback_seat_id(i))
            })
            .collect()
    }

    /// Explicit source x seat completeness/order check for measurement
    /// loaders. Each entry of `per_source_seat_keys` holds the seat keys of
    /// one sub/source in that source's own order. Empty return means every
    /// source covers exactly the canonical IDs in order; otherwise one error
    /// string per violation (wrong count, or a seat key at the wrong
    /// position, e.g. a reordered source list).
    pub fn check_source_seat_coverage(
        &self,
        num_seats: usize,
        per_source_seat_keys: &[Vec<String>],
    ) -> Vec<String> {
        ContinuousListeningAreaConfig::check_keys_against(
            &self.effective_ids(num_seats),
            per_source_seat_keys,
        )
    }
}

impl Default for MultiSeatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: MultiSeatStrategy::MinimizeVariance,
            primary_seat: 0,
            max_deviation_db: default_max_deviation_db(),
            optimize_polarity: false,
            allpass_filters_per_sub: 0,
            per_sub_peq: default_multiseat_per_sub_peq(),
            global_eq: default_multiseat_global_eq(),
            all_channel_enabled: default_all_channel_multiseat_enabled(),
            all_channel_strategy: default_all_channel_multiseat_strategy(),
            seat_weights: None,
            primary_seat_weight: default_primary_seat_weight(),
            continuous_area: None,
            seat_identity: None,
            search: None,
        }
    }
}

impl MultiSeatConfig {
    /// Effective search contract: the explicit `search` block, or the
    /// historical engine behavior encoded as defaults (no explicit seed or
    /// budgets, generous resource caps). See [`MultiSeatSearchConfig`].
    pub fn effective_search(&self) -> MultiSeatSearchConfig {
        self.search.clone().unwrap_or_default()
    }
}
