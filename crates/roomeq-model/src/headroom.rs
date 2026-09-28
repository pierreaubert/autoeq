//! Unified headroom doctrine: one budget object, asymmetric by role.
//!
//! # Contract (Phase 0)
//!
//! Mains define system SPL and must not spend it cheaply; subwoofers flex by
//! cutting (narrow-peak cuts are nearly free) and must never boost past
//! headroom. Since either side can be the limiting speaker, capability is
//! assessed per fixture and the strategy follows the assessment.
//!
//! Consumers and their bounds (all read from [`HeadroomBudget`], none invent
//! private limits):
//!
//! | Consumer | Objective | Bounds | Regularization |
//! |---|---|---|
//! | Route DE trim | splice sum vs target + underfill <= 1 dB | up: joint share of sub boost after MSO-applied gains (Phase 3a); down: `optimizer.min_db` | domination penalty prefers cutting the hot branch (Phase 3a) |
//! | MSO/sub outputs | array flatness per input | up: `max_sub_boost_db` share; down: `gain - max(max_db, 6)` | none (gains are the answer, not the cost) |
//! | FIR/PEQ authority | target match per band | mains: SPL-loss budget + narrow-cut preference; subs: deep narrow cuts, boosts headroom-capped (Phase 3d) | Q-dependent boost caps |
//! | Safety/finalization | peaks under ceiling | ceiling `output_ceiling_dbfs`, attenuation `max_attenuation_db`, useful loss per role | none (hard gates) |
//! | Bus routing | redirected sums under ceiling | redirect matrix scaled to bus budget (Phase 3b) | prefer fewer redirects over hotter bus |
//!
//! Staging: Phase 1 derives exactly today's constants (behavior-neutral).
//! Phase 2 adds an advisory-only capability/feasibility pre-pass. Phase 3
//! wires consumers to measurement-derived budgets one at a time. Phase 4
//! adds best-feasible-mode fallback. Each phase is gated on zero verdict
//! flips for currently-accepting measured fixtures.
//!
//! Regression fixtures (measured): unknown iir/fir (boost/safety stalemate),
//! kef fir/mixed (redirected-bus overload), genelec (modal XO), 5.1.4
//! fir/mixed (structural bus overload). See per-fixture ORIGIN.md triage notes.

// Rust guideline compliant 2026-02-21

use crate::RoomConfig;

/// Which branch is allowed to flex to meet the splice at target level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeadroomStrategy {
    /// Fixed legacy behavior: no assessed strategy (Phase 1).
    #[default]
    Fixed,
    /// Both branches within budget; match by cutting the hotter one.
    Balanced,
    /// Sub has headroom: cut sub peaks/level freely to meet the mains.
    SubFlexes,
    /// Sub cannot keep up: no sub boost, protect mains, report shortfall.
    MainProtects,
}

/// Per-role correction authority caps, in dB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoleHeadroomAuthority {
    /// Maximum upward correction (boost) anywhere in the role band.
    pub max_boost_db: f64,
    /// Maximum cut depth for narrow (high-Q / modal) features.
    pub max_narrow_cut_db: f64,
    /// Maximum cut depth for wide (low-Q / broadband) features.
    pub max_wide_cut_db: f64,
    /// Maximum broadband level loss vs the uncorrected role.
    pub max_spl_loss_db: f64,
}

impl RoleHeadroomAuthority {
    /// Phase 1 legacy authority: uniform caps, no asymmetry. Asymmetric
    /// values land per consumer in Phase 3d.
    pub fn legacy(max_boost_db: f64) -> Self {
        Self {
            max_boost_db,
            max_narrow_cut_db: f64::INFINITY,
            max_wide_cut_db: f64::INFINITY,
            max_spl_loss_db: f64::INFINITY,
        }
    }
}

/// Whole-graph headroom budget. Every optimizer, gate, and safety stage reads
/// its limits from one instance instead of re-deriving private constants.
/// Copy so optimizer closures capture limits without borrow plumbing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadroomBudget {
    pub main: RoleHeadroomAuthority,
    pub sub: RoleHeadroomAuthority,
    /// Route-trim upward bound before joint MSO accounting (Phase 1 value).
    pub max_route_trim_up_db: f64,
    /// Route-trim downward bound (cutting the bass to meet the mains).
    pub max_route_trim_down_db: f64,
    /// MSO/sub-output upward bound (same pool as route trim in Phase 3a).
    pub max_output_boost_db: f64,
    pub output_ceiling_dbfs: f64,
    pub max_attenuation_db: f64,
    pub max_useful_output_loss_db: f64,
    pub strategy: HeadroomStrategy,
}

impl HeadroomBudget {
    /// Derive exactly today's constants from config.
    ///
    /// Phase 1 behavior is neutral by construction: every field equals the
    /// expression its consumer previously inlined.
    ///
    /// # Examples
    ///
    /// ```
    /// use roomeq_model::{RoomConfig, headroom::HeadroomBudget};
    ///
    /// let budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
    /// assert_eq!(budget.output_ceiling_dbfs, 0.0);
    /// assert_eq!(budget.max_attenuation_db, 12.0);
    /// ```
    pub fn from_legacy_config(config: &RoomConfig) -> Self {
        let sub_boost = config
            .system
            .as_ref()
            .and_then(|system| system.bass_management.as_ref())
            .map(|bm| bm.max_sub_boost_db.max(0.0))
            .unwrap_or(config.optimizer.max_db.max(0.0));
        let authority = RoleHeadroomAuthority::legacy(config.optimizer.max_db.max(0.0));
        Self {
            main: authority,
            sub: authority,
            max_route_trim_up_db: sub_boost,
            max_route_trim_down_db: config.optimizer.min_db,
            max_output_boost_db: sub_boost,
            output_ceiling_dbfs: config.optimizer.finalization.output_ceiling_dbfs,
            max_attenuation_db: config.optimizer.finalization.max_attenuation_db,
            max_useful_output_loss_db: config.optimizer.finalization.max_useful_output_loss_db,
            strategy: HeadroomStrategy::Fixed,
        }
    }

    /// Joint route-trim upward bound after MSO-applied sub boosts.
    ///
    /// The trim and the sub-output gains draw from one pool: a trim that
    /// stacks on top of already-applied MSO boost re-spends headroom the
    /// array already consumed (measured: +6 trim on +6 MSO, then safety cuts
    /// ~-10 and the splice plays cold). Never negative: with the pool spent,
    /// the route stage may only cut.
    ///
    /// # Examples
    ///
    /// ```
    /// use roomeq_model::{RoomConfig, headroom::HeadroomBudget};
    ///
    /// let mut budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
    /// budget.max_route_trim_up_db = 6.0;
    /// assert_eq!(budget.joint_route_trim_up_db(4.0), 2.0);
    /// assert_eq!(budget.joint_route_trim_up_db(9.0), 0.0);
    /// ```
    pub fn joint_route_trim_up_db(&self, applied_sub_boost_db: f64) -> f64 {
        (self.max_route_trim_up_db - applied_sub_boost_db.max(0.0)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_derivation_matches_inlined_config_values() {
        let config = RoomConfig::default();
        let budget = HeadroomBudget::from_legacy_config(&config);
        let expected_boost = config
            .system
            .as_ref()
            .and_then(|system| system.bass_management.as_ref())
            .map(|bm| bm.max_sub_boost_db.max(0.0))
            .unwrap_or(config.optimizer.max_db.max(0.0));
        assert_eq!(budget.max_route_trim_up_db, expected_boost);
        assert_eq!(budget.max_output_boost_db, expected_boost);
        assert_eq!(budget.max_route_trim_down_db, config.optimizer.min_db);
        assert_eq!(
            budget.output_ceiling_dbfs,
            config.optimizer.finalization.output_ceiling_dbfs
        );
        assert_eq!(
            budget.max_attenuation_db,
            config.optimizer.finalization.max_attenuation_db
        );
        assert_eq!(
            budget.max_useful_output_loss_db,
            config.optimizer.finalization.max_useful_output_loss_db
        );
        assert_eq!(budget.strategy, HeadroomStrategy::Fixed);
    }

    #[test]
    fn legacy_derivation_matches_default_policy_numbers() {
        // Pure default: no system block, so the trim ceiling falls back to
        // optimizer.max_db (4.0). Fixtures with a bass-management block get
        // max_sub_boost_db (6.0); see below.
        let budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
        assert_eq!(budget.max_route_trim_up_db, 4.0);
        assert_eq!(budget.max_output_boost_db, 4.0);
        assert_eq!(budget.output_ceiling_dbfs, 0.0);
        assert_eq!(budget.max_attenuation_db, 12.0);
        assert_eq!(budget.max_useful_output_loss_db, 3.0);
    }

    #[test]
    fn legacy_derivation_prefers_system_sub_boost_ceiling() {
        let mut config = RoomConfig::default();
        let mut system = crate::SystemConfig::default();
        system.bass_management = Some(crate::BassManagementConfig {
            max_sub_boost_db: 6.0,
            ..Default::default()
        });
        config.system = Some(system);
        let budget = HeadroomBudget::from_legacy_config(&config);
        assert_eq!(budget.max_route_trim_up_db, 6.0);
        assert_eq!(budget.max_output_boost_db, 6.0);
    }

    #[test]
    fn legacy_role_authority_is_uniform() {
        let budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
        assert_eq!(budget.main, budget.sub);
    }

    #[test]
    fn joint_trim_up_spends_one_pool_and_floors_at_zero() {
        let mut budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
        // Fixture-like ceiling (measured unknown: 6.0 dB pool).
        budget.max_route_trim_up_db = 6.0;
        assert!((budget.joint_route_trim_up_db(0.0) - 6.0).abs() < 1e-12);
        assert!((budget.joint_route_trim_up_db(4.833) - 1.167).abs() < 1e-9);
        assert_eq!(budget.joint_route_trim_up_db(6.0), 0.0);
        assert_eq!(budget.joint_route_trim_up_db(20.0), 0.0);
        assert_eq!(budget.joint_route_trim_up_db(-3.0), 6.0);
    }
}
