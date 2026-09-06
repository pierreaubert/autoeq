//! Realization-chain constraints and temporal gates (Stage 3).
//!
//! Numerical stability, headroom, latency, and export limits are defined
//! numbers on the final chain — not perceptual claims. Temporal gates
//! additionally require trustworthy input timing *and* a stated basis:
//! an engineering limit (e.g. an output-class latency budget) or a
//! validated perceptual rule pinned to its Stage 2 protocol hash. Without
//! both, the finding is reported as advisory, never enforced. Pre-ringing
//! evidence carries its measurement definition (normalization, band,
//! time origin, window, peak-vs-energy), because a bare dB number without
//! that definition cannot be compared against any bound.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Stated basis for a temporal gate limit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TemporalGateBasis {
    /// Engineering limit with its rationale, e.g. `"FIR output-class
    /// latency budget"`. Never presented as a hearing threshold.
    EngineeringLimit {
        /// Why this limit exists.
        rationale: String,
    },
    /// Perceptual rule validated by a staged listening protocol.
    ValidatedPerceptual {
        /// Preregistration hash of the Stage 2 protocol that validated it.
        protocol_hash: String,
    },
}

/// How a pre-ringing number was measured. All fields are required: a
/// pre-ringing bound is an engineering limit unless its normalization,
/// band weighting, time origin/window, and peak-versus-energy definition
/// are specified and validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PreRingingDefinition {
    /// Normalization, e.g. `"peak-normalized to direct arrival"`.
    pub normalization: String,
    /// Analysis band in Hz.
    pub band_hz: [f64; 2],
    /// Time origin, e.g. `"direct-arrival peak"`.
    pub time_origin: String,
    /// Analysis window in ms.
    pub window_ms: f64,
    /// Whether the number is a precursor peak or windowed energy.
    pub peak_or_energy: PeakOrEnergy,
}

/// Peak-versus-energy definition of a pre-ringing number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PeakOrEnergy {
    /// Precursor peak relative to the main transient.
    Peak,
    /// Precursor-window energy relative to total energy.
    Energy,
}

impl PreRingingDefinition {
    /// Every definition field must be stated and the band/window sane.
    pub fn validate(&self) -> Result<(), String> {
        for (name, text) in [
            ("normalization", &self.normalization),
            ("time_origin", &self.time_origin),
        ] {
            if text.trim().is_empty() {
                return Err(format!("pre-ringing {name} must be stated"));
            }
        }
        if !(0.0 < self.band_hz[0] && self.band_hz[0] < self.band_hz[1])
            || !self.band_hz[1].is_finite()
        {
            return Err(String::from("pre-ringing band_hz must be a finite ascending pair"));
        }
        if !self.window_ms.is_finite() || self.window_ms <= 0.0 {
            return Err(String::from("pre-ringing window_ms must be finite and positive"));
        }
        Ok(())
    }
}

/// One temporal gate: a limit plus the conditions for enforcing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemporalGate {
    /// Gate name, e.g. `"induced-group-delay-rms"`.
    pub name: String,
    /// Enforcement limit, in the measured units.
    pub limit: f64,
    /// Stated basis for the limit.
    pub basis: TemporalGateBasis,
    /// Whether the input timing behind the measurement is trustworthy.
    /// Temporal gates require trustworthy timing: without it the finding
    /// is advisory even when the number exceeds the limit.
    pub timing_trusted: bool,
}

/// Outcome of a temporal gate. Advisory outcomes are reports, never
/// violations — enforcement needs a stated basis, trusted timing, and a
/// measured excess, all three.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum GateOutcome {
    /// Enforced and within the limit.
    Pass,
    /// Enforced and over the limit (a violation for the caller to record).
    Fail {
        /// Human-readable excess description.
        detail: String,
    },
    /// Reported but not enforced, with the reason.
    Advisory {
        /// Why this finding is advisory rather than enforced.
        reason: String,
    },
}

impl TemporalGate {
    /// Validate the gate shape: named, finite limit, stated basis.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err(String::from("temporal gate needs a name"));
        }
        if !self.limit.is_finite() {
            return Err(String::from("temporal gate limit must be finite"));
        }
        match &self.basis {
            TemporalGateBasis::EngineeringLimit { rationale } if rationale.trim().is_empty() => {
                Err(String::from("engineering-limit gates need a stated rationale"))
            }
            TemporalGateBasis::ValidatedPerceptual { protocol_hash }
                if protocol_hash.trim().is_empty() =>
            {
                Err(String::from(
                    "perceptual gates need the validating protocol hash",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Apply the gate to one measurement (`None` = evidence missing).
    pub fn apply(&self, measured: Option<f64>, unit: &str) -> Result<GateOutcome, String> {
        self.validate()?;
        let Some(value) = measured else {
            return Ok(GateOutcome::Advisory {
                reason: format!("{}: evidence missing, unassessed", self.name),
            });
        };
        if !value.is_finite() {
            return Ok(GateOutcome::Advisory {
                reason: format!("{}: measurement non-finite, unassessed", self.name),
            });
        }
        if !self.timing_trusted {
            return Ok(GateOutcome::Advisory {
                reason: format!(
                    "{} measured {value:.3} {unit} against limit {:.3} {unit}, but input timing is untrusted: reported, not enforced",
                    self.name, self.limit
                ),
            });
        }
        if value > self.limit {
            Ok(GateOutcome::Fail {
                detail: format!(
                    "{} measured {value:.3} {unit} exceeds {} limit {:.3} {unit}",
                    self.name,
                    match &self.basis {
                        TemporalGateBasis::EngineeringLimit { .. } => "engineering",
                        TemporalGateBasis::ValidatedPerceptual { .. } => "validated-perceptual",
                    },
                    self.limit
                ),
            })
        } else {
            Ok(GateOutcome::Pass)
        }
    }
}

/// Defined limits on the final realization chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChainConstraints {
    /// Numerical-stability ceiling: final chain peak gain in dB.
    pub max_peak_gain_db: f64,
    /// Available-headroom floor in dB (digital headroom only — not
    /// driver-excursion protection, which needs calibrated
    /// loudspeaker limits and is otherwise reported as unassessed).
    pub min_headroom_db: f64,
    /// Latency ceiling in ms.
    pub max_latency_ms: f64,
    /// Export sample rates permitted for this chain.
    pub export_sample_rates_hz: Vec<u32>,
}

/// Measured evidence for the final chain. `None` fields are missing
/// evidence and fail closed as violations, matching the runtime
/// acceptance precedent (`pre_ringing_evidence_missing`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChainEvidence {
    /// Final chain peak gain in dB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gain_db: Option<f64>,
    /// Available headroom in dB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headroom_db: Option<f64>,
    /// End-to-end latency in ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    /// Actual export sample rate in Hz.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export_sample_rate_hz: Option<u32>,
}

/// Check final-chain evidence against defined constraints.
///
/// Returns the violation list (empty = pass). Missing evidence is a
/// violation, never a pass: an unmeasured chain is not a stable chain.
pub fn evaluate_chain_constraints(
    evidence: &ChainEvidence,
    constraints: &ChainConstraints,
) -> Result<Vec<String>, String> {
    if !constraints.max_peak_gain_db.is_finite()
        || !constraints.min_headroom_db.is_finite()
        || !constraints.max_latency_ms.is_finite()
        || constraints.max_latency_ms < 0.0
    {
        return Err(String::from("chain constraints contain invalid limits"));
    }
    if constraints.export_sample_rates_hz.is_empty() {
        return Err(String::from(
            "chain constraints must permit at least one export sample rate",
        ));
    }
    let mut violations = Vec::new();
    match evidence.peak_gain_db {
        Some(value) if value.is_finite() => {
            if value > constraints.max_peak_gain_db {
                violations.push(format!(
                    "peak_gain {value:.2} dB exceeds stability ceiling {:.2} dB",
                    constraints.max_peak_gain_db
                ));
            }
        }
        _ => violations.push(String::from("peak_gain_evidence_missing")),
    }
    match evidence.headroom_db {
        Some(value) if value.is_finite() => {
            if value < constraints.min_headroom_db {
                violations.push(format!(
                    "headroom {value:.2} dB below floor {:.2} dB",
                    constraints.min_headroom_db
                ));
            }
        }
        _ => violations.push(String::from("headroom_evidence_missing")),
    }
    match evidence.latency_ms {
        Some(value) if value.is_finite() => {
            if value > constraints.max_latency_ms {
                violations.push(format!(
                    "latency {value:.2} ms exceeds ceiling {:.2} ms",
                    constraints.max_latency_ms
                ));
            }
        }
        _ => violations.push(String::from("latency_evidence_missing")),
    }
    match evidence.export_sample_rate_hz {
        Some(rate) => {
            if !constraints.export_sample_rates_hz.contains(&rate) {
                violations.push(format!(
                    "export rate {rate} Hz not in permitted {rates:?}",
                    rates = constraints.export_sample_rates_hz
                ));
            }
        }
        None => violations.push(String::from("export_rate_evidence_missing")),
    }
    Ok(violations)
}

#[cfg(test)]
mod chain_constraints_tests {
    use super::*;

    fn engineering_gate() -> TemporalGate {
        TemporalGate {
            name: String::from("induced-group-delay-rms"),
            limit: 10.0,
            basis: TemporalGateBasis::EngineeringLimit {
                rationale: String::from("hybrid output-class latency budget"),
            },
            timing_trusted: true,
        }
    }

    #[test]
    fn gate_enforces_only_with_basis_and_trusted_timing() {
        // Trusted timing + stated basis: excess fails, margin passes.
        assert!(matches!(
            engineering_gate().apply(Some(4.0), "ms").unwrap(),
            GateOutcome::Pass
        ));
        let fail = engineering_gate().apply(Some(12.0), "ms").unwrap();
        assert!(matches!(fail, GateOutcome::Fail { .. }));
        // Untrusted timing: same excess is advisory, never a failure.
        let mut untrusted = engineering_gate();
        untrusted.timing_trusted = false;
        let outcome = untrusted.apply(Some(12.0), "ms").unwrap();
        assert!(matches!(outcome, GateOutcome::Advisory { .. }));
        // Missing or non-finite evidence is advisory, never enforced.
        assert!(matches!(
            engineering_gate().apply(None, "ms").unwrap(),
            GateOutcome::Advisory { .. }
        ));
        assert!(matches!(
            engineering_gate().apply(Some(f64::NAN), "ms").unwrap(),
            GateOutcome::Advisory { .. }
        ));
    }

    #[test]
    fn gate_basis_must_be_stated() {
        let mut unstated = engineering_gate();
        unstated.basis = TemporalGateBasis::EngineeringLimit {
            rationale: String::from("  "),
        };
        assert!(unstated.apply(Some(1.0), "ms").is_err());
        unstated.basis = TemporalGateBasis::ValidatedPerceptual {
            protocol_hash: String::new(),
        };
        assert!(unstated.apply(Some(1.0), "ms").is_err());
        unstated.basis = TemporalGateBasis::ValidatedPerceptual {
            protocol_hash: String::from("abc123"),
        };
        assert!(matches!(
            unstated.apply(Some(1.0), "ms").unwrap(),
            GateOutcome::Pass
        ));
        unstated.limit = f64::INFINITY;
        assert!(unstated.apply(Some(1.0), "ms").is_err());
    }

    #[test]
    fn pre_ringing_definition_must_be_complete() {
        let definition = PreRingingDefinition {
            normalization: String::from("peak-normalized to direct arrival"),
            band_hz: [2000.0, 8000.0],
            time_origin: String::from("direct-arrival peak"),
            window_ms: 10.0,
            peak_or_energy: PeakOrEnergy::Peak,
        };
        assert!(definition.validate().is_ok());
        let mut blank = definition.clone();
        blank.normalization = String::from("");
        assert!(blank.validate().is_err());
        blank = definition.clone();
        blank.band_hz = [8000.0, 2000.0];
        assert!(blank.validate().is_err());
        blank = definition.clone();
        blank.window_ms = 0.0;
        assert!(blank.validate().is_err());
    }

    fn constraints() -> ChainConstraints {
        ChainConstraints {
            max_peak_gain_db: 12.0,
            min_headroom_db: -12.0,
            max_latency_ms: 100.0,
            export_sample_rates_hz: vec![44_100, 48_000],
        }
    }

    fn evidence() -> ChainEvidence {
        ChainEvidence {
            peak_gain_db: Some(6.0),
            headroom_db: Some(-3.0),
            latency_ms: Some(20.0),
            export_sample_rate_hz: Some(48_000),
        }
    }

    #[test]
    fn chain_within_limits_passes() {
        assert!(evaluate_chain_constraints(&evidence(), &constraints()).unwrap().is_empty());
    }

    #[test]
    fn chain_excess_is_reported_per_limit() {
        let mut hot = evidence();
        hot.peak_gain_db = Some(15.0);
        hot.headroom_db = Some(-20.0);
        hot.latency_ms = Some(400.0);
        hot.export_sample_rate_hz = Some(96_000);
        let violations = evaluate_chain_constraints(&hot, &constraints()).unwrap();
        assert_eq!(violations.len(), 4);
    }

    #[test]
    fn missing_chain_evidence_fails_closed() {
        let missing = ChainEvidence {
            peak_gain_db: None,
            headroom_db: None,
            latency_ms: None,
            export_sample_rate_hz: None,
        };
        let violations = evaluate_chain_constraints(&missing, &constraints()).unwrap();
        assert_eq!(violations.len(), 4);
        assert!(violations.iter().all(|violation| violation.ends_with("missing")));
    }
}
