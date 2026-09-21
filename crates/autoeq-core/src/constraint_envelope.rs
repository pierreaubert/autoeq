//! Pure frequency-dependent maximum-Q constraint envelope (K3).
//!
//! No equivalent pure local-Q type exists elsewhere: gain envelopes live
//! in the model/optimizer layers, so this crate owns the reusable Q
//! values. Evaluation interpolates linearly in log frequency with
//! endpoint hold; the effective limit is the stricter of the global
//! cap and the local envelope. An absent envelope is identity.

// Rust guideline compliant 2026-02-21

use crate::error::{AutoeqError, Result};
use serde::{Deserialize, Serialize};

/// One maximum-Q knot: at most `max_q` applies at `freq_hz`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalQKnot {
    /// Knot frequency in Hz: finite, positive, strictly increasing.
    pub freq_hz: f64,
    /// Maximum allowed Q at this knot: finite and positive.
    pub max_q: f64,
}

/// Frequency-dependent maximum-Q envelope for optimizer filter centers.
///
/// Knots use the same contract as the model gain envelopes: linear
/// interpolation in log frequency, endpoint hold within the requested
/// correction band, no validity claimed beyond measured support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalQEnvelope {
    /// Ordered knots, at least one.
    pub knots: Vec<LocalQKnot>,
}

impl LocalQEnvelope {
    /// Build an envelope, rejecting invalid knots.
    ///
    /// # Errors
    /// Returns [`AutoeqError::InvalidConfiguration`] for an empty knot
    /// list, non-finite or non-positive frequencies, non-finite or
    /// non-positive Q values, or unordered/duplicated frequencies.
    pub fn new(knots: Vec<LocalQKnot>) -> Result<Self> {
        let envelope = Self { knots };
        envelope.validate("local-Q envelope")?;
        Ok(envelope)
    }

    /// Check knot finiteness, positivity, and strict ordering.
    ///
    /// # Errors
    /// Returns [`AutoeqError::InvalidConfiguration`] as in [`Self::new`].
    pub fn validate(&self, context: &str) -> Result<()> {
        if self.knots.is_empty() {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!("{context} needs at least one knot"),
            });
        }
        for (index, knot) in self.knots.iter().enumerate() {
            if !knot.freq_hz.is_finite() || knot.freq_hz <= 0.0 {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "{context} knot {index} frequency must be finite and positive, got {}",
                        knot.freq_hz
                    ),
                });
            }
            if !knot.max_q.is_finite() || knot.max_q <= 0.0 {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "{context} knot {index} max-Q must be finite and positive, got {}",
                        knot.max_q
                    ),
                });
            }
        }
        if self
            .knots
            .windows(2)
            .any(|pair| pair[0].freq_hz >= pair[1].freq_hz)
        {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!("{context} knot frequencies must be strictly increasing"),
            });
        }
        Ok(())
    }

    /// Maximum Q at a filter-center frequency in Hz.
    ///
    /// Interior frequencies interpolate linearly in log frequency;
    /// frequencies outside the knot span hold the nearest endpoint.
    ///
    /// # Errors
    /// Returns [`AutoeqError::InvalidConfiguration`] for an invalid
    /// envelope or a non-finite/non-positive query frequency.
    pub fn max_q_at_freq(&self, freq_hz: f64) -> Result<f64> {
        self.validate("local-Q envelope")?;
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "local-Q query frequency must be finite and positive, got {freq_hz}"
                ),
            });
        }
        let knots = &self.knots;
        if freq_hz <= knots[0].freq_hz {
            return Ok(knots[0].max_q);
        }
        let last = knots.len() - 1;
        if freq_hz >= knots[last].freq_hz {
            return Ok(knots[last].max_q);
        }
        for pair in knots.windows(2) {
            if freq_hz >= pair[0].freq_hz && freq_hz <= pair[1].freq_hz {
                let fraction = (freq_hz.ln() - pair[0].freq_hz.ln())
                    / (pair[1].freq_hz.ln() - pair[0].freq_hz.ln());
                return Ok(pair[0].max_q + fraction * (pair[1].max_q - pair[0].max_q));
            }
        }
        Ok(knots[last].max_q)
    }
}

/// Effective maximum Q: the stricter of the global cap and the envelope.
///
/// A `None` envelope returns the global cap exactly (identity): absence
/// preserves legacy behavior and never introduces a new default.
///
/// # Errors
/// Returns [`AutoeqError::InvalidConfiguration`] for a non-finite or
/// non-positive global cap, an invalid query frequency, or an invalid
/// envelope.
pub fn effective_max_q(
    envelope: Option<&LocalQEnvelope>,
    freq_hz: f64,
    global_max_q: f64,
) -> Result<f64> {
    if !global_max_q.is_finite() || global_max_q <= 0.0 {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!("global max-Q must be finite and positive, got {global_max_q}"),
        });
    }
    match envelope {
        None => {
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "local-Q query frequency must be finite and positive, got {freq_hz}"
                    ),
                });
            }
            Ok(global_max_q)
        }
        Some(envelope) => Ok(global_max_q.min(envelope.max_q_at_freq(freq_hz)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> LocalQEnvelope {
        LocalQEnvelope::new(vec![
            LocalQKnot {
                freq_hz: 100.0,
                max_q: 2.0,
            },
            LocalQKnot {
                freq_hz: 1000.0,
                max_q: 4.0,
            },
        ])
        .unwrap()
    }

    #[test]
    fn core_constraint_envelope_log_interpolation() {
        let envelope = envelope();
        // Midpoint in log frequency carries the arithmetic midpoint limit.
        let midpoint = (100.0_f64 * 1000.0).sqrt();
        let value = envelope.max_q_at_freq(midpoint).unwrap();
        assert!(
            (value - 3.0).abs() <= 1e-9,
            "expected midpoint limit 3.0, got {value}"
        );

        // Endpoint hold within the requested correction band.
        assert_eq!(envelope.max_q_at_freq(50.0).unwrap(), 2.0);
        assert_eq!(envelope.max_q_at_freq(100.0).unwrap(), 2.0);
        assert_eq!(envelope.max_q_at_freq(1000.0).unwrap(), 4.0);
        assert_eq!(envelope.max_q_at_freq(2000.0).unwrap(), 4.0);

        // Invalid knots are rejected: duplicates, NaN, infinity, bad Q.
        assert!(
            LocalQEnvelope::new(vec![
                LocalQKnot {
                    freq_hz: 100.0,
                    max_q: 2.0,
                },
                LocalQKnot {
                    freq_hz: 100.0,
                    max_q: 3.0,
                },
            ])
            .is_err()
        );
        assert!(
            LocalQEnvelope::new(vec![LocalQKnot {
                freq_hz: f64::NAN,
                max_q: 2.0,
            }])
            .is_err()
        );
        assert!(
            LocalQEnvelope::new(vec![LocalQKnot {
                freq_hz: f64::INFINITY,
                max_q: 2.0,
            }])
            .is_err()
        );
        for bad_q in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                LocalQEnvelope::new(vec![LocalQKnot {
                    freq_hz: 100.0,
                    max_q: bad_q,
                }])
                .is_err(),
                "accepted max-Q {bad_q}"
            );
        }
        assert!(LocalQEnvelope::new(Vec::new()).is_err());
    }

    #[test]
    fn core_constraint_envelope_stricter_global_limit() {
        let envelope = envelope();
        // Local policy never relaxes the global cap.
        assert_eq!(effective_max_q(Some(&envelope), 100.0, 1.5).unwrap(), 1.5);
        assert_eq!(effective_max_q(Some(&envelope), 1000.0, 3.0).unwrap(), 3.0);
        // The local envelope caps where it is stricter.
        assert_eq!(effective_max_q(Some(&envelope), 100.0, 6.0).unwrap(), 2.0);
        assert_eq!(effective_max_q(Some(&envelope), 1000.0, 6.0).unwrap(), 4.0);
        let midpoint = (100.0_f64 * 1000.0).sqrt();
        assert_eq!(
            effective_max_q(Some(&envelope), midpoint, 6.0).unwrap(),
            3.0
        );
        // Effective never exceeds the global cap anywhere in the span.
        for freq in [50.0, 100.0, 316.0, 1000.0, 5000.0] {
            let effective = effective_max_q(Some(&envelope), freq, 2.5).unwrap();
            assert!(
                effective <= 2.5,
                "relaxed global cap at {freq}: {effective}"
            );
        }
    }

    #[test]
    fn core_constraint_envelope_absent_is_identity() {
        // Absence returns the global cap exactly: legacy behavior.
        for (freq, global) in [(50.0, 4.25), (100.0, 1.0), (1000.0, 10.0)] {
            assert_eq!(effective_max_q(None, freq, global).unwrap(), global);
        }
        assert!(effective_max_q(None, 100.0, 0.0).is_err());
        assert!(effective_max_q(None, f64::NAN, 4.0).is_err());
    }
}
