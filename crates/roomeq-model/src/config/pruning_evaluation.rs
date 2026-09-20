//! Explicit measurement, programme, and nominal-level declarations for experimental pruning.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Version of the experimental condition interpretation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruningEvaluationVersion {
    /// Magnitude spectra with each condition anchored to its frozen full chain.
    SpectralV1,
}

/// Programme magnitude spectrum used by the experimental pruning proxy.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PruningProgramme {
    /// Stable, nonempty, slash-free identifier; not a file path.
    pub id: String,
    /// Strictly increasing positive frequencies covering the evaluation grid.
    pub frequencies_hz: Vec<f64>,
    /// Finite relative dB magnitudes, one per frequency.
    ///
    /// This spectrum does not encode temporal masking or prove programme-audio
    /// equivalence. No SPL calibration is implied by its offset.
    pub spectrum_db: Vec<f64>,
}

/// Complete experimental pruning condition declaration for one EQ measurement set.
///
/// All measurements are crossed with all programmes and levels. Identifiers
/// follow `<measurement>/<programme>/<level>phon`, for example
/// `seat-a/music/75phon`. A consumer must retain filters if it cannot supply
/// every measurement or support the declared comparison domain.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PruningEvaluation {
    /// Required version pin; unknown versions are rejected during deserialization.
    pub version: PruningEvaluationVersion,
    /// Stable identifiers in the same order as the supplied measurements.
    ///
    /// The number must match the complete measurement set, including measurements
    /// that an optimizer averages or assigns zero weight. Never silently truncate.
    pub measurement_ids: Vec<String>,
    /// Programme spectra applied independently to each measured response.
    pub programmes: Vec<PruningProgramme>,
    /// Positive finite nominal full-chain levels, evaluated separately per condition.
    ///
    /// These are experimental proxy assumptions, not measured playback SPL.
    pub listening_levels_phon: Vec<f64>,
}

impl PruningEvaluation {
    /// Validates the declaration without claiming measurement or auditory validity.
    ///
    /// # Errors
    /// Returns a reason for empty or duplicated conditions, malformed identifiers,
    /// invalid spectra or levels, or an unrepresentable condition count.
    pub fn validate(&self) -> Result<(), String> {
        validate_ids(
            self.measurement_ids.iter().map(String::as_str),
            "measurement",
        )?;
        validate_ids(
            self.programmes
                .iter()
                .map(|programme| programme.id.as_str()),
            "programme",
        )?;
        if self.listening_levels_phon.is_empty()
            || self
                .listening_levels_phon
                .iter()
                .any(|level| !level.is_finite() || *level <= 0.0)
        {
            return Err(String::from(
                "listening_levels_phon must contain positive finite nominal levels",
            ));
        }
        let levels: std::collections::BTreeSet<_> = self
            .listening_levels_phon
            .iter()
            .map(|level| level.to_bits())
            .collect();
        if levels.len() != self.listening_levels_phon.len() {
            return Err(String::from(
                "listening_levels_phon must not contain duplicates",
            ));
        }
        for programme in &self.programmes {
            if programme.frequencies_hz.len() < 2
                || programme.frequencies_hz.len() != programme.spectrum_db.len()
                || programme
                    .frequencies_hz
                    .iter()
                    .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
                || programme
                    .frequencies_hz
                    .windows(2)
                    .any(|pair| pair[1] <= pair[0])
                || programme
                    .spectrum_db
                    .iter()
                    .any(|magnitude| !magnitude.is_finite())
            {
                return Err(format!(
                    "programme {} requires aligned finite magnitudes and an increasing positive frequency grid",
                    programme.id
                ));
            }
        }
        self.condition_count()?;
        Ok(())
    }

    fn condition_count(&self) -> Result<usize, String> {
        self.measurement_ids
            .len()
            .checked_mul(self.programmes.len())
            .and_then(|count| count.checked_mul(self.listening_levels_phon.len()))
            .ok_or_else(|| String::from("pruning condition count overflow"))
    }

    /// Returns every declared condition identifier in reproducible traversal order.
    ///
    /// # Errors
    /// Returns a validation error or an allocation failure for the condition set.
    pub fn condition_ids(&self) -> Result<Vec<String>, String> {
        self.validate()?;
        let mut ids = Vec::new();
        ids.try_reserve(self.condition_count()?)
            .map_err(|error| format!("cannot allocate pruning condition identifiers: {error}"))?;
        for measurement in &self.measurement_ids {
            for programme in &self.programmes {
                for level in &self.listening_levels_phon {
                    ids.push(format!("{measurement}/{}/{level}phon", programme.id));
                }
            }
        }
        Ok(ids)
    }
}

fn validate_ids<'a>(ids: impl Iterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for id in ids {
        if id.is_empty()
            || id.trim() != id
            || id.contains('/')
            || id.chars().any(char::is_control)
            || !seen.insert(id)
        {
            return Err(format!(
                "{kind} identifiers must be unique, nonempty, slash-free, and free of control characters or surrounding whitespace"
            ));
        }
    }
    if seen.is_empty() {
        return Err(format!("at least one {kind} is required"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration() -> PruningEvaluation {
        PruningEvaluation {
            version: PruningEvaluationVersion::SpectralV1,
            measurement_ids: vec![String::from("seat-a"), String::from("seat-b")],
            programmes: vec![PruningProgramme {
                id: String::from("music"),
                frequencies_hz: vec![20.0, 20_000.0],
                spectrum_db: vec![0.0, -6.0],
            }],
            listening_levels_phon: vec![55.0, 75.0],
        }
    }

    #[test]
    fn qa_roomeq_pruning_conditions_declaration_is_versioned_and_complete() {
        let evaluation = declaration();
        assert_eq!(
            evaluation.condition_ids().unwrap(),
            vec![
                "seat-a/music/55phon",
                "seat-a/music/75phon",
                "seat-b/music/55phon",
                "seat-b/music/75phon",
            ]
        );
        let mut encoded = serde_json::to_value(&evaluation).unwrap();
        assert_eq!(encoded["version"], "spectral-v1");
        assert_eq!(
            serde_json::from_value::<PruningEvaluation>(encoded.clone()).unwrap(),
            evaluation
        );
        encoded["version"] = serde_json::json!("spectral-v2");
        assert!(serde_json::from_value::<PruningEvaluation>(encoded).is_err());
        let legacy: super::super::report_outcome::PruningBudget =
            serde_json::from_str("{}").unwrap();
        assert!(legacy.evaluation.is_none());
        assert!(
            serde_json::to_value(legacy)
                .unwrap()
                .get("evaluation")
                .is_none()
        );
    }

    #[test]
    fn qa_roomeq_pruning_conditions_malformed_declarations_are_rejected() {
        let mut cases = Vec::new();
        let mut duplicate = declaration();
        duplicate.measurement_ids[1] = duplicate.measurement_ids[0].clone();
        cases.push(duplicate);
        let mut empty = declaration();
        empty.programmes.clear();
        cases.push(empty);
        let mut ambiguous = declaration();
        ambiguous.programmes[0].id = String::from("music/75phon");
        cases.push(ambiguous);
        let mut descending = declaration();
        descending.programmes[0].frequencies_hz.reverse();
        cases.push(descending);
        let mut misaligned = declaration();
        misaligned.programmes[0].spectrum_db.pop();
        cases.push(misaligned);
        let mut nonfinite = declaration();
        nonfinite.programmes[0].spectrum_db[0] = f64::NAN;
        cases.push(nonfinite);
        let mut bad_level = declaration();
        bad_level.listening_levels_phon[0] = f64::INFINITY;
        cases.push(bad_level);
        let mut duplicate_level = declaration();
        duplicate_level.listening_levels_phon = vec![75.0, 75.0];
        cases.push(duplicate_level);
        for invalid in cases {
            assert!(invalid.validate().is_err(), "{invalid:?}");
            assert!(invalid.condition_ids().is_err());
        }
    }
}
