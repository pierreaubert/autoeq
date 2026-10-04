//! Corpus membership, held-out separation, and comparison binding (Q3).
//!
//! Validation over acoustic-corpus entries. Independently supplied reference
//! files are hashed and kept distinct from generated controls; baseline and
//! candidate comparisons bind to identical inputs and budgets.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::Read;
use std::path::PathBuf;

// Rust guideline compliant 2026-02-21

/// Where a corpus capture comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusProvenance {
    /// Deterministic programme fixture (e.g. `fem_*` scenarios).
    GeneratedControl,
    /// Vector supplied independently of the programme (model reference).
    IndependentReference,
    /// Contributor or public measured room capture.
    RealMeasurement,
}

impl fmt::Display for CorpusProvenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeneratedControl => write!(f, "generated_control"),
            Self::IndependentReference => write!(f, "independent_reference"),
            Self::RealMeasurement => write!(f, "real_measurement"),
        }
    }
}

/// One capture registered in the acoustic corpus.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CorpusCapture {
    /// Scenario id, e.g. `"fem_medium_multiseat_20"`.
    pub scenario: String,
    /// Source id (loudspeaker or channel source).
    pub source: String,
    /// Seat id (listening position).
    pub seat: String,
    /// Stable capture identity.
    pub capture_id: String,
    /// Content hash of the capture payload.
    pub capture_hash: String,
    /// Where the capture comes from.
    pub provenance: CorpusProvenance,
    /// True when the candidate trains on this capture; false for held-out.
    pub trains_candidate: bool,
}

/// Reject captures with missing seat/source identity or duplicated payloads.
///
/// A duplicated capture hash under a different seat would invent seats by
/// copying; each seat must contribute its own capture. Measured and independent
/// reference sources also need distinct payloads, and one payload cannot carry
/// contradictory acquisition provenance. Generated controls may deliberately
/// use identical responses for different sources at the same seat.
///
/// # Errors
///
/// Returns an error for missing identities, duplicate capture IDs, or conflicting payload identities.
pub fn validate_corpus(captures: &[CorpusCapture]) -> Result<(), String> {
    if captures.is_empty() {
        return Err(String::from("corpus has no captures"));
    }
    let mut ids = HashSet::new();
    let mut hashes: HashMap<&str, &CorpusCapture> = HashMap::new();
    for capture in captures {
        if capture.scenario.trim().is_empty() {
            return Err(format!("capture '{}' has no scenario", capture.capture_id));
        }
        if capture.seat.trim().is_empty() {
            return Err(format!(
                "capture '{}' in scenario '{}' has no seat",
                capture.capture_id, capture.scenario
            ));
        }
        if capture.source.trim().is_empty() {
            return Err(format!(
                "capture '{}' in scenario '{}' has no source",
                capture.capture_id, capture.scenario
            ));
        }
        if capture.capture_id.trim().is_empty() || capture.capture_hash.trim().is_empty() {
            return Err(format!(
                "capture in scenario '{}' seat '{}' has no id or hash",
                capture.scenario, capture.seat
            ));
        }
        if !ids.insert(capture.capture_id.as_str()) {
            return Err(format!("duplicate capture id '{}'", capture.capture_id));
        }
        if let Some(first) = hashes.insert(capture.capture_hash.as_str(), capture) {
            let same_seat = first.scenario == capture.scenario && first.seat == capture.seat;
            if first.provenance != capture.provenance {
                return Err(format!(
                    "capture hash '{}' has contradictory acquisition provenance",
                    capture.capture_hash,
                ));
            }
            if first.source != capture.source
                && capture.provenance != CorpusProvenance::GeneratedControl
            {
                return Err(format!(
                    "capture hash '{}' shared by sources '{}' and '{}': independent sources must not duplicate captures",
                    capture.capture_hash, first.source, capture.source,
                ));
            }
            if !same_seat {
                return Err(format!(
                    "capture hash '{}' shared by '{}:{}' and '{}:{}': seats must not duplicate captures",
                    capture.capture_hash,
                    first.scenario,
                    first.seat,
                    capture.scenario,
                    capture.seat
                ));
            }
        }
    }
    Ok(())
}

/// Enforce held-out partition separation: no seat trains and validates.
///
/// Returns the held-out seat count per scenario. A scenario with no
/// held-out seats is report-only and must not train a candidate.
pub fn validate_held_out_separation(
    captures: &[CorpusCapture],
) -> Result<HashMap<String, usize>, String> {
    validate_corpus(captures)?;
    let mut training: HashSet<(&str, &str)> = HashSet::new();
    let mut held_out: HashSet<(&str, &str)> = HashSet::new();
    for capture in captures {
        let key = (capture.scenario.as_str(), capture.seat.as_str());
        if capture.trains_candidate {
            training.insert(key);
        } else {
            held_out.insert(key);
        }
    }
    let leaked: Vec<String> = training
        .intersection(&held_out)
        .map(|(scenario, seat)| format!("{scenario}:{seat}"))
        .collect();
    if !leaked.is_empty() {
        return Err(format!(
            "held-out seats also train the candidate: {}",
            leaked.join(", ")
        ));
    }
    let mut per_scenario: HashMap<String, usize> = HashMap::new();
    for (scenario, _) in &held_out {
        *per_scenario.entry((*scenario).to_string()).or_default() += 1;
    }
    for capture in captures {
        if capture.trains_candidate && !per_scenario.contains_key(capture.scenario.as_str()) {
            return Err(format!(
                "scenario '{}' trains a candidate with no held-out seat",
                capture.scenario
            ));
        }
    }
    Ok(per_scenario)
}

/// A reference vector with its supply-line identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceVector {
    /// Pinned set identity shared by all vectors used for one model claim.
    pub set_id: String,
    /// Stable vector id.
    pub id: String,
    /// SHA-256 of the exact vector payload.
    pub hash: String,
    /// Local artifact containing the exact independent vector payload.
    pub artifact_path: PathBuf,
    /// Pinned edition of the reference model or published vector set.
    pub edition: String,
    /// License identifier governing use of the vector payload.
    pub license_id: String,
    /// True only for vectors supplied independently of the programme.
    pub independent: bool,
}

fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Keep independently supplied references distinct from generated controls.
///
/// Independent vectors need edition/license metadata and an artifact whose
/// bytes match the declared SHA-256. They must not share that digest with a
/// generated control; a descriptor alone cannot establish external evidence.
pub fn validate_reference_separation(
    independent: &[ReferenceVector],
    generated: &[ReferenceVector],
) -> Result<(), String> {
    if independent.is_empty() {
        return Err(String::from("no independent reference vectors registered"));
    }
    let mut ids = HashSet::new();
    let first = &independent[0];
    for vector in independent {
        if vector.set_id.trim().is_empty() {
            return Err(String::from("reference vector needs a nonempty set id"));
        }
        if !vector.independent {
            return Err(format!(
                "reference '{}' is not flagged as independently supplied",
                vector.id
            ));
        }
        if vector.id.trim().is_empty() || !ids.insert(vector.id.as_str()) {
            return Err(String::from("reference vector needs a unique nonempty id"));
        }
        if vector.edition.trim().is_empty() || vector.license_id.trim().is_empty() {
            return Err(format!(
                "reference '{}' needs a declared edition and license",
                vector.id
            ));
        }
        if vector.set_id != first.set_id
            || vector.edition != first.edition
            || vector.license_id != first.license_id
        {
            return Err(String::from(
                "independent vectors in one set need matching set, edition and license",
            ));
        }
        if vector.hash.len() != 64 || !vector.hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("reference '{}' needs a SHA-256 hash", vector.id));
        }
        let mut artifact = std::fs::File::open(&vector.artifact_path).map_err(|error| {
            format!(
                "reference '{}' artifact '{}' cannot be opened: {error}",
                vector.id,
                vector.artifact_path.display()
            )
        })?;
        let mut hasher = Sha256::new();
        let mut chunk = [0_u8; 64 * 1024];
        let mut bytes_read = 0_u64;
        loop {
            let count = artifact.read(&mut chunk).map_err(|error| {
                format!("reference '{}' artifact read failed: {error}", vector.id)
            })?;
            if count == 0 {
                break;
            }
            bytes_read += count as u64;
            hasher.update(&chunk[..count]);
        }
        if bytes_read == 0 {
            return Err(format!("reference '{}' artifact is empty", vector.id));
        }
        if sha256_hex(hasher.finalize()) != vector.hash.to_ascii_lowercase() {
            return Err(format!("reference '{}' artifact hash mismatch", vector.id));
        }
    }
    let generated_hashes: HashSet<String> = generated
        .iter()
        .map(|vector| vector.hash.to_ascii_lowercase())
        .collect();
    for vector in independent {
        if generated_hashes.contains(&vector.hash.to_ascii_lowercase()) {
            return Err(format!(
                "reference '{}' shares content hash with a generated control",
                vector.id
            ));
        }
    }
    Ok(())
}

/// Inputs a baseline/candidate comparison must hold identical.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ComparisonBinding {
    /// Immutable baseline graph identity.
    pub baseline_graph_hash: String,
    /// Immutable candidate graph identity.
    pub candidate_graph_hash: String,
    /// Hash of the exact comparison inputs (captures, config, stimulus).
    pub input_hash: String,
    /// Optimizer seed.
    pub seed: u64,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Declared evaluation budget (e.g. max evaluations).
    pub budget_evals: u64,
    /// Comparison policy/limits version.
    pub limits_version: String,
}

/// Check two comparison legs bind to identical inputs, seeds, and budgets.
///
/// Graph identities are recorded, not equated: the candidate is expected
/// to differ from the baseline. Everything else must match.
pub fn validate_comparison_binding(binding: &ComparisonBinding) -> Result<(), String> {
    if binding.baseline_graph_hash.trim().is_empty()
        || binding.candidate_graph_hash.trim().is_empty()
    {
        return Err(String::from(
            "comparison needs baseline and candidate graph hashes",
        ));
    }
    if binding.input_hash.trim().is_empty() {
        return Err(String::from("comparison needs an input hash"));
    }
    if binding.limits_version.trim().is_empty() {
        return Err(String::from("comparison needs a limits version"));
    }
    if !binding.sample_rate.is_finite() || binding.sample_rate <= 0.0 {
        return Err(String::from(
            "comparison needs a finite positive sample rate",
        ));
    }
    if binding.budget_evals == 0 {
        return Err(String::from("comparison needs a nonzero evaluation budget"));
    }
    Ok(())
}

/// Check two legs of one comparison share inputs, seeds, rates, and budgets.
pub fn validate_comparison_pair(
    baseline_leg: &ComparisonBinding,
    candidate_leg: &ComparisonBinding,
) -> Result<(), String> {
    validate_comparison_binding(baseline_leg)?;
    validate_comparison_binding(candidate_leg)?;
    if baseline_leg.input_hash != candidate_leg.input_hash {
        return Err(String::from("comparison legs use different input hashes"));
    }
    if baseline_leg.seed != candidate_leg.seed {
        return Err(String::from("comparison legs use different seeds"));
    }
    if baseline_leg.sample_rate != candidate_leg.sample_rate {
        return Err(String::from("comparison legs use different sample rates"));
    }
    if baseline_leg.budget_evals != candidate_leg.budget_evals {
        return Err(String::from("comparison legs declare different budgets"));
    }
    if baseline_leg.limits_version != candidate_leg.limits_version {
        return Err(String::from(
            "comparison legs use different limits versions",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod corpus_tests {
    use super::*;

    fn capture(
        scenario: &str,
        source: &str,
        seat: &str,
        id: &str,
        hash: &str,
        trains: bool,
    ) -> CorpusCapture {
        CorpusCapture {
            scenario: scenario.to_string(),
            source: source.to_string(),
            seat: seat.to_string(),
            capture_id: id.to_string(),
            capture_hash: hash.to_string(),
            provenance: CorpusProvenance::RealMeasurement,
            trains_candidate: trains,
        }
    }

    fn valid_corpus() -> Vec<CorpusCapture> {
        vec![
            capture("room_a", "L", "seat_1", "cap_1", "hash_1", true),
            capture("room_a", "L", "seat_2", "cap_2", "hash_2", true),
            capture("room_a", "L", "held_out_1", "cap_3", "hash_3", false),
        ]
    }

    #[test]
    fn qa_corpus_missing_seat_or_duplicate_capture_rejected() {
        assert!(validate_corpus(&valid_corpus()).is_ok());
        // Missing seat identity is rejected.
        let mut missing_seat = valid_corpus();
        missing_seat[0].seat.clear();
        assert!(validate_corpus(&missing_seat).is_err());
        // A duplicated payload under another seat invents a seat: rejected.
        let mut duplicated = valid_corpus();
        duplicated.push(capture("room_a", "L", "seat_3", "cap_4", "hash_1", true));
        let error = validate_corpus(&duplicated).unwrap_err();
        assert!(error.contains("must not duplicate captures"), "{error}");
        // A duplicated capture id is rejected even with distinct payloads.
        let mut duplicate_id = valid_corpus();
        duplicate_id.push(capture("room_a", "L", "seat_3", "cap_1", "hash_9", true));
        assert!(validate_corpus(&duplicate_id).is_err());
        // An empty corpus carries no evidence.
        assert!(validate_corpus(&[]).is_err());
    }

    #[test]
    fn measured_payload_cannot_invent_an_independent_source() {
        let mut captures = valid_corpus();
        captures.push(capture("room_a", "R", "seat_1", "cap_4", "hash_1", true));
        assert!(
            validate_corpus(&captures)
                .unwrap_err()
                .contains("independent sources")
        );
    }

    #[test]
    fn duplicate_payload_cannot_change_acquisition_provenance() {
        let mut captures = valid_corpus();
        let mut alias = capture("room_a", "L", "seat_1", "cap_4", "hash_1", true);
        alias.provenance = CorpusProvenance::GeneratedControl;
        captures.push(alias);
        assert!(
            validate_corpus(&captures)
                .unwrap_err()
                .contains("contradictory acquisition provenance")
        );
    }

    #[test]
    fn identical_generated_source_responses_remain_valid_controls() {
        let mut left = capture("analytic", "L", "seat_1", "left", "flat-plant", false);
        left.provenance = CorpusProvenance::GeneratedControl;
        let mut right = left.clone();
        right.source = "R".into();
        right.capture_id = "right".into();
        assert!(validate_corpus(&[left, right]).is_ok());
    }

    #[test]
    fn qa_held_out_partition_never_trains_candidate() {
        let held_out = validate_held_out_separation(&valid_corpus()).unwrap();
        assert_eq!(held_out.get("room_a"), Some(&1));
        // A seat that both trains and validates leaks the partition.
        let mut leaked = valid_corpus();
        leaked.push(capture("room_a", "L", "seat_1", "cap_5", "hash_5", false));
        let error = validate_held_out_separation(&leaked).unwrap_err();
        assert!(error.contains("also train the candidate"), "{error}");
        // Training without any held-out seat is rejected.
        let no_held_out = vec![capture("room_b", "L", "seat_1", "cap_6", "hash_6", true)];
        assert!(validate_held_out_separation(&no_held_out).is_err());
        // Report-only captures without training are accepted.
        let report_only = vec![capture("room_c", "L", "seat_1", "cap_7", "hash_7", false)];
        assert!(validate_held_out_separation(&report_only).is_ok());
    }

    #[test]
    fn qa_manifest_provenance_and_held_out_shape() {
        // Binds the validation to the real acoustic-corpus manifest: every
        // scenario carries known provenance, held-out paths are unique per
        // scenario, and scenarios without held-out seats are measured
        // single-position captures (report-only, never training).
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data_tests/roomeq/acoustic_corpus/manifest.json");
        let raw = std::fs::read_to_string(&path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let scenarios = manifest["scenarios"].as_array().unwrap();
        assert!(!scenarios.is_empty(), "corpus manifest has no scenarios");
        for scenario in scenarios {
            let id = scenario["id"].as_str().unwrap_or("");
            assert!(!id.trim().is_empty(), "scenario without id");
            let provenance = scenario["provenance"].as_str().unwrap_or("");
            assert!(
                matches!(provenance, "fem" | "real_measurement" | "synthetic"),
                "scenario '{id}' has unknown provenance '{provenance}'"
            );
            let rate = scenario["sample_rate"].as_f64().unwrap_or(0.0);
            assert!(
                rate.is_finite() && rate > 0.0,
                "scenario '{id}' has no rate"
            );
            let held_out = scenario["held_out"].as_array();
            let mut paths = std::collections::HashSet::new();
            let mut evidence_classes = std::collections::HashSet::new();
            let mut independent_seats = std::collections::HashSet::new();
            let mut independent_response_paths = std::collections::HashSet::new();
            for entry in held_out.into_iter().flatten() {
                let capture = entry["path"].as_str().unwrap_or("");
                assert!(
                    !capture.trim().is_empty(),
                    "scenario '{id}' held-out without path"
                );
                assert!(
                    paths.insert(capture),
                    "scenario '{id}' duplicates held-out '{capture}'"
                );
                let evidence_class = entry
                    .get("evidence_class")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                assert!(
                    matches!(
                        evidence_class,
                        "unknown"
                            | "independent_real_measurement"
                            | "deterministic_perturbation"
                            | "fem_generated"
                            | "synthetic_control"
                    ),
                    "scenario '{id}' has unknown held-out evidence class '{evidence_class}'"
                );
                evidence_classes.insert(evidence_class);
                match evidence_class {
                    "independent_real_measurement" => {
                        assert_eq!(provenance, "real_measurement", "scenario '{id}'");
                        let seat = entry["seat_id"].as_str().unwrap_or("");
                        assert!(
                            !seat.trim().is_empty() && seat.trim() == seat,
                            "scenario '{id}' independent measured row needs an explicit trimmed seat_id"
                        );
                        let channel = entry["channel"].as_str().unwrap_or("");
                        assert!(
                            independent_seats.insert((channel, seat)),
                            "scenario '{id}' repeats independent seat '{seat}' for channel '{channel}'"
                        );
                        assert!(
                            independent_response_paths.insert((channel, capture)),
                            "scenario '{id}' reuses independent response '{capture}' for channel '{channel}'"
                        );
                    }
                    "deterministic_perturbation" => {
                        assert_eq!(provenance, "real_measurement", "scenario '{id}'");
                    }
                    "fem_generated" => assert_eq!(provenance, "fem", "scenario '{id}'"),
                    "synthetic_control" => assert_eq!(provenance, "synthetic", "scenario '{id}'"),
                    "unknown" => {}
                    _ => unreachable!("held-out evidence class was validated above"),
                }
            }
            assert!(
                evidence_classes.len() <= 1,
                "scenario '{id}' mixes held-out evidence classes"
            );
            if paths.is_empty() {
                assert_eq!(
                    provenance, "real_measurement",
                    "scenario '{id}' trains no held-out seat from a non-measured source"
                );
            }
        }
    }

    #[test]
    fn qa_reference_vectors_distinct_from_generated_controls() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("test-vector.json");
        std::fs::write(&path, b"independent test vector").unwrap();
        let hash = sha256_hex(Sha256::digest(b"independent test vector"));
        let independent = vec![ReferenceVector {
            set_id: "test-vectors".to_string(),
            id: "test-vector".to_string(),
            hash: hash.clone(),
            artifact_path: path.clone(),
            edition: "test-edition".to_string(),
            license_id: "test-license".to_string(),
            independent: true,
        }];
        let generated = vec![ReferenceVector {
            set_id: "synthetic-controls".to_string(),
            id: "synthetic-control-1".to_string(),
            hash: sha256_hex(Sha256::digest(b"generated control")),
            artifact_path: temp.path().join("generated-control.json"),
            edition: "synthetic".to_string(),
            license_id: "test-license".to_string(),
            independent: false,
        }];
        assert!(validate_reference_separation(&independent, &generated).is_ok());
        // A shared content hash means the "reference" is a programme copy.
        let mut copied_control = generated.clone();
        copied_control[0].hash = hash.clone();
        let error = validate_reference_separation(&independent, &copied_control).unwrap_err();
        assert!(error.contains("shares content hash"), "{error}");
        // An unflagged vector is not independent evidence.
        let mut unflagged = independent.clone();
        unflagged[0].independent = false;
        assert!(validate_reference_separation(&unflagged, &generated).is_err());
        let mut unlicensed = independent.clone();
        unlicensed[0].license_id.clear();
        assert!(validate_reference_separation(&unlicensed, &generated).is_err());
        let mut uneditioned = independent.clone();
        uneditioned[0].edition.clear();
        assert!(validate_reference_separation(&uneditioned, &generated).is_err());
        let mut mismatched = independent.clone();
        mismatched[0].hash = "0".repeat(64);
        assert!(
            validate_reference_separation(&mismatched, &generated)
                .unwrap_err()
                .contains("hash mismatch")
        );
        let mut missing = independent.clone();
        missing[0].artifact_path = temp.path().join("missing.json");
        assert!(
            validate_reference_separation(&missing, &generated)
                .unwrap_err()
                .contains("cannot be opened")
        );
        let empty_path = temp.path().join("empty.json");
        std::fs::write(&empty_path, b"").unwrap();
        let mut empty = independent.clone();
        empty[0].artifact_path = empty_path;
        empty[0].hash = sha256_hex(Sha256::digest(b""));
        assert!(
            validate_reference_separation(&empty, &generated)
                .unwrap_err()
                .contains("artifact is empty")
        );
        let duplicate = [independent[0].clone(), independent[0].clone()];
        assert!(validate_reference_separation(&duplicate, &generated).is_err());
        let mut mixed_set = independent[0].clone();
        mixed_set.id = "another-vector".to_string();
        mixed_set.set_id = "other-set".to_string();
        assert!(
            validate_reference_separation(&[independent[0].clone(), mixed_set], &generated)
                .unwrap_err()
                .contains("matching set")
        );
        // No references registered means no external validation.
        assert!(validate_reference_separation(&[], &generated).is_err());
    }

    #[test]
    fn qa_comparison_budgets_and_input_hashes_match() {
        let leg = ComparisonBinding {
            baseline_graph_hash: "graph-base".to_string(),
            candidate_graph_hash: "graph-cand".to_string(),
            input_hash: "inputs-1".to_string(),
            seed: 42,
            sample_rate: 48_000.0,
            budget_evals: 600_000,
            limits_version: "limits-v3".to_string(),
        };
        assert!(validate_comparison_pair(&leg, &leg).is_ok());
        let mut drifted = leg.clone();
        drifted.input_hash = "inputs-2".to_string();
        assert!(validate_comparison_pair(&leg, &drifted).is_err());
        let mut cheaper = leg.clone();
        cheaper.budget_evals = 200;
        let error = validate_comparison_pair(&leg, &cheaper).unwrap_err();
        assert!(error.contains("different budgets"), "{error}");
        let mut reseeded = leg.clone();
        reseeded.seed = 7;
        assert!(validate_comparison_pair(&leg, &reseeded).is_err());
        let mut unversioned = leg.clone();
        unversioned.limits_version.clear();
        assert!(validate_comparison_pair(&leg, &unversioned).is_err());
    }
}
