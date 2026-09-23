//! I/O-free measurement descriptors shared by model and loader crates.

use crate::Curve;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Inline measurement data (frequencies, SPL, phase)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InlineMeasurement {
    /// Frequency points in Hz
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frequencies: Vec<f64>,
    /// Sound Pressure Level in dB
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub magnitude_db: Vec<f64>,
    /// Phase in degrees (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_deg: Option<Vec<f64>>,
    /// Optional display name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional path to associated WAV file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wav_path: Option<String>,
    /// Optional path to associated CSV file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csv_path: Option<String>,
}

impl InlineMeasurement {
    pub fn resolve_paths(&mut self, base_dir: &Path) {
        if let Some(csv_path) = &self.csv_path {
            let path = PathBuf::from(csv_path);
            if path.is_relative() {
                self.csv_path = Some(base_dir.join(path).to_string_lossy().into_owned());
            }
        }
        if let Some(wav_path) = &self.wav_path {
            let path = PathBuf::from(wav_path);
            if path.is_relative() {
                self.wav_path = Some(base_dir.join(path).to_string_lossy().into_owned());
            }
        }
    }
}

/// Reference to a measurement file
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum MeasurementRef {
    /// Frozen parsed response plus its original reference, not authenticated acquisition evidence.
    /// Internal workflow handoff; not a public JSON configuration input form.
    #[schemars(skip)]
    Loaded {
        /// Original source metadata, retained without reopening it for numerical loading.
        original: Box<MeasurementRef>,
        /// Complete parsed response on its native grid.
        loaded_response: Box<Curve>,
    },
    /// Inline measurement data (stored directly in JSON)
    Inline(InlineMeasurement),
    /// Named measurement with optional metadata
    Named {
        /// Path to the CSV measurement file.
        path: PathBuf,
        /// Optional display name for the measurement.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Path to CSV file (freq, spl, phase columns)
    Path(PathBuf),
}

impl MeasurementRef {
    /// Return original metadata beneath any loaded-response snapshots.
    pub fn original(&self) -> &Self {
        let mut current = self;
        while let Self::Loaded { original, .. } = current {
            current = original;
        }
        current
    }

    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            Self::Loaded { original, .. } => original.path(),
            Self::Path(path) | Self::Named { path, .. } => Some(path),
            Self::Inline(_) => None,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Loaded { original, .. } => original.name(),
            Self::Path(_) => None,
            Self::Named { name, .. } => name.as_deref(),
            Self::Inline(inline) => inline.name.as_deref(),
        }
    }

    pub fn is_inline(&self) -> bool {
        matches!(self.original(), Self::Inline(_))
    }

    pub fn inline_data(&self) -> Option<&InlineMeasurement> {
        match self.original() {
            Self::Inline(data) => Some(data),
            _ => None,
        }
    }

    pub fn resolve_paths(&mut self, base_dir: &Path) {
        match self {
            Self::Loaded { original, .. } => original.resolve_paths(base_dir),
            Self::Path(path) | Self::Named { path, .. } if path.is_relative() => {
                *path = base_dir.join(&*path);
            }
            Self::Inline(inline) => inline.resolve_paths(base_dir),
            _ => {}
        }
    }
}

/// Declared acquisition kind of one measurement source.
///
/// This is a declaration, not an inference: the loader never guesses a kind
/// from data shape. Undeclared sources stay [`ProvenanceCaptureKind::Unknown`]
/// and unknown never authorizes phase-critical work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceCaptureKind {
    /// Stationary impulse-response capture with a timing reference.
    StationaryIr,
    /// Spatial magnitude capture (including moving-microphone averages)
    /// without a timing reference.
    SpatialMagnitude,
    /// Direct-sound capture.
    DirectSound,
    /// Exported-backend simulation or rendering; not an acoustic recording.
    SimulatedBackend,
    /// Capture kind not stated.
    #[default]
    Unknown,
}

/// Declared acquisition provenance of one measurement source.
///
/// Every field is optional at the schema level so old JSON stays readable;
/// undeclared provenance degrades to unknown, never to an authorization.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MeasurementProvenance {
    /// Declared acquisition kind (default unknown).
    #[serde(default)]
    pub capture_kind: ProvenanceCaptureKind,
    /// Calibration identity applied at acquisition, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_id: Option<String>,
    /// Shared stationary timing-reference identity, if any. Phase-critical
    /// work needs the same identity across coherently combined sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_reference_id: Option<String>,
    /// Whether measured (not nominal) SPL backs this source.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_measured_spl: bool,
    /// Gate-limited valid band in Hz, if a time gate bounds the evidence
    /// (e.g. quasi-anechoic valid band). Absent means the full grid support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_band_hz: Option<[f64; 2]>,
    /// Whether off-axis/angular coverage backs direct-sound detail claims.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub has_direct_angular: bool,
    /// Capture facts and explicit policy for quasi-anechoic assessment.
    /// The legacy angular boolean alone does not authorize detail correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_sound: Option<crate::direct_sound::DirectSoundEvidence>,
    /// Ordered per-device capture facts; absent for legacy acquisition workflows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capture: Option<crate::capture_provenance::CaptureProvenance>,
}

/// Single measurement with metadata
///
/// Custom implementation to support both string path and object with speaker_name
#[derive(Debug, Clone, JsonSchema)]
pub struct MeasurementSingle {
    pub measurement: MeasurementRef,
    pub speaker_name: Option<String>,
    /// Declared acquisition provenance (default unknown when absent).
    pub provenance: MeasurementProvenance,
}

impl Serialize for MeasurementSingle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.speaker_name.is_none() && self.provenance == MeasurementProvenance::default() {
            return self.measurement.serialize(serializer);
        }
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        match &self.measurement {
            MeasurementRef::Loaded {
                original,
                loaded_response,
            } => {
                map.serialize_entry("original", original)?;
                map.serialize_entry("loaded_response", loaded_response)?;
            }
            MeasurementRef::Path(path) => map.serialize_entry("path", path)?,
            MeasurementRef::Named { path, name } => {
                map.serialize_entry("path", path)?;
                if let Some(name) = name {
                    map.serialize_entry("name", name)?;
                }
            }
            MeasurementRef::Inline(inline) => map.serialize_entry("inline", inline)?,
        }
        map.serialize_entry("speaker_name", &self.speaker_name)?;
        if self.provenance != MeasurementProvenance::default() {
            map.serialize_entry("provenance", &self.provenance)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for MeasurementSingle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            original: Option<Box<MeasurementRef>>,
            loaded_response: Option<Box<Curve>>,
            path: Option<PathBuf>,
            name: Option<String>,
            inline: Option<InlineMeasurement>,
            speaker_name: Option<String>,
            #[serde(default)]
            provenance: MeasurementProvenance,
        }

        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(path) = value.as_str() {
            return Ok(Self {
                measurement: MeasurementRef::Path(path.into()),
                speaker_name: None,
                provenance: MeasurementProvenance::default(),
            });
        }
        if let Ok(helper) = serde_json::from_value::<Helper>(value.clone()) {
            if let (Some(original), Some(loaded_response)) =
                (helper.original, helper.loaded_response)
            {
                return Ok(Self {
                    measurement: MeasurementRef::Loaded {
                        original,
                        loaded_response,
                    },
                    speaker_name: helper.speaker_name,
                    provenance: helper.provenance,
                });
            }
            if let Some(inline) = helper.inline {
                return Ok(Self {
                    measurement: MeasurementRef::Inline(inline),
                    speaker_name: helper.speaker_name,
                    provenance: helper.provenance,
                });
            }
            if let Some(path) = helper.path {
                let measurement = match helper.name {
                    Some(name) => MeasurementRef::Named {
                        path,
                        name: Some(name),
                    },
                    None => MeasurementRef::Path(path),
                };
                return Ok(Self {
                    measurement,
                    speaker_name: helper.speaker_name,
                    provenance: helper.provenance,
                });
            }
        }
        let measurement = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            measurement,
            speaker_name: None,
            provenance: MeasurementProvenance::default(),
        })
    }
}

/// Multiple measurements with metadata
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MeasurementMultiple {
    pub measurements: Vec<MeasurementRef>,
    /// Optional speaker name (e.g., "Genelec 8361A")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
    /// Declared acquisition provenance shared by the set (default unknown).
    #[serde(default, skip_serializing_if = "is_default_provenance")]
    pub provenance: MeasurementProvenance,
}

/// True when a provenance declaration carries no information.
fn is_default_provenance(provenance: &MeasurementProvenance) -> bool {
    *provenance == MeasurementProvenance::default()
}

/// Source of measurements (single file, multiple files for averaging, or in-memory curve)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum MeasurementSource {
    /// A single measurement file with optional speaker name
    Single(MeasurementSingle),
    /// Multiple measurement files to be averaged with optional speaker name
    Multiple(MeasurementMultiple),
    /// In-memory curve data (not serializable to JSON config files).
    /// Use this when curves are already loaded in memory.
    #[serde(skip)]
    InMemory(Curve),
    #[serde(skip)]
    /// Multiple in-memory curves (e.g., multi-mic recordings).
    /// Not serializable — use for GPUI in-memory data.
    InMemoryMultiple(Vec<Curve>),
}

impl MeasurementSource {
    pub fn speaker_name(&self) -> Option<&str> {
        match self {
            Self::Single(single) => single.speaker_name.as_deref(),
            Self::Multiple(multiple) => multiple.speaker_name.as_deref(),
            Self::InMemory(_) | Self::InMemoryMultiple(_) => None,
        }
    }

    /// Declared acquisition provenance, if the source carries any.
    ///
    /// In-memory curves carry no declaration and report unknown: the loader
    /// never infers a capture kind from data shape.
    pub fn provenance(&self) -> MeasurementProvenance {
        let (mut provenance, measurement_count) = match self {
            Self::Single(single) => (single.provenance.clone(), 1),
            Self::Multiple(multiple) => (multiple.provenance.clone(), multiple.measurements.len()),
            Self::InMemory(_) | Self::InMemoryMultiple(_) => {
                return MeasurementProvenance::default();
            }
        };
        if provenance
            .timing_reference_id
            .as_deref()
            .is_some_and(|reference| {
                reference.trim().is_empty() || reference.trim().eq_ignore_ascii_case("unknown")
            })
        {
            // Match the coherent-array admission rule: a placeholder is not a
            // shared clock identity. Preserve the original source declaration.
            provenance.timing_reference_id = None;
        }
        if let Some(capture) = &provenance.capture {
            let valid = capture
                .coherent_reference(measurement_count)
                .is_ok_and(|reference| {
                    provenance.timing_reference_id.as_deref() == Some(reference)
                });
            if !valid {
                // Keep the original capture block for diagnostics, but never let
                // a top-level timing label override failed per-device evidence.
                provenance.timing_reference_id = None;
                provenance.capture_kind = ProvenanceCaptureKind::SpatialMagnitude;
            }
        }
        provenance
    }

    /// Associated recording WAV path, when the source carries inline data.
    ///
    /// Multi-measurement sources retain the historical convention of using the
    /// first position's recording for channel-level arrival and SSIR analysis.
    pub fn wav_path(&self) -> Option<&str> {
        let measurement = match self {
            Self::Single(single) => &single.measurement,
            Self::Multiple(multiple) => multiple.measurements.first()?,
            Self::InMemory(_) | Self::InMemoryMultiple(_) => return None,
        };
        measurement.inline_data()?.wav_path.as_deref()
    }

    pub fn resolve_paths(&mut self, base_dir: &Path) {
        match self {
            Self::Single(single) => single.measurement.resolve_paths(base_dir),
            Self::Multiple(multiple) => {
                for measurement in &mut multiple.measurements {
                    measurement.resolve_paths(base_dir);
                }
            }
            Self::InMemory(_) | Self::InMemoryMultiple(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inline(wav_path: Option<&str>) -> MeasurementRef {
        MeasurementRef::Inline(InlineMeasurement {
            frequencies: vec![100.0],
            magnitude_db: vec![80.0],
            phase_deg: None,
            name: None,
            wav_path: wav_path.map(String::from),
            csv_path: None,
        })
    }

    #[test]
    fn loaded_reference_preserves_inline_recording_metadata() {
        let reference = MeasurementRef::Loaded {
            original: Box::new(inline(Some("original-recording.wav"))),
            loaded_response: Box::new(Curve {
                freq: vec![40.0, 80.0].into(),
                spl: vec![80.0, 81.0].into(),
                ..Default::default()
            }),
        };
        assert!(reference.is_inline());
        assert_eq!(
            reference.inline_data().unwrap().wav_path.as_deref(),
            Some("original-recording.wav")
        );
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: reference,
            speaker_name: Some("speaker metadata".into()),
            provenance: MeasurementProvenance::default(),
        });
        let mut decoded: MeasurementSource =
            serde_json::from_value(serde_json::to_value(&source).unwrap()).unwrap();
        assert_eq!(decoded.wav_path(), source.wav_path());
        decoded.resolve_paths(Path::new("/Volumes/home_tmp/tmp"));
        assert_eq!(
            decoded.wav_path(),
            Some("/Volumes/home_tmp/tmp/original-recording.wav")
        );
        let MeasurementSource::Single(single) = decoded else {
            panic!("single snapshot");
        };
        let MeasurementRef::Loaded {
            loaded_response, ..
        } = single.measurement
        else {
            panic!("snapshot retained");
        };
        assert_eq!(loaded_response.spl.to_vec(), vec![80.0, 81.0]);
    }

    #[test]
    fn provenance_defaults_to_unknown_and_old_json_stays_readable() {
        // Bare path JSON predates provenance: it reads with unknown kind.
        let source: MeasurementSource = serde_json::from_str("\"meas.csv\"").unwrap();
        assert_eq!(
            source.provenance().capture_kind,
            ProvenanceCaptureKind::Unknown
        );
        let named: MeasurementSource =
            serde_json::from_str("{\"path\": \"meas.csv\", \"speaker_name\": \"s\"}").unwrap();
        assert_eq!(
            named.provenance().capture_kind,
            ProvenanceCaptureKind::Unknown
        );
    }

    #[test]
    fn provenance_round_trips_and_in_memory_stays_unknown() {
        let source = MeasurementSource::Single(MeasurementSingle {
            measurement: MeasurementRef::Path(PathBuf::from("meas.csv")),
            speaker_name: None,
            provenance: MeasurementProvenance {
                capture_kind: ProvenanceCaptureKind::StationaryIr,
                calibration_id: Some(String::from("spl-cal-94db")),
                timing_reference_id: Some(String::from("loopback-1")),
                has_measured_spl: true,
                valid_band_hz: Some([50.0, 8000.0]),
                has_direct_angular: false,
                direct_sound: None,
                capture: None,
            },
        });
        let json = serde_json::to_string(&source).unwrap();
        let back: MeasurementSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provenance(), source.provenance());
        // In-memory curves carry no declaration: never inferred.
        assert_eq!(
            MeasurementSource::InMemory(Curve::default()).provenance(),
            MeasurementProvenance::default()
        );
    }

    #[test]
    fn unknown_timing_labels_do_not_authorize_capture_provenance() {
        for label in ["", " \t", "unknown", " UnKnOwN "] {
            let source: MeasurementSource = serde_json::from_value(serde_json::json!({
                "path": "not-loaded.csv",
                "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": label}
            }))
            .unwrap();
            assert_eq!(source.provenance().timing_reference_id, None, "{label:?}");
            // Admission sanitization must not rewrite the retained declaration.
            assert_eq!(
                serde_json::to_value(&source).unwrap()["provenance"]["timing_reference_id"],
                label
            );
        }
        let source: MeasurementSource = serde_json::from_value(serde_json::json!({
            "path": "not-loaded.csv",
            "provenance": {"capture_kind": "stationary_ir", "timing_reference_id": "clock-a"}
        }))
        .unwrap();
        assert_eq!(
            source.provenance().timing_reference_id.as_deref(),
            Some("clock-a")
        );
    }

    #[test]
    fn measurement_source_wav_path_uses_single_or_first_position() {
        let single = MeasurementSource::Single(MeasurementSingle {
            measurement: inline(Some("single.wav")),
            speaker_name: None,
            provenance: MeasurementProvenance::default(),
        });
        assert_eq!(single.wav_path(), Some("single.wav"));

        let multiple = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![inline(Some("first.wav")), inline(Some("second.wav"))],
            speaker_name: None,
            provenance: MeasurementProvenance::default(),
        });
        assert_eq!(multiple.wav_path(), Some("first.wav"));
        assert!(
            MeasurementSource::InMemory(Curve::default())
                .wav_path()
                .is_none()
        );
    }
}

/// I/O-free CEA2034 / Spinorama curve bundle.
#[derive(Debug, Clone)]
pub struct SpinoramaBundle {
    pub on_axis: Curve,
    pub listening_window: Curve,
    pub early_reflections: Curve,
    pub sound_power: Curve,
    pub estimated_in_room: Curve,
    pub er_di: Curve,
    pub sp_di: Curve,
    pub curves: HashMap<String, Curve>,
}

impl SpinoramaBundle {
    pub fn pir(&self) -> &Curve {
        &self.estimated_in_room
    }
}
