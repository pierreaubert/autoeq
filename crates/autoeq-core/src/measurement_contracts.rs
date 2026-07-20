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
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            Self::Path(path) | Self::Named { path, .. } => Some(path),
            Self::Inline(_) => None,
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Path(_) => None,
            Self::Named { name, .. } => name.as_deref(),
            Self::Inline(inline) => inline.name.as_deref(),
        }
    }

    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }

    pub fn inline_data(&self) -> Option<&InlineMeasurement> {
        match self {
            Self::Inline(data) => Some(data),
            _ => None,
        }
    }

    pub fn resolve_paths(&mut self, base_dir: &Path) {
        match self {
            Self::Path(path) | Self::Named { path, .. } if path.is_relative() => {
                *path = base_dir.join(&*path);
            }
            Self::Inline(inline) => inline.resolve_paths(base_dir),
            _ => {}
        }
    }
}

/// Single measurement with metadata
///
/// Custom implementation to support both string path and object with speaker_name
#[derive(Debug, Clone, JsonSchema)]
pub struct MeasurementSingle {
    pub measurement: MeasurementRef,
    pub speaker_name: Option<String>,
}

impl Serialize for MeasurementSingle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.speaker_name.is_none() {
            return self.measurement.serialize(serializer);
        }
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        match &self.measurement {
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
            path: Option<PathBuf>,
            name: Option<String>,
            inline: Option<InlineMeasurement>,
            speaker_name: Option<String>,
        }

        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(path) = value.as_str() {
            return Ok(Self {
                measurement: MeasurementRef::Path(path.into()),
                speaker_name: None,
            });
        }
        if let Ok(helper) = serde_json::from_value::<Helper>(value.clone()) {
            if let Some(inline) = helper.inline {
                return Ok(Self {
                    measurement: MeasurementRef::Inline(inline),
                    speaker_name: helper.speaker_name,
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
                });
            }
        }
        let measurement = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            measurement,
            speaker_name: None,
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
    fn measurement_source_wav_path_uses_single_or_first_position() {
        let single = MeasurementSource::Single(MeasurementSingle {
            measurement: inline(Some("single.wav")),
            speaker_name: None,
        });
        assert_eq!(single.wav_path(), Some("single.wav"));

        let multiple = MeasurementSource::Multiple(MeasurementMultiple {
            measurements: vec![inline(Some("first.wav")), inline(Some("second.wav"))],
            speaker_name: None,
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
