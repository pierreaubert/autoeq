//! Product-oriented speaker and headphone preparation with separate source,
//! target, measurement-rig, and device-profile declarations.

use crate::{Curve, OptimParams, read};
use autoeq_measurements::{
    MeasurementOrigin, MeasurementRecord, MeasurementRigIdentity, ValidationMode,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Top-level product workflow selector. Room correction is owned by the
/// RoomEQ command and is rejected here with an explicit delegation message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProductMode {
    Speaker,
    Headphone,
    Room,
}

/// Source selection for a product workflow. API identities are explicit and
/// remain separate from locally supplied measurement-rig declarations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProductSource {
    SpeakerApi {
        speaker: String,
        version: String,
        measurement: String,
        #[serde(default = "default_curve_name")]
        curve_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        measurement_rig: Option<MeasurementRigIdentity>,
    },
    HeadphoneApi {
        headphone: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        measurement_rig: Option<MeasurementRigIdentity>,
    },
    Csv {
        path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        measurement_rig: Option<MeasurementRigIdentity>,
    },
}

/// Target curve source and caller-declared target-domain support. The target
/// record remains distinct from the measured source record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProductTarget {
    Named {
        name: String,
        #[serde(default)]
        supported_measurement_rigs: Vec<MeasurementRigIdentity>,
    },
    Csv {
        path: PathBuf,
        #[serde(default)]
        supported_measurement_rigs: Vec<MeasurementRigIdentity>,
    },
}

/// Explicit renderer selection. Only `EqualizerApo` currently has a checked
/// product export path. Other existing writers remain available through the
/// legacy CLI flow, but this facade refuses to imply their device guarantees.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductRenderer {
    EqualizerApo,
    RmeTotalMix,
    AppleAu,
}

/// Schema version for the machine-readable product renderer capability report.
pub const PRODUCT_RENDERER_CAPABILITIES_SCHEMA_VERSION: u32 = 3;

const PRODUCT_PAIR_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Frozen Equalizer APO source revision used to verify the profiled shelf
/// coefficient mapping. This is a source-derived check, not a runtime claim.
pub const PROFILED_APO_SHELF_SOURCE_REVISION: &str = "bbfcc3e5024cbb9d61ba75fc88d78605cc4c9687";
/// Conservative minimum consumer version assumed for the documented LSC/HSC behavior.
pub const PROFILED_APO_SHELF_MINIMUM_VERSION_ASSUMPTION: &str = "1.2.1";
/// Slope used by the only shelf mapping verified for profiled output.
pub const PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE: u8 = 12;
/// Multiplier for the normalized source/core coefficient comparison bound.
pub const PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER: u32 = 16;
/// Maximum source/core sampled transfer difference for the shelf mapping.
pub const PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB: f64 = 1.0e-10;

/// Format emitted by the corresponding legacy or checked product serializer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductRendererFormat {
    EqualizerApoTextPreset,
    RmeTotalMixRoomEqXml,
    AppleAudioUnitPreset,
}

/// Whether a renderer has a product export path with profile-bound checks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductProfileExportStatus {
    /// The product workflow validates the caller's profile and records the
    /// exact serialized preset and realized filter parameters.
    Verified,
    /// A legacy serializer exists, but the product workflow cannot make a
    /// versioned device-compatibility claim for it.
    LegacyOnly,
}

/// Stable machine-readable reason why a renderer is unavailable to profiled
/// product export.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductRendererRefusalCode {
    /// No versioned consumer/device contract is available for the legacy
    /// serializer, so its output cannot be bound to a caller's device profile.
    NoVerifiedVersionedDeviceContract,
}

impl ProductRendererRefusalCode {
    /// Return the human-readable explanation shared by API validation and CLI
    /// diagnostics.
    pub fn message(self) -> &'static str {
        match self {
            Self::NoVerifiedVersionedDeviceContract => {
                "has no verified profile export path; use the legacy exporter without device-guarantee claims"
            }
        }
    }
}

/// Checks that the profiled product exporter verifies for a renderer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductRendererVerifiedFeature {
    ExplicitPerMachineDeviceProfile,
    SourceTargetAndRigProvenanceSidecar,
    PresetSha256Binding,
    QuantizedFilterTransferCheck,
    StrictEmittedTextRoundTripCheck,
    SourceDerivedApoShelfCoefficientAndTransferCheck,
}

/// Source and numeric contract for profiled Equalizer APO shelf serialization.
/// The local verifier does not parse with or execute Equalizer APO itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProfiledApoShelfContract {
    pub emitted_filter_types: Vec<String>,
    pub slope_db_per_octave: u8,
    pub frequency_convention: String,
    pub q_encoded: bool,
    pub equalizer_apo_source_revision: String,
    pub minimum_consumer_version_assumption: String,
    pub coefficient_comparison: String,
    pub max_scaled_coefficient_delta_epsilon: u32,
    pub max_sampled_transfer_delta_db: f64,
    pub consumer_runtime_checked: bool,
}

/// Return the exact shelf contract advertised by the profile verifier.
pub fn profiled_apo_shelf_contract() -> ProfiledApoShelfContract {
    ProfiledApoShelfContract {
        emitted_filter_types: vec!["LSC".into(), "HSC".into()],
        slope_db_per_octave: PROFILED_APO_SHELF_SLOPE_DB_PER_OCTAVE,
        frequency_convention: "center_frequency_fc".into(),
        q_encoded: false,
        equalizer_apo_source_revision: PROFILED_APO_SHELF_SOURCE_REVISION.into(),
        minimum_consumer_version_assumption: PROFILED_APO_SHELF_MINIMUM_VERSION_ASSUMPTION.into(),
        coefficient_comparison:
            "max_abs_delta <= 16*f64::EPSILON*max(1,max_abs(source_coefficients,core_coefficients))"
                .into(),
        max_scaled_coefficient_delta_epsilon: PROFILED_APO_SHELF_COEFFICIENT_EPSILON_MULTIPLIER,
        max_sampled_transfer_delta_db: PROFILED_APO_SHELF_MAX_TRANSFER_DELTA_DB,
        consumer_runtime_checked: false,
    }
}

/// Known behavior that limits what a renderer capability report promises.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProductRendererLimitation {
    /// Writing a preset does not install it or query the running application.
    PresetRequiresUserInstallation,
    /// The APO check validates serialization and declared limits, not playback
    /// hardware behavior or audibility.
    RuntimeDeviceAndAudibilityAreNotVerified,
    /// Profiled APO output refuses filter syntax outside its checked subset.
    ProfiledApoRefusesUnverifiedFilterSyntax,
    /// APO text does not bind a device or channel; both are inherited from the
    /// including Equalizer APO configuration.
    ProfiledApoRoutingIsInheritedFromIncludingConfiguration,
    /// The legacy RME writer can reshape or replace filters to fit its fixed
    /// topology and supported filter slots.
    LegacyRmeWriterMayTransformFilterTopology,
    /// The legacy RME XML writer emits at most nine filters for each channel.
    LegacyRmeWriterCapsAtNineFiltersPerChannel,
    /// The legacy RME XML writer uses fixed zero channel gain and delay fields.
    LegacyRmeWriterUsesZeroChannelGainAndDelay,
    /// The legacy Apple Audio Unit writer emits at most sixteen bands.
    LegacyAppleWriterCapsAtSixteenBands,
    /// The legacy Apple Audio Unit writer stores filter values as f32.
    LegacyAppleWriterUsesSinglePrecisionParameters,
    /// The legacy Apple Audio Unit preset has no sample-rate binding.
    LegacyAppleWriterHasNoSampleRateBinding,
}

/// Capability for one renderer in the product-profile export workflow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProductRendererCapability {
    pub renderer: ProductRenderer,
    pub format: ProductRendererFormat,
    pub legacy_export_available: bool,
    pub product_profile_export: ProductProfileExportStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal_code: Option<ProductRendererRefusalCode>,
    pub verified_features: Vec<ProductRendererVerifiedFeature>,
    pub known_limitations: Vec<ProductRendererLimitation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profiled_apo_shelf_contract: Option<ProfiledApoShelfContract>,
}

/// Versioned machine-readable capabilities for product-profile exports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProductRendererCapabilities {
    pub schema_version: u32,
    pub renderers: Vec<ProductRendererCapability>,
}

impl ProductRenderer {
    /// Describe the current product-profile and legacy export contracts.
    ///
    /// A `Verified` product export means the software checks the selected
    /// serializer against an explicit caller-supplied profile and binds its
    /// provenance. It does not mean the preset is installed or verified on a
    /// running physical device.
    pub fn capability(self) -> ProductRendererCapability {
        match self {
            Self::EqualizerApo => ProductRendererCapability {
                renderer: self,
                format: ProductRendererFormat::EqualizerApoTextPreset,
                legacy_export_available: true,
                product_profile_export: ProductProfileExportStatus::Verified,
                refusal_code: None,
                verified_features: vec![
                    ProductRendererVerifiedFeature::ExplicitPerMachineDeviceProfile,
                    ProductRendererVerifiedFeature::SourceTargetAndRigProvenanceSidecar,
                    ProductRendererVerifiedFeature::PresetSha256Binding,
                    ProductRendererVerifiedFeature::QuantizedFilterTransferCheck,
                    ProductRendererVerifiedFeature::StrictEmittedTextRoundTripCheck,
                    ProductRendererVerifiedFeature::SourceDerivedApoShelfCoefficientAndTransferCheck,
                ],
                known_limitations: vec![
                    ProductRendererLimitation::PresetRequiresUserInstallation,
                    ProductRendererLimitation::RuntimeDeviceAndAudibilityAreNotVerified,
                    ProductRendererLimitation::ProfiledApoRefusesUnverifiedFilterSyntax,
                    ProductRendererLimitation::ProfiledApoRoutingIsInheritedFromIncludingConfiguration,
                ],
                profiled_apo_shelf_contract: Some(profiled_apo_shelf_contract()),
            },
            Self::RmeTotalMix => ProductRendererCapability {
                renderer: self,
                format: ProductRendererFormat::RmeTotalMixRoomEqXml,
                legacy_export_available: true,
                product_profile_export: ProductProfileExportStatus::LegacyOnly,
                refusal_code: Some(ProductRendererRefusalCode::NoVerifiedVersionedDeviceContract),
                verified_features: vec![],
                known_limitations: vec![
                    ProductRendererLimitation::LegacyRmeWriterMayTransformFilterTopology,
                    ProductRendererLimitation::LegacyRmeWriterCapsAtNineFiltersPerChannel,
                    ProductRendererLimitation::LegacyRmeWriterUsesZeroChannelGainAndDelay,
                ],
                profiled_apo_shelf_contract: None,
            },
            Self::AppleAu => ProductRendererCapability {
                renderer: self,
                format: ProductRendererFormat::AppleAudioUnitPreset,
                legacy_export_available: true,
                product_profile_export: ProductProfileExportStatus::LegacyOnly,
                refusal_code: Some(ProductRendererRefusalCode::NoVerifiedVersionedDeviceContract),
                verified_features: vec![],
                known_limitations: vec![
                    ProductRendererLimitation::LegacyAppleWriterCapsAtSixteenBands,
                    ProductRendererLimitation::LegacyAppleWriterUsesSinglePrecisionParameters,
                    ProductRendererLimitation::LegacyAppleWriterHasNoSampleRateBinding,
                ],
                profiled_apo_shelf_contract: None,
            },
        }
    }
}

/// Return the current versioned product renderer capability report.
pub fn product_renderer_capabilities() -> ProductRendererCapabilities {
    ProductRendererCapabilities {
        schema_version: PRODUCT_RENDERER_CAPABILITIES_SCHEMA_VERSION,
        renderers: [
            ProductRenderer::EqualizerApo,
            ProductRenderer::RmeTotalMix,
            ProductRenderer::AppleAu,
        ]
        .into_iter()
        .map(ProductRenderer::capability)
        .collect(),
    }
}

/// Numeric interval declared by a caller-provided playback-device profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeviceRange {
    pub minimum: f64,
    pub maximum: f64,
}

impl DeviceRange {
    fn validate(self, label: &str) -> Result<(), String> {
        if !self.minimum.is_finite() || !self.maximum.is_finite() {
            return Err(format!("device {label} bounds must be finite"));
        }
        if self.minimum > self.maximum {
            return Err(format!("device {label} minimum exceeds its maximum"));
        }
        Ok(())
    }

    fn contains(self, range: DeviceRange) -> bool {
        range.minimum >= self.minimum && range.maximum <= self.maximum
    }
}

/// Caller-provided device settings. These values vary by machine; there is no
/// inferred hardware profile or assumed playback chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    pub id: String,
    /// Explicit playback endpoint identifier for the current machine.
    pub playback_device_id: String,
    pub renderer: ProductRenderer,
    pub sample_rate_hz: f64,
    pub maximum_filter_count: usize,
    /// PEQ model names use the stable `PeqModel::to_string()` spelling.
    pub supported_peq_models: Vec<String>,
    /// Renderer short names for filter types, e.g. `PK`, `LS`, or `HP`.
    pub supported_filter_types: Vec<String>,
    pub frequency_hz: DeviceRange,
    pub q: DeviceRange,
    pub gain_db: DeviceRange,
    /// Explicit APO preamp value. The checked writer serializes this at 0.1 dB.
    pub preamp_db: Option<f64>,
}

/// Return every filter type the optimizer can produce for this topology.
/// Free topologies can choose any encoded biquad type, so their profile must
/// explicitly support all of those types before the search is allowed to run.
fn required_filter_types(model: crate::PeqModel) -> Vec<&'static str> {
    match model {
        crate::PeqModel::Pk => vec!["PK"],
        crate::PeqModel::HpPk => vec!["HP", "PK"],
        crate::PeqModel::HpPkLp => vec!["HP", "PK", "LP"],
        crate::PeqModel::LsPk => vec!["LS", "PK"],
        crate::PeqModel::LsPkHs => vec!["LS", "PK", "HS"],
        crate::PeqModel::PkLsHs => vec!["PK", "LS", "HS"],
        crate::PeqModel::FreePkFree | crate::PeqModel::Free => vec![
            "PK", "LP", "HP", "LS", "HS", "HPQ", "BP", "NO", "AP", "LSO", "HSO", "PKM",
        ],
    }
}

fn required_profiled_apo_filter_types(
    model: crate::PeqModel,
    num_filters: usize,
    minimum_q: f64,
    maximum_q: f64,
) -> Vec<&'static str> {
    let mut types = match model {
        crate::PeqModel::Pk => vec!["PK"],
        crate::PeqModel::HpPk => {
            if num_filters <= 1 {
                vec!["HPQ"]
            } else {
                vec!["HPQ", "PK"]
            }
        }
        crate::PeqModel::HpPkLp => {
            if num_filters <= 1 {
                vec!["HPQ"]
            } else {
                let mut result = vec!["HPQ", "LPQ"];
                if num_filters > 2 {
                    result.push("PK");
                }
                // setup_bounds constrains HpPkLp's last Q to
                // [max(1, min_q.max(0.1)).min(max_q),
                //  max(1.5, lower).min(max_q)]. Use that actual interval: the
                // ordinary global Q range often includes the default while
                // the low-pass Q interval does not.
                let q_lower = minimum_q.max(0.1);
                let lowpass_q_min = 1.0_f64.max(q_lower).min(maximum_q);
                let lowpass_q_max = 1.5_f64.max(lowpass_q_min).min(maximum_q);
                if lowpass_q_min <= crate::iir::DEFAULT_Q_HIGH_LOW_PASS
                    && crate::iir::DEFAULT_Q_HIGH_LOW_PASS <= lowpass_q_max
                {
                    result.push("LP");
                }
                result
            }
        }
        crate::PeqModel::LsPk => {
            if num_filters <= 1 {
                vec!["LSC"]
            } else {
                vec!["LSC", "PK"]
            }
        }
        crate::PeqModel::LsPkHs => match num_filters {
            0 | 1 => vec!["LSC"],
            2 => vec!["LSC", "HSC"],
            _ => vec!["LSC", "PK", "HSC"],
        },
        crate::PeqModel::PkLsHs => match num_filters {
            0 | 1 => vec!["HSC"],
            2 => vec!["LSC", "HSC"],
            _ => vec!["PK", "LSC", "HSC"],
        },
        crate::PeqModel::FreePkFree | crate::PeqModel::Free => vec![
            "PK", "LP", "LPQ", "HP", "HPQ", "LSC", "HSC", "BP", "NO", "AP", "LSO", "HSO", "PKM",
        ],
    };
    types.sort_unstable();
    types.dedup();
    types
}

fn profiled_apo_filter_kind(filter: &crate::iir::Biquad) -> Result<&'static str, String> {
    let kind = match filter.filter_type {
        crate::iir::BiquadFilterType::Peak => "PK",
        crate::iir::BiquadFilterType::Lowpass => {
            if (filter.q - crate::iir::DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                "LP"
            } else {
                "LPQ"
            }
        }
        crate::iir::BiquadFilterType::Highpass => {
            if (filter.q - crate::iir::DEFAULT_Q_HIGH_LOW_PASS).abs() < f64::EPSILON {
                "HP"
            } else {
                "HPQ"
            }
        }
        crate::iir::BiquadFilterType::HighpassVariableQ => "HPQ",
        crate::iir::BiquadFilterType::Lowshelf => "LSC",
        crate::iir::BiquadFilterType::Highshelf => "HSC",
        crate::iir::BiquadFilterType::AllPass => "AP",
        crate::iir::BiquadFilterType::Bandpass
        | crate::iir::BiquadFilterType::Notch
        | crate::iir::BiquadFilterType::LowshelfOrf
        | crate::iir::BiquadFilterType::HighshelfOrf
        | crate::iir::BiquadFilterType::PeakMatched => {
            return Err(format!(
                "profiled Equalizer APO export refuses unverified filter type '{}'",
                filter.filter_type.short_name()
            ));
        }
    };
    if !matches!(kind, "PK" | "LSC" | "HSC") && filter.db_gain != 0.0 {
        return Err(format!(
            "profiled Equalizer APO {kind} output does not encode filter gain"
        ));
    }
    Ok(kind)
}

fn is_supported_profiled_apo_filter_kind(kind: &str) -> bool {
    matches!(
        kind,
        "PK" | "LP" | "LPQ" | "HP" | "HPQ" | "LSC" | "HSC" | "AP"
    )
}

fn profiled_apo_filter_refusal(kind: &str) -> String {
    if matches!(kind, "LS" | "HS") {
        format!(
            "profiled Equalizer APO shelf output uses {kind} only in the legacy path; profiles must declare LSC/HSC with a 12 dB slope"
        )
    } else {
        format!("profiled Equalizer APO export refuses unverified filter type '{kind}'")
    }
}

fn validate_declared_tokens(
    values: &[String],
    label: &str,
    allowed: &[&str],
) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!(
            "device profile must declare at least one supported {label}"
        ));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() || value.trim() != value {
            return Err(format!(
                "device profile {label} names must be non-empty and trimmed"
            ));
        }
        if !allowed.contains(&value.as_str()) {
            return Err(format!("device profile declares unknown {label} '{value}'"));
        }
        if !seen.insert(value.as_str()) {
            return Err(format!("device profile repeats {label} '{value}'"));
        }
    }
    Ok(())
}

impl DeviceProfile {
    /// Reject invalid or unsupported device settings before an optimizer runs.
    pub fn validate_for_optimizer(&self, params: &OptimParams) -> Result<(), String> {
        if self.id.trim().is_empty() || self.id.trim() != self.id {
            return Err("device profile id must be non-empty and trimmed".into());
        }
        if self.playback_device_id.trim().is_empty()
            || self.playback_device_id.trim() != self.playback_device_id
        {
            return Err("device playback identifier must be non-empty and trimmed".into());
        }
        for (label, range) in [
            ("frequency", self.frequency_hz),
            ("Q", self.q),
            ("gain", self.gain_db),
        ] {
            range.validate(label)?;
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err("device sample rate must be finite and positive".into());
        }
        if self.maximum_filter_count == 0 {
            return Err("device profile must allow at least one filter".into());
        }
        if !params.sample_rate.is_finite() || params.sample_rate <= 0.0 {
            return Err("optimizer sample rate must be finite and positive".into());
        }
        if (self.sample_rate_hz - params.sample_rate).abs()
            > self.sample_rate_hz.abs().max(params.sample_rate.abs()) * 1e-12
        {
            return Err(format!(
                "device profile sample rate {} Hz does not match optimizer sample rate {} Hz",
                self.sample_rate_hz, params.sample_rate
            ));
        }
        if !params.min_freq.is_finite()
            || !params.max_freq.is_finite()
            || params.min_freq <= 0.0
            || params.max_freq < params.min_freq
            || params.max_freq >= self.sample_rate_hz / 2.0
        {
            return Err(
                "optimizer frequency bounds must be finite, ordered, positive, and below Nyquist"
                    .into(),
            );
        }
        if self.frequency_hz.minimum <= 0.0
            || self.frequency_hz.maximum >= self.sample_rate_hz / 2.0
            || self.q.minimum <= 0.0
        {
            return Err(
                "device frequency range must be positive and below Nyquist, and Q must be positive"
                    .into(),
            );
        }
        validate_declared_tokens(
            &self.supported_peq_models,
            "PEQ model",
            &[
                "pk",
                "hp-pk",
                "hp-pk-lp",
                "ls-pk",
                "ls-pk-hs",
                "pk-ls-hs",
                "free-pk-free",
                "free",
            ],
        )?;
        validate_declared_tokens(
            &self.supported_filter_types,
            "filter type",
            &[
                "PK", "LP", "LPQ", "HP", "HS", "HPQ", "LS", "BP", "NO", "AP", "LSO", "HSO", "PKM",
                "LSC", "HSC",
            ],
        )?;
        if params.num_filters > self.maximum_filter_count {
            return Err(format!(
                "optimizer requests {} filters but device profile '{}' allows at most {}",
                params.num_filters, self.id, self.maximum_filter_count
            ));
        }
        if !self
            .supported_peq_models
            .iter()
            .any(|model| model == &params.peq_model.to_string())
        {
            return Err(format!(
                "device profile '{}' does not declare PEQ model '{}'",
                self.id, params.peq_model
            ));
        }
        let required_types = if self.renderer == ProductRenderer::EqualizerApo {
            required_profiled_apo_filter_types(
                params.peq_model,
                params.num_filters,
                params.min_q,
                params.max_q,
            )
        } else {
            required_filter_types(params.peq_model)
        };
        if self.renderer == ProductRenderer::EqualizerApo
            && let Some(unsupported) = required_types
                .iter()
                .find(|filter_type| !is_supported_profiled_apo_filter_kind(filter_type))
        {
            return Err(profiled_apo_filter_refusal(unsupported));
        }
        if let Some(unsupported) = required_types.iter().find(|filter_type| {
            !self
                .supported_filter_types
                .iter()
                .any(|v| v == **filter_type)
        }) {
            return Err(format!(
                "device profile '{}' does not declare filter type '{}' required by PEQ model '{}'",
                self.id, unsupported, params.peq_model
            ));
        }
        if !self.frequency_hz.contains(DeviceRange {
            minimum: params.min_freq,
            maximum: params.max_freq,
        }) {
            return Err("optimizer frequency bounds exceed the device profile".into());
        }
        if !self.q.contains(DeviceRange {
            minimum: params.min_q,
            maximum: params.max_q,
        }) {
            return Err("optimizer Q bounds exceed the device profile".into());
        }
        if !self.gain_db.contains(DeviceRange {
            minimum: params.min_db,
            maximum: params.max_db,
        }) {
            return Err("optimizer gain bounds exceed the device profile".into());
        }
        if self.preamp_db.is_none_or(|value| !value.is_finite()) {
            return Err("device profile requires an explicit finite preamp value".into());
        }
        if self.renderer == ProductRenderer::EqualizerApo
            && self.preamp_db.is_some_and(|value| value > 0.0)
        {
            return Err(
                "profiled Equalizer APO export requires a non-positive preamp value".into(),
            );
        }
        if let Some(reason) = self.renderer.capability().refusal_code {
            return Err(format!("renderer {:?} {}", self.renderer, reason.message()));
        }
        Ok(())
    }

    /// Validate an actual designed filter list against declared device limits.
    pub fn validate_filters(
        &self,
        sample_rate_hz: f64,
        filters: &[crate::iir::Biquad],
    ) -> Result<(), String> {
        if !self.sample_rate_hz.is_finite()
            || self.sample_rate_hz <= 0.0
            || !sample_rate_hz.is_finite()
            || sample_rate_hz <= 0.0
        {
            return Err("filter and device sample rates must be finite and positive".into());
        }
        if (sample_rate_hz - self.sample_rate_hz).abs()
            > self.sample_rate_hz.abs().max(sample_rate_hz.abs()) * 1e-12
        {
            return Err("designed filters use a sample rate outside the device profile".into());
        }
        if filters.len() > self.maximum_filter_count {
            return Err(format!(
                "designed filter count {} exceeds device limit {}",
                filters.len(),
                self.maximum_filter_count
            ));
        }
        for (index, filter) in filters.iter().enumerate() {
            let q = filter.q;
            let gain = filter.db_gain;
            if !filter.srate.is_finite()
                || (filter.srate - sample_rate_hz).abs()
                    > sample_rate_hz.abs().max(filter.srate.abs()) * 1e-12
            {
                return Err(format!(
                    "designed filter {} uses a sample rate outside the device profile",
                    index + 1
                ));
            }
            if !filter.freq.is_finite()
                || filter.freq <= 0.0
                || filter.freq >= sample_rate_hz / 2.0
                || !q.is_finite()
                || q <= 0.0
                || !gain.is_finite()
                || !self.frequency_hz.contains(DeviceRange {
                    minimum: filter.freq,
                    maximum: filter.freq,
                })
                || !self.q.contains(DeviceRange {
                    minimum: q,
                    maximum: q,
                })
                || !self.gain_db.contains(DeviceRange {
                    minimum: gain,
                    maximum: gain,
                })
            {
                return Err(format!(
                    "designed filter {} falls outside device limits before serialization",
                    index + 1
                ));
            }
            let filter_type = if self.renderer == ProductRenderer::EqualizerApo {
                profiled_apo_filter_kind(filter)?
            } else {
                filter.filter_type.short_name()
            };
            if !self
                .supported_filter_types
                .iter()
                .any(|supported| supported == filter_type)
            {
                return Err(format!(
                    "device profile '{}' does not declare filter type '{}', used by filter {}",
                    self.id,
                    filter_type,
                    index + 1
                ));
            }
        }
        Ok(())
    }

    /// Return the actual numeric parameters represented by the current APO
    /// text writer and reject any filter that leaves the profile after its
    /// frequency, Q, or gain is serialized.
    pub fn apo_serialized_filters(
        &self,
        sample_rate_hz: f64,
        filters: &[crate::iir::Biquad],
    ) -> Result<Vec<crate::iir::Biquad>, String> {
        if self.renderer != ProductRenderer::EqualizerApo {
            return Err(
                "serialized filter validation is only implemented for Equalizer APO".into(),
            );
        }
        self.validate_filters(sample_rate_hz, filters)?;
        let mut serialized = Vec::with_capacity(filters.len());
        for filter in filters {
            if !filter.freq.is_finite()
                || filter.freq < 0.0
                || filter.freq.round() > i32::MAX as f64
            {
                return Err(
                    "APO center frequency cannot be represented as an integer Hz value".into(),
                );
            }
            // APO rounds center frequency to integer Hz and formats Q/gain
            // with two decimals. Rebuild coefficients from those realized
            // values so callers never evaluate stale coefficients.
            let freq = filter.freq.round();
            let q: f64 = format!("{:.2}", filter.q)
                .parse()
                .map_err(|_| "APO Q serialization was not numeric")?;
            let gain: f64 = format!("{:+.2}", filter.db_gain)
                .parse()
                .map_err(|_| "APO gain serialization was not numeric")?;
            serialized.push(crate::iir::Biquad::new(
                filter.filter_type,
                freq,
                sample_rate_hz,
                q,
                gain,
            ));
        }
        self.validate_filters(sample_rate_hz, &serialized)?;
        Ok(serialized)
    }

    /// The APO file has one decimal place of preamp precision. Returning the
    /// rounded value makes the emitted profile auditable against what is saved.
    pub fn apo_serialized_preamp_db(&self) -> Result<f64, String> {
        if self.renderer != ProductRenderer::EqualizerApo {
            return Err("preamp serialization is only implemented for Equalizer APO".into());
        }
        let preamp = self
            .preamp_db
            .filter(|value| value.is_finite())
            .ok_or_else(|| "device profile requires an explicit finite preamp value".to_string())?;
        if preamp > 0.0 {
            return Err(
                "profiled Equalizer APO export requires a non-positive preamp value".into(),
            );
        }
        let serialized = format!("{preamp:.1}")
            .parse()
            .map_err(|_| "APO preamp serialization was not numeric".to_string())?;
        if serialized > 0.0 {
            return Err("serialized Equalizer APO preamp must be non-positive".into());
        }
        Ok(serialized)
    }
}

/// Compatibility is a declaration assessment only; it does not prove
/// calibration, reference approval, or physical equivalence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TargetCompatibilityStatus {
    DeclaredMatch,
    DeclaredMismatch,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetCompatibility {
    pub status: TargetCompatibilityStatus,
    pub measurement_rig: Option<MeasurementRigIdentity>,
    pub supported_measurement_rigs: Vec<MeasurementRigIdentity>,
    pub explanation: String,
}

/// Target data and its independent source lineage plus declared rig support.
#[derive(Debug, Clone)]
pub struct TargetProfile {
    pub record: MeasurementRecord,
    pub supported_measurement_rigs: Vec<MeasurementRigIdentity>,
}

impl TargetProfile {
    pub fn new(
        record: MeasurementRecord,
        supported_measurement_rigs: Vec<MeasurementRigIdentity>,
    ) -> Result<Self, String> {
        let validation = record.validate(ValidationMode::Warn);
        if !validation.is_valid() {
            return Err(format!(
                "invalid target measurement record: {}",
                validation.errors.join("; ")
            ));
        }
        let mut seen = BTreeSet::new();
        for rig in &supported_measurement_rigs {
            rig.validate().map_err(|error| error.to_string())?;
            if !seen.insert(rig) {
                return Err(format!(
                    "duplicate target-supported measurement rig {:?}:{}:{}",
                    rig.kind, rig.domain, rig.id
                ));
            }
        }
        Ok(Self {
            record,
            supported_measurement_rigs,
        })
    }

    pub fn assess_compatibility(&self, measurement: &MeasurementRecord) -> TargetCompatibility {
        let measurement_rig = measurement.provenance.measurement_rig.clone();
        let status = match (&measurement_rig, self.supported_measurement_rigs.is_empty()) {
            (Some(rig), false) if self.supported_measurement_rigs.contains(rig) => {
                TargetCompatibilityStatus::DeclaredMatch
            }
            (Some(_), false) => TargetCompatibilityStatus::DeclaredMismatch,
            _ => TargetCompatibilityStatus::Unknown,
        };
        let explanation = match status {
            TargetCompatibilityStatus::DeclaredMatch => {
                "measurement rig identity appears in the target's caller-declared support list; this does not verify calibration or physical equivalence".into()
            }
            TargetCompatibilityStatus::DeclaredMismatch => {
                "measurement rig identity is absent from the target's caller-declared support list".into()
            }
            TargetCompatibilityStatus::Unknown => {
                "measurement rig or target support declaration is missing; compatibility is unknown".into()
            }
        };
        TargetCompatibility {
            status,
            measurement_rig,
            supported_measurement_rigs: self.supported_measurement_rigs.clone(),
            explanation,
        }
    }
}

/// Complete product-mode request. Room requests are deliberately delegated to
/// the existing RoomEQ workflow instead of being interpreted here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProductRequest {
    pub mode: ProductMode,
    pub source: ProductSource,
    pub target: ProductTarget,
    pub device_profile: DeviceProfile,
    /// Refuse a declared mismatch by default; setting this false is an
    /// explicit caller decision to continue with the recorded mismatch.
    #[serde(default = "default_reject_target_mismatch")]
    pub reject_declared_target_mismatch: bool,
}

/// Return the lowercase SHA-256 digest of product output bytes.
pub fn product_bytes_sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Verify that an APO provenance sidecar is bound to these exact preset bytes.
/// Call this before trusting the sidecar's device or realized-filter metadata.
/// The check validates content identity; it does not make two-file publication
/// power-loss atomic.
///
/// # Errors
///
/// Returns an error when the sidecar is malformed, lacks a preset digest, or
/// is bound to different preset bytes.
pub fn verify_apo_preset_binding(preset_bytes: &[u8], sidecar_bytes: &[u8]) -> Result<(), String> {
    let sidecar: serde_json::Value = serde_json::from_slice(sidecar_bytes)
        .map_err(|error| format!("invalid product provenance sidecar: {error}"))?;
    let recorded_hash = sidecar
        .get("apo_serialization")
        .and_then(|serialization| serialization.get("preset_sha256"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "product sidecar has no APO preset SHA-256 binding".to_owned())?;
    let actual_hash = product_bytes_sha256_hex(preset_bytes);
    if recorded_hash != actual_hash {
        return Err("product sidecar SHA-256 does not match the APO preset bytes".into());
    }
    Ok(())
}

/// Publish a verified APO preset and its provenance sidecar as one product operation.
///
/// Both files are staged and checked before either destination changes. If a
/// returned write error follows a successful first replacement, the previous
/// file is restored only while the destination still contains transaction
/// bytes. The two renames are not power-loss atomic as a pair.
///
/// # Errors
///
/// Returns an error when paths are invalid, staging or verification fails, an
/// output exceeds the rollback limit, or publication/rollback encounters a
/// filesystem error or concurrent replacement.
pub fn publish_apo_preset_pair<V>(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    verify_staged_preset: V,
) -> io::Result<()>
where
    V: FnOnce(&[u8]) -> io::Result<()>,
{
    publish_apo_preset_pair_with_hook(
        preset_path,
        preset_bytes,
        sidecar_path,
        sidecar_bytes,
        verify_staged_preset,
        write_product_output,
    )
}

fn publish_apo_preset_pair_with_hook<V, F>(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    verify_staged_preset: V,
    publish_sidecar: F,
) -> io::Result<()>
where
    V: FnOnce(&[u8]) -> io::Result<()>,
    F: FnOnce(&Path, &[u8]) -> io::Result<()>,
{
    let parent = output_parent(preset_path);
    let sidecar_parent = output_parent(sidecar_path);
    let canonical_parent = fs::canonicalize(parent)?;
    if canonical_parent != fs::canonicalize(sidecar_parent)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "profiled APO preset and sidecar must share a directory",
        ));
    }
    let preset_name = preset_path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "APO preset path has no file name",
        )
    })?;
    let sidecar_name = sidecar_path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "APO sidecar path has no file name",
        )
    })?;
    if canonical_parent.join(preset_name) == canonical_parent.join(sidecar_name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "profiled APO preset and sidecar must be distinct files",
        ));
    }

    let staging = tempfile::Builder::new()
        .prefix(".autoeq-product-pair-")
        .tempdir_in(parent)?;
    let staged_preset = staging.path().join("preset.txt");
    let staged_sidecar = staging.path().join("provenance.json");
    write_staged_product_file(&staged_preset, preset_bytes)?;
    write_staged_product_file(&staged_sidecar, sidecar_bytes)?;
    let staged_preset_bytes = read_product_file_bounded(&staged_preset)?;
    let staged_sidecar_bytes = read_product_file_bounded(&staged_sidecar)?;
    verify_staged_preset(&staged_preset_bytes)?;
    verify_apo_preset_binding(&staged_preset_bytes, &staged_sidecar_bytes)
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;

    let previous_preset = read_existing_product_file(preset_path)?;
    let previous_sidecar = read_existing_product_file(sidecar_path)?;
    if let Err(publish_error) = write_product_output(preset_path, &staged_preset_bytes) {
        return Err(rollback_product_publication_error(
            "APO preset",
            publish_error,
            restore_product_file_if_unchanged(
                preset_path,
                previous_preset.as_deref(),
                &staged_preset_bytes,
            ),
        ));
    }

    let sidecar_publish = publish_sidecar(sidecar_path, &staged_sidecar_bytes);
    if let Err(publish_error) = sidecar_publish {
        let sidecar_rollback = restore_product_file_if_unchanged(
            sidecar_path,
            previous_sidecar.as_deref(),
            &staged_sidecar_bytes,
        );
        if let Err(rollback_error) = sidecar_rollback {
            return Err(io::Error::other(format!(
                "failed to publish product provenance sidecar ({publish_error}); sidecar changed concurrently or could not be restored ({rollback_error}); APO preset was left untouched"
            )));
        }
        if let Err(rollback_error) = restore_product_file_if_unchanged(
            preset_path,
            previous_preset.as_deref(),
            &staged_preset_bytes,
        ) {
            return Err(io::Error::other(format!(
                "failed to publish product provenance sidecar ({publish_error}); failed to restore the prior APO preset ({rollback_error})"
            )));
        }
        return Err(io::Error::other(format!(
            "failed to publish product provenance sidecar; prior APO preset pair was restored: {publish_error}"
        )));
    }
    Ok(())
}

fn rollback_product_publication_error(
    label: &str,
    publish_error: io::Error,
    rollback_result: io::Result<()>,
) -> io::Error {
    match rollback_result {
        Ok(()) => io::Error::new(
            publish_error.kind(),
            format!("failed to publish {label}; prior file was restored: {publish_error}"),
        ),
        Err(rollback_error) => io::Error::other(format!(
            "failed to publish {label} ({publish_error}); failed to restore the prior file ({rollback_error})"
        )),
    }
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn write_staged_product_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if bytes.len() as u64 > PRODUCT_PAIR_MAX_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "staged product output exceeds the 16 MiB limit",
        ));
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn read_product_file_bounded(path: &Path) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut bytes = Vec::new();
    file.take(PRODUCT_PAIR_MAX_FILE_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > PRODUCT_PAIR_MAX_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "product output exceeds the 16 MiB limit",
        ));
    }
    Ok(bytes)
}

fn read_existing_product_file(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match read_product_file_bounded(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn write_product_output(path: &Path, bytes: &[u8]) -> io::Result<()> {
    super::atomic_file::replace_atomically(path, |file| file.write_all(bytes))
}

fn restore_product_file_if_unchanged(
    path: &Path,
    previous: Option<&[u8]>,
    transaction_bytes: &[u8],
) -> io::Result<()> {
    let current = read_existing_product_file(path)?;
    if current.as_deref() == previous {
        return Ok(());
    }
    if current.as_deref() != Some(transaction_bytes) {
        return Err(io::Error::other(format!(
            "{} changed concurrently; rollback left the newer file untouched",
            path.display()
        )));
    }
    match previous {
        Some(previous) => write_product_output(path, previous),
        None => {
            fs::remove_file(path)?;
            super::atomic_file::sync_parent_directory(path)
        }
    }
}

/// Adapter bundle used to keep API/cache sources replaceable in tests and
/// offline workflows.
pub struct ProductSourceAdapters<'a> {
    pub cache_root: &'a Path,
    pub backend: &'a dyn read::MeasurementBackend,
    pub cache: &'a dyn read::MeasurementCache,
}

/// Loaded records and curve collections ready for normalization and objective
/// construction. Source and target provenance remain independent.
#[derive(Debug, Clone)]
pub struct PreparedProduct {
    pub mode: ProductMode,
    pub source_record: MeasurementRecord,
    pub target_profile: TargetProfile,
    pub target_compatibility: TargetCompatibility,
    pub spin_curves: Option<HashMap<String, Curve>>,
}

impl PreparedProduct {
    /// Load a product request through caller-supplied source/cache adapters.
    pub async fn load(
        request: &ProductRequest,
        params: &OptimParams,
        adapters: &ProductSourceAdapters<'_>,
    ) -> Result<Self, Box<dyn Error>> {
        if request.mode == ProductMode::Room {
            return Err("Room correction is handled by the existing RoomEQ CLI/workflow".into());
        }
        request
            .device_profile
            .validate_for_optimizer(params)
            .map_err(std::io::Error::other)?;
        validate_source_mode(request.mode, &request.source)?;
        let mode_matches_loss = match request.mode {
            ProductMode::Speaker => matches!(
                params.loss,
                crate::LossType::SpeakerFlat
                    | crate::LossType::SpeakerFlatAsymmetric
                    | crate::LossType::SpeakerScore
                    | crate::LossType::Epa
            ),
            ProductMode::Headphone => matches!(
                params.loss,
                crate::LossType::HeadphoneFlat | crate::LossType::HeadphoneScore
            ),
            ProductMode::Room => false,
        };
        if !mode_matches_loss {
            return Err(format!(
                "product mode {:?} does not match optimizer loss {:?}",
                request.mode, params.loss
            )
            .into());
        }

        let (mut source_record, spin_curves) =
            load_product_source(&request.source, adapters).await?;
        let declared_rig = match &request.source {
            ProductSource::Csv {
                measurement_rig, ..
            }
            | ProductSource::SpeakerApi {
                measurement_rig, ..
            }
            | ProductSource::HeadphoneApi {
                measurement_rig, ..
            } => measurement_rig,
        };
        if let Some(measurement_rig) = declared_rig {
            source_record.provenance.measurement_rig = Some(measurement_rig.clone());
        }
        if let Some(rig) = &source_record.provenance.measurement_rig {
            rig.validate()?;
        }
        let target_profile = load_product_target(&request.target, &source_record.curve)?;
        let target_compatibility = target_profile.assess_compatibility(&source_record);
        if request.reject_declared_target_mismatch
            && target_compatibility.status == TargetCompatibilityStatus::DeclaredMismatch
        {
            return Err(
                format!("target rig mismatch: {}", target_compatibility.explanation).into(),
            );
        }
        Ok(Self {
            mode: request.mode,
            source_record,
            target_profile,
            target_compatibility,
            spin_curves,
        })
    }

    /// Normalize source and target onto one validated grid for objective
    /// construction while keeping the raw source/target records available.
    pub fn curves_for_optimizer(
        &self,
        params: &OptimParams,
    ) -> Result<ProductCurves, Box<dyn Error>> {
        let points = if self.mode == ProductMode::Headphone {
            120
        } else {
            200
        };
        let grid = crate::workflow::VisualizationGridConfig {
            points,
            min_freq: Some(params.min_freq),
            max_freq: Some(params.max_freq),
        }
        .create_frequency_grid(params)?;
        // Product mode refuses extrapolation rather than treating a missing
        // end of a measurement or target as measured support.
        ensure_curve_covers_band(
            &self.source_record.curve,
            "measurement source",
            params.min_freq,
            params.max_freq,
        )?;
        ensure_curve_covers_band(
            &self.target_profile.record.curve,
            "target source",
            params.min_freq,
            params.max_freq,
        )?;
        if let Some(curves) = &self.spin_curves {
            for (name, curve) in curves {
                ensure_curve_covers_band(
                    curve,
                    &format!("speaker scoring curve '{name}'"),
                    params.min_freq,
                    params.max_freq,
                )?;
            }
        }
        let (input_curve, target_curve, deviation_curve) = normalize_product_curves(self, &grid);
        let spin_curves = self.spin_curves.as_ref().map(|curves| {
            curves
                .iter()
                .map(|(name, curve)| (name.clone(), read::interpolate_log_space(&grid, curve)))
                .collect()
        });
        Ok(ProductCurves {
            standard_freq: grid,
            input_curve,
            target_curve,
            deviation_curve,
            spin_curves,
        })
    }
}

/// Curves prepared from a [`PreparedProduct`] for use by the optimizer.
#[derive(Debug, Clone)]
pub struct ProductCurves {
    pub standard_freq: ndarray::Array1<f64>,
    pub input_curve: Curve,
    pub target_curve: Curve,
    pub deviation_curve: Curve,
    pub spin_curves: Option<HashMap<String, Curve>>,
}

fn validate_source_mode(mode: ProductMode, source: &ProductSource) -> Result<(), Box<dyn Error>> {
    if mode == ProductMode::Room {
        return Err("Room correction is handled by the existing RoomEQ CLI/workflow".into());
    }
    let source_identities: Vec<&str> = match source {
        ProductSource::SpeakerApi {
            speaker,
            version,
            measurement,
            curve_name,
            ..
        } => vec![speaker, version, measurement, curve_name],
        ProductSource::HeadphoneApi { headphone, .. } => vec![headphone],
        ProductSource::Csv { path, .. } => {
            if path.as_os_str().is_empty() {
                return Err("CSV source path must not be empty".into());
            }
            Vec::new()
        }
    };
    if source_identities
        .iter()
        .any(|identity| identity.trim().is_empty() || identity.trim() != *identity)
    {
        return Err("API source identifiers must be non-empty and trimmed".into());
    }
    match (mode, source) {
        (ProductMode::Speaker, ProductSource::HeadphoneApi { .. })
        | (ProductMode::Headphone, ProductSource::SpeakerApi { .. }) => {
            Err("product mode does not match the selected API source".into())
        }
        (_, ProductSource::Csv { .. })
        | (ProductMode::Speaker, ProductSource::SpeakerApi { .. })
        | (ProductMode::Headphone, ProductSource::HeadphoneApi { .. }) => Ok(()),
        (ProductMode::Room, _) => {
            Err("Room correction is handled by the existing RoomEQ CLI/workflow".into())
        }
    }
}

async fn load_product_source(
    source: &ProductSource,
    adapters: &ProductSourceAdapters<'_>,
) -> Result<(MeasurementRecord, Option<HashMap<String, Curve>>), Box<dyn Error>> {
    match source {
        ProductSource::Csv { path, .. } => {
            let record = read::read_record_from_csv(path)?;
            Ok((record, None))
        }
        ProductSource::HeadphoneApi { headphone, .. } => {
            let (_cache_path, curve) =
                read::fetch_headphone_frequency_response_with_backend_at_cache_root(
                    headphone,
                    adapters.cache_root,
                    adapters.backend,
                    adapters.cache,
                )
                .await?;
            let uri = format!(
                "https://api.spinorama.org/v1/headphone/{}/frequency_response",
                urlencoding::encode(headphone)
            );
            let mut record = MeasurementRecord::from_api(curve, uri)?;
            record.id = format!(
                "spinorama:headphone:{headphone}:{}",
                record.provenance.content_hash
            );
            record.provenance.source_id = Some(format!("spinorama:headphone:{headphone}"));
            Ok((record, None))
        }
        ProductSource::SpeakerApi {
            speaker,
            version,
            measurement,
            curve_name,
            ..
        } => {
            let fetched_measurement = if measurement == "Estimated In-Room Response" {
                "CEA2034"
            } else {
                measurement
            };
            let plot = read::fetch_measurement_plot_data_with_backend_at_cache_root(
                speaker,
                version,
                fetched_measurement,
                adapters.cache_root,
                adapters.backend,
                adapters.cache,
            )
            .await?;
            let extracted_spin = if fetched_measurement == "CEA2034" {
                Some(read::extract_cea2034_curves_original(&plot, "CEA2034")?)
            } else {
                None
            };
            let selected_name = if measurement == "Estimated In-Room Response" {
                "Estimated In-Room Response"
            } else {
                curve_name
            };
            let curve = if let Some(curves) = &extracted_spin {
                curves
                    .get(selected_name)
                    .cloned()
                    .ok_or_else(|| format!("curve '{selected_name}' is absent from CEA2034 data"))?
            } else {
                read::extract_curve_by_name(&plot, fetched_measurement, selected_name)?
            };
            let mut record = MeasurementRecord::from_api(
                curve,
                format!(
                    "https://api.spinorama.org/v1/speaker/{}/version/{}/measurements/{}?measurement_format=json#{}",
                    urlencoding::encode(speaker),
                    urlencoding::encode(version),
                    urlencoding::encode(fetched_measurement),
                    urlencoding::encode(selected_name),
                ),
            )?;
            record.id = format!(
                "spinorama:speaker:{speaker}:{version}:{measurement}:{}",
                record.provenance.content_hash
            );
            record.provenance.source_id = Some(format!(
                "spinorama:speaker:{speaker}:{version}:{measurement}"
            ));
            Ok((record, extracted_spin))
        }
    }
}

fn load_product_target(
    target: &ProductTarget,
    input: &Curve,
) -> Result<TargetProfile, Box<dyn Error>> {
    match target {
        ProductTarget::Csv {
            path,
            supported_measurement_rigs,
        } => {
            let record = read::read_record_from_csv(path)?;
            TargetProfile::new(record, supported_measurement_rigs.clone())
                .map_err(|error| error.into())
        }
        ProductTarget::Named {
            name,
            supported_measurement_rigs,
        } => {
            if name.trim().is_empty() {
                return Err("named target must not be empty".into());
            }
            let curve = crate::workflow::build_target_curve_by_name(name, &input.freq, input)?;
            let mut record = MeasurementRecord::legacy(curve)?;
            record.id = format!(
                "named-target:{}:{}",
                name.trim(),
                record.provenance.content_hash
            );
            record.provenance.origin = MeasurementOrigin::Synthetic;
            record.provenance.source_id = Some(format!("named-target:{}", name.trim()));
            TargetProfile::new(record, supported_measurement_rigs.clone())
                .map_err(|error| error.into())
        }
    }
}

fn default_curve_name() -> String {
    "Listening Window".into()
}

fn default_reject_target_mismatch() -> bool {
    true
}

fn ensure_curve_covers_band(
    curve: &Curve,
    label: &str,
    minimum_hz: f64,
    maximum_hz: f64,
) -> Result<(), Box<dyn Error>> {
    curve.validate(label)?;
    let first = curve.freq[0];
    let last = *curve
        .freq
        .last()
        .expect("validated curve has frequency points");
    if first > minimum_hz || last < maximum_hz {
        return Err(format!(
            "{label} covers {first:.3}..{last:.3} Hz but optimization requests {minimum_hz:.3}..{maximum_hz:.3} Hz; product mode refuses frequency extrapolation"
        )
        .into());
    }
    Ok(())
}

/// Keep source and target records intact while constructing a grid-specific
/// target curve for an optimizer request.
pub fn normalize_product_curves(
    prepared: &PreparedProduct,
    grid: &ndarray::Array1<f64>,
) -> (Curve, Curve, Curve) {
    let input = read::normalize_and_interpolate_response(grid, &prepared.source_record.curve);
    let target =
        read::normalize_and_interpolate_response(grid, &prepared.target_profile.record.curve);
    let deviation = Curve {
        freq: grid.clone(),
        spl: &target.spl - &input.spl,
        phase: None,
        ..Default::default()
    };
    (input, target, deviation)
}

/// Convenience: return the CSV target path when a caller needs to report its
/// source separately from the measurement path.
pub fn product_target_path(target: &ProductTarget) -> Option<&Path> {
    match target {
        ProductTarget::Csv { path, .. } => Some(path),
        ProductTarget::Named { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_measurements::{
        MeasurementRigIdentity, MeasurementRigKind,
        read::{InMemoryMeasurementCache, MockMeasurementBackend},
    };
    use ndarray::Array1;

    #[test]
    fn apo_sidecar_binding_verifier_detects_corrupted_and_unbound_presets() {
        let preset = b"GraphicEQ: 100 1\n";
        let sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(preset)
            }
        }))
        .unwrap();

        verify_apo_preset_binding(preset, &sidecar).unwrap();
        assert!(verify_apo_preset_binding(b"GraphicEQ: 100 2\n", &sidecar).is_err());
        assert!(verify_apo_preset_binding(preset, b"{}").is_err());
    }

    #[test]
    fn verified_product_pair_restores_both_files_after_second_publish_failure() {
        let directory = tempfile::tempdir().unwrap();
        let preset_path = directory.path().join("preset.txt");
        let sidecar_path = directory.path().join("preset.provenance.json");
        let old_preset = b"old preset\n";
        let old_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(old_preset)
            }
        }))
        .unwrap();
        fs::write(&preset_path, old_preset).unwrap();
        fs::write(&sidecar_path, &old_sidecar).unwrap();

        let new_preset = b"new preset\n";
        let new_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(new_preset)
            }
        }))
        .unwrap();
        let error = publish_apo_preset_pair_with_hook(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            |_| Ok(()),
            |path, bytes| {
                write_product_output(path, bytes)?;
                Err(io::Error::other("injected post-install durability error"))
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prior APO preset pair was restored")
        );
        assert_eq!(fs::read(&preset_path).unwrap(), old_preset);
        assert_eq!(fs::read(&sidecar_path).unwrap(), old_sidecar);
        verify_apo_preset_binding(
            &fs::read(&preset_path).unwrap(),
            &fs::read(&sidecar_path).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn product_pair_rollback_does_not_overwrite_concurrent_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let preset_path = directory.path().join("preset.txt");
        let sidecar_path = directory.path().join("preset.provenance.json");
        let old_preset = b"old preset\n";
        let old_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(old_preset)
            }
        }))
        .unwrap();
        fs::write(&preset_path, old_preset).unwrap();
        fs::write(&sidecar_path, &old_sidecar).unwrap();

        let new_preset = b"new preset\n";
        let new_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(new_preset)
            }
        }))
        .unwrap();
        let concurrent_preset = b"concurrent preset\n";
        let error = publish_apo_preset_pair_with_hook(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            |_| Ok(()),
            |_, _| {
                fs::write(&preset_path, concurrent_preset)?;
                Err(io::Error::other("injected sidecar write failure"))
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("changed concurrently"));
        assert_eq!(fs::read(&preset_path).unwrap(), concurrent_preset);
        assert_eq!(fs::read(&sidecar_path).unwrap(), old_sidecar);
    }

    #[test]
    fn product_pair_refuses_aliases_and_preserves_prior_outputs_on_verification_error() {
        let directory = tempfile::tempdir().unwrap();
        let preset_path = directory.path().join("preset.txt");
        let old_preset = b"old preset\n";
        fs::write(&preset_path, old_preset).unwrap();
        let old_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(old_preset)
            }
        }))
        .unwrap();
        let sidecar_path = directory.path().join("preset.provenance.json");
        fs::write(&sidecar_path, &old_sidecar).unwrap();

        let new_preset = b"new preset\n";
        let new_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": product_bytes_sha256_hex(new_preset)
            }
        }))
        .unwrap();
        let error = publish_apo_preset_pair(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            |_| Err(io::Error::other("invalid preset")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("invalid preset"));
        assert_eq!(fs::read(&preset_path).unwrap(), old_preset);
        assert_eq!(fs::read(&sidecar_path).unwrap(), old_sidecar);

        let alias = directory.path().join("./preset.txt");
        let error =
            publish_apo_preset_pair(&preset_path, new_preset, &alias, &new_sidecar, |_| Ok(()))
                .unwrap_err();
        assert!(error.to_string().contains("must be distinct files"));
        assert_eq!(fs::read(&preset_path).unwrap(), old_preset);
    }

    fn rig(kind: MeasurementRigKind, domain: &str, id: &str) -> MeasurementRigIdentity {
        MeasurementRigIdentity::new(kind, domain, id).unwrap()
    }

    fn device_profile(id: &str, sample_rate_hz: f64) -> DeviceProfile {
        DeviceProfile {
            id: id.into(),
            playback_device_id: format!("{id}-playback"),
            renderer: ProductRenderer::EqualizerApo,
            sample_rate_hz,
            maximum_filter_count: 10,
            supported_peq_models: vec!["pk".into()],
            supported_filter_types: vec!["PK".into()],
            frequency_hz: DeviceRange {
                minimum: 20.0,
                maximum: 20_000.0,
            },
            q: DeviceRange {
                minimum: 0.5,
                maximum: 10.0,
            },
            gain_db: DeviceRange {
                minimum: -12.0,
                maximum: 12.0,
            },
            preamp_db: Some(-3.5),
        }
    }

    fn optimizer_params() -> OptimParams {
        OptimParams::from(&crate::cli::Args::speaker_defaults())
    }

    fn headphone_optimizer_params() -> OptimParams {
        OptimParams::from(&crate::cli::Args::headphone_defaults())
    }

    fn curve(spl: Vec<f64>) -> Curve {
        Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1_000.0, 20_000.0]),
            spl: Array1::from_vec(spl),
            ..Default::default()
        }
    }

    #[test]
    fn target_compatibility_keeps_match_mismatch_and_unknown_distinct() {
        let acoustic = rig(
            MeasurementRigKind::AcousticMeasurement,
            "lab-chain",
            "mic-a",
        );
        let coupler = rig(
            MeasurementRigKind::HeadphoneCoupler,
            "coupler-db",
            "fixture-x",
        );
        let target = TargetProfile::new(
            MeasurementRecord::legacy(curve(vec![0.0; 4])).unwrap(),
            vec![acoustic.clone()],
        )
        .unwrap();

        let mut matching = MeasurementRecord::legacy(curve(vec![1.0; 4])).unwrap();
        matching.provenance.measurement_rig = Some(acoustic);
        assert_eq!(
            target.assess_compatibility(&matching).status,
            TargetCompatibilityStatus::DeclaredMatch
        );

        let mut mismatch = MeasurementRecord::legacy(curve(vec![1.0; 4])).unwrap();
        mismatch.provenance.measurement_rig = Some(coupler);
        assert_eq!(
            target.assess_compatibility(&mismatch).status,
            TargetCompatibilityStatus::DeclaredMismatch
        );

        let unknown = MeasurementRecord::legacy(curve(vec![1.0; 4])).unwrap();
        assert_eq!(
            target.assess_compatibility(&unknown).status,
            TargetCompatibilityStatus::Unknown
        );
        assert!(
            TargetProfile::new(
                MeasurementRecord::legacy(curve(vec![0.0; 4])).unwrap(),
                vec![],
            )
            .unwrap()
            .assess_compatibility(&matching)
            .status
                == TargetCompatibilityStatus::Unknown
        );
    }

    #[test]
    fn device_profiles_are_explicit_per_machine_and_gate_optimizer_settings() {
        let params = optimizer_params();
        let studio_profile: DeviceProfile = serde_json::from_str(include_str!(
            "../../tests/fixtures/device_profile_studio_48k.json"
        ))
        .unwrap();
        let portable_profile: DeviceProfile = serde_json::from_str(include_str!(
            "../../tests/fixtures/device_profile_portable_96k.json"
        ))
        .unwrap();
        assert_ne!(
            studio_profile.playback_device_id,
            portable_profile.playback_device_id
        );
        assert!(studio_profile.validate_for_optimizer(&params).is_ok());
        assert!(portable_profile.validate_for_optimizer(&params).is_err());

        for renderer in [ProductRenderer::RmeTotalMix, ProductRenderer::AppleAu] {
            let mut unverified_renderer = studio_profile.clone();
            unverified_renderer.renderer = renderer;
            let refusal = unverified_renderer
                .validate_for_optimizer(&params)
                .expect_err("legacy serializer has no verified product profile contract");
            let reason = renderer
                .capability()
                .refusal_code
                .expect("unsupported product renderer should have a reason code");
            assert_eq!(
                refusal,
                format!("renderer {:?} {}", renderer, reason.message())
            );
        }

        let mut implicit_preamp = studio_profile;
        implicit_preamp.preamp_db = None;
        assert!(implicit_preamp.validate_for_optimizer(&params).is_err());
    }

    #[test]
    fn renderer_capability_report_is_versioned_and_exposes_loss_reasons() {
        let report = product_renderer_capabilities();
        assert_eq!(
            report.schema_version,
            PRODUCT_RENDERER_CAPABILITIES_SCHEMA_VERSION
        );
        let apo = ProductRenderer::EqualizerApo.capability();
        let shelf_contract = apo
            .profiled_apo_shelf_contract
            .expect("profiled APO capability exposes the checked shelf mapping");
        assert_eq!(shelf_contract.emitted_filter_types, ["LSC", "HSC"]);
        assert_eq!(shelf_contract.slope_db_per_octave, 12);
        assert_eq!(shelf_contract.frequency_convention, "center_frequency_fc");
        assert!(!shelf_contract.q_encoded);
        assert_eq!(
            shelf_contract.equalizer_apo_source_revision,
            PROFILED_APO_SHELF_SOURCE_REVISION
        );
        assert_eq!(shelf_contract.max_scaled_coefficient_delta_epsilon, 16);
        assert_eq!(shelf_contract.max_sampled_transfer_delta_db, 1.0e-10);
        assert!(!shelf_contract.consumer_runtime_checked);
        assert!(apo.verified_features.contains(
            &ProductRendererVerifiedFeature::SourceDerivedApoShelfCoefficientAndTransferCheck
        ));
        assert_eq!(report.renderers.len(), 3);

        let apo = ProductRenderer::EqualizerApo.capability();
        assert_eq!(
            apo.product_profile_export,
            ProductProfileExportStatus::Verified
        );
        assert_eq!(apo.refusal_code, None);
        assert!(apo.legacy_export_available);
        assert!(
            apo.verified_features
                .contains(&ProductRendererVerifiedFeature::SourceTargetAndRigProvenanceSidecar)
        );
        assert!(
            apo.verified_features
                .contains(&ProductRendererVerifiedFeature::QuantizedFilterTransferCheck)
        );
        assert!(
            apo.verified_features
                .contains(&ProductRendererVerifiedFeature::StrictEmittedTextRoundTripCheck)
        );
        assert!(apo.verified_features.contains(
            &ProductRendererVerifiedFeature::SourceDerivedApoShelfCoefficientAndTransferCheck
        ));
        assert!(
            apo.known_limitations
                .contains(&ProductRendererLimitation::RuntimeDeviceAndAudibilityAreNotVerified)
        );
        assert!(apo.known_limitations.contains(
            &ProductRendererLimitation::ProfiledApoRoutingIsInheritedFromIncludingConfiguration
        ));
        let shelf_contract = apo.profiled_apo_shelf_contract.unwrap();
        assert_eq!(shelf_contract.emitted_filter_types, ["LSC", "HSC"]);
        assert!(!shelf_contract.consumer_runtime_checked);

        let rme = ProductRenderer::RmeTotalMix.capability();
        assert_eq!(
            rme.product_profile_export,
            ProductProfileExportStatus::LegacyOnly
        );
        assert_eq!(
            rme.refusal_code,
            Some(ProductRendererRefusalCode::NoVerifiedVersionedDeviceContract)
        );
        assert!(rme.verified_features.is_empty());
        assert!(rme.legacy_export_available);
        assert!(
            rme.known_limitations
                .contains(&ProductRendererLimitation::LegacyRmeWriterCapsAtNineFiltersPerChannel)
        );

        let apple = ProductRenderer::AppleAu.capability();
        assert_eq!(
            apple.product_profile_export,
            ProductProfileExportStatus::LegacyOnly
        );
        assert_eq!(
            apple.refusal_code,
            Some(ProductRendererRefusalCode::NoVerifiedVersionedDeviceContract)
        );
        assert!(apple.verified_features.is_empty());
        assert!(
            apple.known_limitations.contains(
                &ProductRendererLimitation::LegacyAppleWriterUsesSinglePrecisionParameters
            )
        );
        assert!(
            apple
                .known_limitations
                .contains(&ProductRendererLimitation::LegacyAppleWriterHasNoSampleRateBinding)
        );

        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(
            encoded["schema_version"],
            PRODUCT_RENDERER_CAPABILITIES_SCHEMA_VERSION
        );
        assert_eq!(encoded["renderers"][0]["renderer"], "equalizer_apo");
        assert_eq!(encoded["renderers"][1]["renderer"], "rme_total_mix");
        assert_eq!(encoded["renderers"][2]["renderer"], "apple_au");
        assert_eq!(
            encoded["renderers"][1]["refusal_code"],
            "no_verified_versioned_device_contract"
        );
    }

    #[test]
    fn device_profile_rejects_invalid_type_declarations_and_nyquist_bounds() {
        let params = optimizer_params();
        let mut duplicate_type = device_profile("studio", 48_000.0);
        duplicate_type.supported_filter_types.push("PK".into());
        assert!(
            duplicate_type
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("repeats filter type")
        );

        let mut unknown_model = device_profile("studio", 48_000.0);
        unknown_model.supported_peq_models.push("pk typo".into());
        assert!(
            unknown_model
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("unknown PEQ model")
        );

        let mut above_nyquist = params;
        above_nyquist.max_freq = 24_000.0;
        assert!(
            device_profile("studio", 48_000.0)
                .validate_for_optimizer(&above_nyquist)
                .unwrap_err()
                .contains("below Nyquist")
        );
    }

    #[test]
    fn apo_quantization_is_checked_against_device_ranges() {
        let profile = device_profile("studio", 48_000.0);
        let filter = crate::iir::Biquad::new(
            crate::iir::BiquadFilterType::Peak,
            1_000.49,
            48_000.0,
            1.234,
            2.345,
        );
        let serialized = profile.apo_serialized_filters(48_000.0, &[filter]).unwrap();
        assert_eq!(serialized[0].freq, 1_000.0);
        assert_eq!(serialized[0].q, 1.23);
        assert_eq!(serialized[0].db_gain, 2.35);
        assert_eq!(profile.apo_serialized_preamp_db().unwrap(), -3.5);

        let mut narrow = profile;
        narrow.frequency_hz.minimum = 1_000.25;
        let edge = crate::iir::Biquad::new(
            crate::iir::BiquadFilterType::Peak,
            1_000.26,
            48_000.0,
            1.0,
            0.0,
        );
        assert!(narrow.apo_serialized_filters(48_000.0, &[edge]).is_err());
    }

    #[test]
    fn free_peq_models_require_every_optimizer_filter_type_in_profile() {
        let mut params = optimizer_params();
        params.peq_model = crate::PeqModel::Free;
        let mut profile = device_profile("studio", 48_000.0);
        profile.supported_peq_models.push("free".into());
        assert!(
            profile
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("filter type")
        );
    }

    #[test]
    fn profiled_apo_refuses_unsupported_types_and_positive_preamp_before_search() {
        let params = optimizer_params();
        let mut positive_preamp = device_profile("studio", 48_000.0);
        positive_preamp.preamp_db = Some(1.0);
        assert!(
            positive_preamp
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("non-positive preamp")
        );

        let mut free_params = params.clone();
        free_params.peq_model = crate::PeqModel::Free;
        let mut free_profile = device_profile("studio", 48_000.0);
        free_profile.supported_peq_models.push("free".into());
        free_profile.supported_filter_types.extend(
            [
                "LPQ", "HP", "HPQ", "LS", "HS", "LSC", "HSC", "BP", "NO", "AP", "LSO", "HSO", "PKM",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        let refusal = free_profile
            .validate_for_optimizer(&free_params)
            .expect_err("free APO profiles must refuse types outside the checked subset");
        assert!(refusal.contains("unverified filter type"), "{refusal}");

        let mut profile = device_profile("studio", 48_000.0);
        profile.supported_filter_types.push("NO".into());
        let notch = crate::iir::Biquad::new(
            crate::iir::BiquadFilterType::Notch,
            1_000.0,
            48_000.0,
            1.0,
            0.0,
        );
        let refusal = profile
            .validate_filters(48_000.0, &[notch])
            .expect_err("declaring a device type cannot expand the verified APO syntax subset");
        assert!(
            refusal.contains("refuses unverified filter type"),
            "{refusal}"
        );
    }

    #[test]
    fn profiled_apo_requires_lsc_hsc_and_validates_fixed_slope_shelves() {
        for model in [crate::PeqModel::LsPk, crate::PeqModel::PkLsHs] {
            let mut params = optimizer_params();
            params.peq_model = model;
            params.num_filters = 3;
            let mut profile = device_profile("studio", 48_000.0);
            profile.supported_peq_models = vec![model.to_string()];
            profile.supported_filter_types = vec!["PK".into(), "LSC".into(), "HSC".into()];

            profile
                .validate_for_optimizer(&params)
                .expect("shelf models pass when the device profile declares LSC/HSC");

            let mut legacy_declaration = profile.clone();
            legacy_declaration.supported_filter_types = vec!["PK".into(), "LS".into(), "HS".into()];
            let refusal = legacy_declaration
                .validate_for_optimizer(&params)
                .expect_err("legacy LS/HS declarations do not attest to LSC/HSC output");
            assert!(refusal.contains("filter type"), "{refusal}");
        }

        let mut profile = device_profile("studio", 48_000.0);
        profile.supported_filter_types = vec!["PK".into(), "LSC".into(), "HSC".into()];
        for shelf in [
            crate::iir::Biquad::new(
                crate::iir::BiquadFilterType::Lowshelf,
                100.0,
                48_000.0,
                0.71,
                3.0,
            ),
            crate::iir::Biquad::new(
                crate::iir::BiquadFilterType::Highshelf,
                10_000.0,
                48_000.0,
                0.71,
                -3.0,
            ),
        ] {
            profile
                .validate_filters(48_000.0, &[shelf])
                .expect("12 dB center shelves are supported by the profiled serializer");
        }
    }

    #[test]
    fn profiled_apo_optimizer_gate_uses_actual_hpq_and_lpq_emission_kinds() {
        let mut params = optimizer_params();
        params.peq_model = crate::PeqModel::HpPk;
        params.num_filters = 2;
        let mut profile = device_profile("studio", 48_000.0);
        profile.supported_peq_models = vec!["hp-pk".into()];
        profile.supported_filter_types = vec!["HPQ".into(), "PK".into()];
        assert!(profile.validate_for_optimizer(&params).is_ok());

        profile.supported_filter_types = vec!["HP".into(), "PK".into()];
        let refusal = profile
            .validate_for_optimizer(&params)
            .expect_err("HPQ output must be declared explicitly");
        assert!(refusal.contains("HPQ"), "{refusal}");

        params.peq_model = crate::PeqModel::HpPkLp;
        params.num_filters = 3;
        profile.supported_peq_models = vec!["hp-pk-lp".into()];
        profile.supported_filter_types = vec!["HPQ".into(), "PK".into(), "LPQ".into()];
        assert!(profile.validate_for_optimizer(&params).is_ok());
        profile.supported_filter_types.retain(|kind| kind != "LPQ");
        assert!(
            profile
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("LPQ")
        );

        params.max_q = crate::iir::DEFAULT_Q_HIGH_LOW_PASS;
        profile.supported_filter_types = vec!["HPQ".into(), "PK".into(), "LPQ".into()];
        assert!(
            profile
                .validate_for_optimizer(&params)
                .unwrap_err()
                .contains("LP' required")
        );
        profile.supported_filter_types.push("LP".into());
        assert!(profile.validate_for_optimizer(&params).is_ok());
    }

    #[test]
    fn room_mode_delegates_even_when_source_is_a_csv() {
        let source = ProductSource::Csv {
            path: PathBuf::from("measurement.csv"),
            measurement_rig: None,
        };
        assert!(
            validate_source_mode(ProductMode::Room, &source)
                .unwrap_err()
                .to_string()
                .contains("RoomEQ CLI/workflow")
        );
    }

    #[tokio::test]
    async fn headphone_api_source_uses_injected_offline_backend_and_cache() {
        let cache_root = tempfile::tempdir().unwrap();
        let backend =
            MockMeasurementBackend::new("frequency,spl\n20,-2\n100,-1\n1000,0\n20000,-3\n");
        let cache = InMemoryMeasurementCache::new();
        let coupler = rig(
            MeasurementRigKind::HeadphoneCoupler,
            "coupler-db",
            "fixture-x",
        );
        let target = ProductTarget::Named {
            name: "flat".into(),
            supported_measurement_rigs: vec![coupler.clone()],
        };
        let request = ProductRequest {
            mode: ProductMode::Headphone,
            source: ProductSource::HeadphoneApi {
                headphone: "Example Headphone".into(),
                measurement_rig: Some(coupler),
            },
            target,
            device_profile: device_profile("headphone-dac", 48_000.0),
            reject_declared_target_mismatch: true,
        };
        let adapters = ProductSourceAdapters {
            cache_root: cache_root.path(),
            backend: &backend,
            cache: &cache,
        };
        let prepared = PreparedProduct::load(&request, &headphone_optimizer_params(), &adapters)
            .await
            .unwrap();
        assert_eq!(
            prepared.source_record.provenance.origin,
            MeasurementOrigin::Api
        );
        assert_eq!(
            prepared.source_record.provenance.source_id.as_deref(),
            Some("spinorama:headphone:Example Headphone")
        );
        assert_eq!(
            prepared.target_compatibility.status,
            TargetCompatibilityStatus::DeclaredMatch
        );
        assert!(prepared.source_record.provenance.measurement_rig.is_some());
        assert!(
            prepared
                .target_profile
                .record
                .provenance
                .source_artifacts
                .is_empty()
        );
        let headphone_cache =
            read::headphone_cache_dir_for_cache_root(cache_root.path(), "Example Headphone");
        assert!(
            cache
                .get(&headphone_cache.join("frequency_response_raw.csv"))
                .is_some()
        );
    }

    #[tokio::test]
    async fn local_custom_target_keeps_separate_lineage_and_unknown_rig() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.csv");
        let target_path = directory.path().join("target.csv");
        std::fs::write(
            &source_path,
            "frequency,spl\n20,70\n100,71\n1000,75\n20000,72\n",
        )
        .unwrap();
        std::fs::write(
            &target_path,
            "frequency,spl\n20,0\n100,1\n1000,2\n20000,0\n",
        )
        .unwrap();
        let backend = MockMeasurementBackend::new("");
        let cache = InMemoryMeasurementCache::new();
        let request = ProductRequest {
            mode: ProductMode::Speaker,
            source: ProductSource::Csv {
                path: source_path.clone(),
                measurement_rig: None,
            },
            target: ProductTarget::Csv {
                path: target_path.clone(),
                supported_measurement_rigs: vec![],
            },
            device_profile: device_profile("speaker-dac", 48_000.0),
            reject_declared_target_mismatch: false,
        };
        let adapters = ProductSourceAdapters {
            cache_root: directory.path(),
            backend: &backend,
            cache: &cache,
        };
        let prepared = PreparedProduct::load(&request, &optimizer_params(), &adapters)
            .await
            .unwrap();
        assert_ne!(
            prepared.source_record.provenance.content_hash,
            prepared.target_profile.record.provenance.content_hash
        );
        assert_eq!(
            prepared.source_record.provenance.source_artifacts[0]
                .uri
                .as_deref(),
            Some(source_path.to_str().unwrap())
        );
        assert_eq!(
            prepared.target_profile.record.provenance.source_artifacts[0]
                .uri
                .as_deref(),
            Some(target_path.to_str().unwrap())
        );
        assert_eq!(
            prepared.target_compatibility.status,
            TargetCompatibilityStatus::Unknown
        );
        assert!(prepared.target_profile.record.curve.spl[2] > 1.0);
    }

    #[test]
    fn product_json_rejects_unknown_fields_and_defaults_to_strict_mismatch() {
        let mut profile: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/device_profile_studio_48k.json"
        ))
        .unwrap();
        profile["playbak_device_id"] = serde_json::Value::String("typo".into());
        assert!(serde_json::from_value::<DeviceProfile>(profile).is_err());

        let request = serde_json::json!({
            "mode": "speaker",
            "source": {"kind": "csv", "path": "input.csv"},
            "target": {"kind": "csv", "path": "target.csv"},
            "device_profile": serde_json::from_str::<serde_json::Value>(include_str!(
                "../../tests/fixtures/device_profile_studio_48k.json"
            )).unwrap()
        });
        let parsed: ProductRequest = serde_json::from_value(request.clone()).unwrap();
        assert!(parsed.reject_declared_target_mismatch);

        let mut unknown_request = request;
        unknown_request["playbak_device_id"] = serde_json::Value::String("typo".into());
        assert!(serde_json::from_value::<ProductRequest>(unknown_request).is_err());
    }

    #[test]
    fn product_mode_refuses_to_extrapolate_source_or_target_bands() {
        let short = Curve {
            freq: Array1::from_vec(vec![20.0, 100.0, 1_000.0, 10_000.0]),
            spl: Array1::from_vec(vec![0.0; 4]),
            ..Default::default()
        };
        let error = ensure_curve_covers_band(&short, "target source", 20.0, 20_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("refuses frequency extrapolation"));
    }
}
