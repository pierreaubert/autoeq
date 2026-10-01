//! Optional measured room impulse responses per channel.
//!
//! A declared IR is a `time_ms,amplitude` CSV (the same layout the output
//! bundle sidecars use) captured in the room, e.g. a swept-sine
//! deconvolution at the listening position. Declared IRs back the R1–R5
//! acoustic report fields (early reflections, early/late curves, octave
//! T60, waterfall/resonances, wavelet) through the math-audio report
//! primitives. Channels without a declaration keep synthesized IRs and
//! their report cells stay pending: the workflow never runs room-acoustic
//! analysis on a synthesized IR.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One measured room impulse response backing a channel's acoustic report.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MeasuredIrSource {
    /// Delivered channel; defaults to the declaration's map key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_channel: Option<String>,
    /// Exact delivered driver name; omit for a whole-channel capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    /// IR file path, resolved against the configuration directory.
    /// CSV with a `time_ms,amplitude` header.
    pub path: PathBuf,
    /// Declared IR sample rate in Hz. When present it is verified against
    /// the file's time grid; a mismatch fails ingestion fail-closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<f64>,
    /// Shared clock identity across simultaneously captured channels.
    /// Optional for independent diagnostics. Absence never establishes
    /// eligibility for coherent sums or cross-channel timing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_reference_id: Option<String>,
}

impl MeasuredIrSource {
    /// Check declared values (file existence is verified at ingestion,
    /// where the configuration directory is known).
    ///
    /// # Errors
    ///
    /// Returns a reason for a non-finite or out-of-range declared sample
    /// rate, or a blank timing reference.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(rate) = self.sample_rate_hz
            && (!rate.is_finite() || rate < 1_000.0 || rate > 192_000.0)
        {
            return Err(format!(
                "measured IR sample rate must be within 1–192 kHz, got {rate}"
            ));
        }
        if let Some(timing) = &self.timing_reference_id
            && (timing.trim().is_empty() || timing.trim().eq_ignore_ascii_case("unknown"))
        {
            return Err(String::from(
                "measured IR timing reference must be a real clock identity, not a placeholder",
            ));
        }
        for target in [&self.output_channel, &self.driver].into_iter().flatten() {
            if target.trim().is_empty() {
                return Err("measured IR output channel and driver must not be blank".into());
            }
        }
        Ok(())
    }
}

/// Validate a channel-keyed measured-IR map.
///
/// # Errors
///
/// Returns a reason for an invalid entry or duplicate delivered target.
/// This validates independent diagnostics, not coherent timing eligibility.
pub fn validate_measured_ir_map(map: &BTreeMap<String, MeasuredIrSource>) -> Result<(), String> {
    for (channel, source) in map {
        source
            .validate()
            .map_err(|reason| format!("channel '{channel}': {reason}"))?;
    }
    let mut targets = std::collections::BTreeSet::new();
    for (key, source) in map {
        let target = (
            source.output_channel.as_deref().unwrap_or(key),
            source.driver.as_deref(),
        );
        if !targets.insert(target) {
            return Err(format!("duplicate measured IR target for '{key}'"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(rate: Option<f64>, timing: Option<&str>) -> MeasuredIrSource {
        MeasuredIrSource {
            output_channel: None,
            driver: None,
            path: PathBuf::from("left__ir.csv"),
            sample_rate_hz: rate,
            timing_reference_id: timing.map(str::to_string),
        }
    }

    #[test]
    fn bare_path_validates() {
        assert!(source(None, None).validate().is_ok());
    }

    #[test]
    fn declared_rate_must_be_audio_range() {
        assert!(source(Some(48_000.0), None).validate().is_ok());
        assert!(source(Some(0.0), None).validate().is_err());
        assert!(source(Some(f64::NAN), None).validate().is_err());
        assert!(source(Some(3_000.0), None).validate().is_ok());
        assert!(source(Some(999.0), None).validate().is_err());
    }

    #[test]
    fn placeholder_timing_rejected() {
        assert!(source(None, Some("  ")).validate().is_err());
        assert!(source(None, Some("unknown")).validate().is_err());
        assert!(source(None, Some("clk-1")).validate().is_ok());
    }

    #[test]
    fn single_channel_needs_no_timing() {
        let map = BTreeMap::from([("L".to_string(), source(None, None))]);
        assert!(validate_measured_ir_map(&map).is_ok());
    }

    #[test]
    fn duplicate_driver_target_rejected() {
        let mut sub = source(Some(3_000.0), None);
        sub.output_channel = Some("Sub1".into());
        sub.driver = Some("Sub2".into());
        let map = BTreeMap::from([("first".into(), sub.clone()), ("second".into(), sub)]);
        assert!(validate_measured_ir_map(&map).is_err());
    }

    #[test]
    fn independent_captures_do_not_require_shared_timing() {
        let map = BTreeMap::from([
            ("L".to_string(), source(None, Some("clk-1"))),
            ("R".to_string(), source(None, Some("clk-1"))),
        ]);
        assert!(validate_measured_ir_map(&map).is_ok());
        let map = BTreeMap::from([
            ("L".to_string(), source(None, Some("clk-1"))),
            ("R".to_string(), source(None, None)),
        ]);
        assert!(validate_measured_ir_map(&map).is_ok());
        let map = BTreeMap::from([
            ("L".to_string(), source(None, Some("clk-1"))),
            ("R".to_string(), source(None, Some("clk-2"))),
        ]);
        assert!(validate_measured_ir_map(&map).is_ok());
    }
}
