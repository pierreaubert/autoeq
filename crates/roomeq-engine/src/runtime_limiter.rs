//! Explicit native sub-output limiter contract and its small-signal latency.
//!
//! Frequency responses describe only operation below the limiter threshold.
//! Overload protection is nonlinear and must be retained by the playback host.

use roomeq_model::{AutoeqError, PluginConfigWrapper, Result};

/// Label identifying RoomEQ's mandatory physical-output protection.
pub const LABEL: &str = "room_eq_sub_output_limiter";
/// Five milliseconds gives the native peak detector time to anticipate bass peaks.
pub const LOOKAHEAD_MS: f64 = 5.0;
/// A 100 ms release spans multiple bass periods, reducing gain modulation.
pub const RELEASE_MS: f64 = 100.0;

/// Create the supported, fully wet, hard sample-peak limiter.
pub fn plugin(output_ceiling_dbfs: f64) -> PluginConfigWrapper {
    PluginConfigWrapper {
        plugin_type: "limiter".into(),
        parameters: serde_json::json!({
            "threshold_db": output_ceiling_dbfs.min(-1.0),
            "release_ms": RELEASE_MS,
            "lookahead_ms": LOOKAHEAD_MS,
            "soft": false, "true_peak": false, "isp_mode": false,
            "dual_release": false, "mix": 1.0, "feed_forward": true,
            "link_amount": 1.0,
            "label": LABEL, "room_eq_stage": "post_route"
        }),
    }
}

/// Validate the exact supported limiter and return its sample-peak ceiling in dBFS.
///
/// # Errors
/// Rejects unknown, bypassed, partially wet, or unsupported limiter settings.
pub fn ceiling(plugin: &PluginConfigWrapper) -> Result<f64> {
    let p = &plugin.parameters;
    let threshold = p["threshold_db"].as_f64().unwrap_or(f64::NAN);
    let expected = self::plugin(threshold);
    if plugin.plugin_type != "limiter"
        || !threshold.is_finite()
        || !(-20.0..=-1.0).contains(&threshold)
        || p != &expected.parameters
    {
        return Err(AutoeqError::InvalidConfiguration {
            message: "unsupported runtime sub-output limiter contract".into(),
        });
    }
    Ok(threshold)
}

/// Return the native limiter's integer-sample lookahead delay.
pub fn latency_samples(sample_rate: f64) -> usize {
    // Match the plugin's f32 calculation and truncation, including at 44.1 kHz.
    (LOOKAHEAD_MS as f32 * 0.001 * sample_rate as f32) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_is_explicit_and_fail_closed() {
        assert_eq!(ceiling(&plugin(0.0)).unwrap(), -1.0);
        assert_eq!(ceiling(&plugin(-3.0)).unwrap(), -3.0);
        for (key, value) in [
            ("mix", serde_json::json!(0.5)),
            ("soft", serde_json::json!(true)),
            ("lookahead_ms", serde_json::json!(0.0)),
            ("threshold_db", serde_json::json!(1.0)),
            ("room_eq_stage", serde_json::json!("pre_route")),
        ] {
            let mut modified = plugin(0.0);
            modified.parameters[key] = value;
            assert!(ceiling(&modified).is_err(), "{key}");
        }
        assert_eq!(latency_samples(48_000.0), 240);
        assert_eq!(latency_samples(44_100.0), 220);
    }
}
