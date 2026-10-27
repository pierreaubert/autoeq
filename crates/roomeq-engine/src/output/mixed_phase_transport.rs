//! Internal mixed-phase report transport on convolution plugins.
//!
//! The report type is engine-owned (`crate::mixed_phase`), so the
//! transport helpers stay in the engine while the portable plugin
//! builders live in `roomeq-dsp`.

use roomeq_dsp::output::create_convolution_plugin;
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};
use std::collections::HashMap;

const MIXED_PHASE_REPORT_TRANSPORT_KEY: &str = "_roomeq_mixed_phase_report";

/// Create a convolution plugin carrying internal mixed-phase report data.
///
/// The marker is removed by [`take_mixed_phase_reports`] before the public DSP
/// chain is returned, so external convolution-plugin contracts remain stable.
pub fn create_mixed_phase_convolution_plugin(
    wav_path: &str,
    report: &crate::mixed_phase::MixedPhaseCorrectionReport,
) -> PluginConfigWrapper {
    let mut plugin = create_convolution_plugin(wav_path);
    if let Some(parameters) = plugin.parameters.as_object_mut() {
        parameters.insert(
            MIXED_PHASE_REPORT_TRANSPORT_KEY.to_string(),
            serde_json::to_value(report).unwrap_or(serde_json::Value::Null),
        );
    }
    plugin
}

/// Collect exact mixed-phase reports and remove their internal transport
/// markers from convolution plugin parameters.
pub fn take_mixed_phase_reports(
    channels: &mut HashMap<String, ChannelDspChain>,
) -> Option<HashMap<String, crate::mixed_phase::MixedPhaseCorrectionReport>> {
    let mut reports = HashMap::new();
    for (channel_name, chain) in channels {
        for plugin in &mut chain.plugins {
            let Some(parameters) = plugin.parameters.as_object_mut() else {
                continue;
            };
            let Some(value) = parameters.remove(MIXED_PHASE_REPORT_TRANSPORT_KEY) else {
                continue;
            };
            match serde_json::from_value(value) {
                Ok(report) => {
                    reports.insert(channel_name.clone(), report);
                }
                Err(error) => log::warn!(
                    "failed to decode internal mixed-phase report for '{channel_name}': {error}"
                ),
            }
        }
    }
    (!reports.is_empty()).then_some(reports)
}
