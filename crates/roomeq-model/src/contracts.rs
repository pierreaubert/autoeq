//! Engine-neutral contracts shared by RoomEQ execution and exporters.

use crate::{ChannelDspChain, CurveData, OptimizationMetadata, PluginConfigWrapper};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plugin {
    pub kind: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ChannelChain {
    pub plugins: Vec<Plugin>,
}

/// DSP chain output (AudioEngine PluginConfig format)
//
// This is the canonical graph external renderers consume: global routing
// plugins, stable channel identities, per-driver branches, resolved topology
// metadata, and optimization curves for lossless conversion.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(rename = "DspChainOutput")]
pub struct DspGraph {
    /// Output version
    #[serde(default = "crate::config::default_config_version")]
    pub version: String,
    /// Global graph-level plugins, e.g. matrix routing that combines several
    /// programme inputs before per-output correction chains.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_plugins: Vec<PluginConfigWrapper>,
    /// Per-channel DSP chains
    pub channels: HashMap<String, ChannelDspChain>,
    /// Coherent deployed response for each logical input after routing.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub deployed_source_curves: HashMap<String, CurveData>,
    /// Metadata about the optimization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<OptimizationMetadata>,
}

impl DspGraph {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            global_plugins: Vec::new(),
            channels: HashMap::new(),
            deployed_source_curves: HashMap::new(),
            metadata: None,
        }
    }

    pub fn add_channel(&mut self, name: impl Into<String>, plugins: Vec<Plugin>) {
        let name = name.into();
        self.channels.insert(
            name.clone(),
            ChannelDspChain {
                channel: name,
                plugins: plugins
                    .into_iter()
                    .map(|plugin| PluginConfigWrapper {
                        plugin_type: plugin.kind,
                        parameters: plugin.parameters,
                    })
                    .collect(),
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        );
    }

    /// Validate the backend-neutral graph invariants shared by all exporters.
    pub fn validate(&self) -> Result<(), String> {
        if self.version.trim().is_empty() {
            return Err("DSP graph version must not be empty".to_string());
        }
        if self.channels.is_empty() {
            return Err("DSP graph requires at least one channel".to_string());
        }
        validate_plugins("global graph", &self.global_plugins)?;
        for (name, chain) in &self.channels {
            if name.trim().is_empty() {
                return Err("DSP graph channel names must not be empty".to_string());
            }
            if chain.channel != *name {
                return Err(format!(
                    "DSP graph channel key '{name}' does not match embedded channel name '{}'",
                    chain.channel
                ));
            }
            validate_plugins(&format!("channel '{name}'"), &chain.plugins)?;
            if let Some(drivers) = &chain.drivers {
                for driver in drivers {
                    validate_plugins(
                        &format!("channel '{name}' driver '{}'", driver.name),
                        &driver.plugins,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn validate_plugins(context: &str, plugins: &[PluginConfigWrapper]) -> Result<(), String> {
    if plugins
        .iter()
        .any(|plugin| plugin.plugin_type.trim().is_empty())
    {
        return Err(format!(
            "DSP graph {context} contains a plugin with an empty type"
        ));
    }
    Ok(())
}
