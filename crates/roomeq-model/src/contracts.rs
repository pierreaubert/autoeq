//! Small, engine-neutral contracts shared by RoomEQ execution and exporters.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DspGraph {
    pub version: String,
    pub channels: BTreeMap<String, ChannelChain>,
}

impl DspGraph {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            channels: BTreeMap::new(),
        }
    }

    pub fn add_channel(&mut self, name: impl Into<String>, plugins: Vec<Plugin>) {
        self.channels.insert(name.into(), ChannelChain { plugins });
    }

    /// Validate the minimum contract shared by engines and exporters.
    pub fn validate(&self) -> Result<(), String> {
        if self.version.trim().is_empty() {
            return Err("DSP graph version must not be empty".to_string());
        }
        if self.channels.is_empty() {
            return Err("DSP graph requires at least one channel".to_string());
        }
        for (name, chain) in &self.channels {
            if name.trim().is_empty() {
                return Err("DSP graph channel names must not be empty".to_string());
            }
            if chain
                .plugins
                .iter()
                .any(|plugin| plugin.kind.trim().is_empty())
            {
                return Err(format!(
                    "DSP graph channel '{name}' contains a plugin with an empty kind"
                ));
            }
        }
        Ok(())
    }
}
