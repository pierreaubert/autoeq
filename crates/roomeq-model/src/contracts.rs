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
}
