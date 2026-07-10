use super::super::types::{ChannelDspChain, PluginConfigWrapper};

/// Collect all plugins from a channel: combined (channel-level) + per-driver plugins
pub(super) fn collect_all_plugins(chain: &ChannelDspChain) -> Vec<&PluginConfigWrapper> {
    let mut all = Vec::new();
    if let Some(drivers) = &chain.drivers {
        for driver in drivers {
            all.extend(driver.plugins.iter());
        }
    }
    all.extend(chain.plugins.iter());
    all
}
