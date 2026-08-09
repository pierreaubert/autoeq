use super::types::BiquadExport;
use roomeq_model::PluginConfigWrapper;

/// Extract all biquad filters from a plugin list
pub(super) fn extract_eq_filters(
    plugins: &[PluginConfigWrapper],
) -> anyhow::Result<Vec<BiquadExport>> {
    let mut filters = Vec::new();
    for (plugin_index, plugin) in plugins.iter().enumerate() {
        if plugin.plugin_type != "eq" {
            continue;
        }
        let arr = plugin
            .parameters
            .get("filters")
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!("EQ plugin #{plugin_index} requires an array field 'filters'")
            })?;
        for (filter_index, value) in arr.iter().enumerate() {
            let filter = value.as_object().ok_or_else(|| {
                anyhow::anyhow!(
                    "EQ plugin #{plugin_index} filter #{filter_index} must be an object"
                )
            })?;
            let context = format!("EQ plugin #{plugin_index} filter #{filter_index}");
            let required_f64 = |name: &str| {
                filter
                    .get(name)
                    .and_then(|value| value.as_f64())
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| anyhow::anyhow!("{context} requires finite numeric '{name}'"))
            };
            filters.push(BiquadExport {
                filter_type: filter
                    .get("filter_type")
                    .and_then(|value| value.as_str())
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("{context} requires string 'filter_type'"))?
                    .to_string(),
                freq: required_f64("freq")?,
                q: required_f64("q")?,
                gain_db: required_f64("db_gain")?,
            });
        }
    }
    Ok(filters)
}

/// Sum all gain values from gain plugins
pub(super) fn extract_gain_db(plugins: &[PluginConfigWrapper]) -> f64 {
    plugins
        .iter()
        .filter(|p| p.plugin_type == "gain")
        .filter_map(|p| p.parameters.get("gain_db").and_then(|v| v.as_f64()))
        .sum()
}

/// Extract delay in ms (sum of all delay plugins)
pub(super) fn extract_delay_ms(plugins: &[PluginConfigWrapper]) -> Option<f64> {
    let total: f64 = plugins
        .iter()
        .filter(|p| p.plugin_type == "delay")
        .filter_map(|p| p.parameters.get("delay_ms").and_then(|v| v.as_f64()))
        .sum();
    if total.abs() > 0.001 {
        Some(total)
    } else {
        None
    }
}

/// Extract convolution IR file paths
pub(super) fn extract_convolution_paths(plugins: &[PluginConfigWrapper]) -> Vec<String> {
    plugins
        .iter()
        .filter(|p| p.plugin_type == "convolution")
        .filter_map(|p| {
            p.parameters
                .get("ir_file")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect()
}
