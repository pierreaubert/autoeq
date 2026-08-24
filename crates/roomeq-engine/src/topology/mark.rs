/// Plugin parameter used to group processors which must be retained or
/// reverted as one correction stage.
pub const CORRECTION_STAGE_PARAMETER: &str = "room_eq_correction_stage";

/// The split, branch processors, and merge which implement Hybrid mode form
/// one indivisible DSP block.  Removing only its FIR or PEQ branch leaves an
/// invalid topology behind.
pub const HYBRID_CROSSOVER_CORRECTION_STAGE: &str = "hybrid_crossover";

pub fn mark_plugin_correction_stage(
    mut plugin: roomeq_model::PluginConfigWrapper,
    stage: &str,
) -> roomeq_model::PluginConfigWrapper {
    if let Some(params) = plugin.parameters.as_object_mut() {
        params.insert(
            CORRECTION_STAGE_PARAMETER.to_string(),
            serde_json::json!(stage),
        );
    }
    plugin
}

pub fn mark_route_owned_plugin(
    mut plugin: roomeq_model::PluginConfigWrapper,
) -> roomeq_model::PluginConfigWrapper {
    if let Some(params) = plugin.parameters.as_object_mut() {
        params.insert(
            "room_eq_stage".to_string(),
            serde_json::json!("route_owned"),
        );
        params.insert(
            "label".to_string(),
            serde_json::json!("room_eq_route_owned"),
        );
    }
    plugin
}

pub fn mark_plugin_stage(
    mut plugin: roomeq_model::PluginConfigWrapper,
    stage: &str,
) -> roomeq_model::PluginConfigWrapper {
    if let Some(params) = plugin.parameters.as_object_mut() {
        params.insert("room_eq_stage".to_string(), serde_json::json!(stage));
    }
    plugin
}

pub fn mark_plugins_stage(
    plugins: Vec<roomeq_model::PluginConfigWrapper>,
    stage: &str,
) -> Vec<roomeq_model::PluginConfigWrapper> {
    plugins
        .into_iter()
        .map(|plugin| mark_plugin_stage(plugin, stage))
        .collect()
}
