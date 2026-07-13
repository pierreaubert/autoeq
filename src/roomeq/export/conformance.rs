use super::super::home_cinema::BassManagementRoutingGraph;
use super::super::types::{DspChainOutput, PluginConfigWrapper};
use super::channel::sorted_channels;
use super::export_format::ExportFormat;
use super::write::camilladsp_crossover_filter_type;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ExportNodeKind {
    Processor,
    Router,
}

impl ExportNodeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Processor => "processor",
            Self::Router => "router",
        }
    }
}

#[derive(Debug)]
struct ExportNodeReference {
    kind: ExportNodeKind,
    name: String,
    context: String,
}

/// Backend-neutral accounting for nodes emitted by an external exporter.
///
/// Exporters register every definition and pipeline reference while rendering.
/// The final validation prevents dangling references and duplicate identifiers
/// without requiring a backend-specific parser. Later exporters can reuse the
/// same accounting and add their own native syntax/schema validation.
#[derive(Debug)]
pub(super) struct ExportArtifactManifest {
    format: ExportFormat,
    definitions: BTreeSet<(ExportNodeKind, String)>,
    references: Vec<ExportNodeReference>,
}

impl ExportArtifactManifest {
    pub(super) fn new(format: ExportFormat) -> Self {
        Self {
            format,
            definitions: BTreeSet::new(),
            references: Vec::new(),
        }
    }

    pub(super) fn define_node(&mut self, kind: ExportNodeKind, name: &str) -> anyhow::Result<()> {
        if name.is_empty() {
            anyhow::bail!(
                "{:?} export emitted an empty {} name",
                self.format,
                kind.label()
            );
        }
        if !self.definitions.insert((kind, name.to_string())) {
            anyhow::bail!(
                "{:?} export emitted duplicate {} name '{name}'",
                self.format,
                kind.label()
            );
        }
        Ok(())
    }

    pub(super) fn reference_node(
        &mut self,
        kind: ExportNodeKind,
        name: &str,
        context: impl Into<String>,
    ) {
        self.references.push(ExportNodeReference {
            kind,
            name: name.to_string(),
            context: context.into(),
        });
    }

    pub(super) fn reference_channel(
        &self,
        channel_index: usize,
        channel_count: usize,
        context: &str,
    ) -> anyhow::Result<()> {
        if channel_index >= channel_count {
            anyhow::bail!(
                "{:?} export {context} references channel {channel_index}, but only \
                 {channel_count} channels are available",
                self.format
            );
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> anyhow::Result<()> {
        for reference in &self.references {
            if !self
                .definitions
                .contains(&(reference.kind, reference.name.clone()))
            {
                anyhow::bail!(
                    "{:?} export {} references missing {} '{}'",
                    self.format,
                    reference.context,
                    reference.kind.label(),
                    reference.name
                );
            }
        }
        for (kind, name) in &self.definitions {
            if !self
                .references
                .iter()
                .any(|reference| reference.kind == *kind && reference.name == *name)
            {
                anyhow::bail!(
                    "{:?} export emitted unreferenced {} '{name}'",
                    self.format,
                    kind.label()
                );
            }
        }
        Ok(())
    }
}

type PluginValidator =
    fn(plugin: &PluginConfigWrapper, sample_rate: Option<f64>, context: &str) -> anyhow::Result<()>;

struct ExportCapabilities {
    format: ExportFormat,
    supported_plugin_types: &'static [&'static str],
    supports_driver_branches: bool,
    routed_stages: &'static [&'static str],
    normalize_identifier: fn(&str) -> String,
    validate_plugin: PluginValidator,
}

const CAMILLADSP_PLUGIN_TYPES: &[&str] = &["gain", "delay", "eq", "crossover", "convolution"];
const CAMILLADSP_ROUTED_STAGES: &[&str] = &["pre_route", "route_owned", "post_route"];
const PIPEWIRE_PLUGIN_TYPES: &[&str] = &["gain", "delay", "eq", "crossover", "convolution"];

fn camilladsp_capabilities() -> ExportCapabilities {
    ExportCapabilities {
        format: ExportFormat::CamillaDsp,
        supported_plugin_types: CAMILLADSP_PLUGIN_TYPES,
        supports_driver_branches: false,
        routed_stages: CAMILLADSP_ROUTED_STAGES,
        normalize_identifier: normalize_export_identifier,
        validate_plugin: validate_camilladsp_plugin,
    }
}

pub(super) fn normalize_export_identifier(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

pub(super) fn validate_camilladsp_input(
    output: &DspChainOutput,
    sample_rate: Option<f64>,
) -> anyhow::Result<()> {
    let graph = super::camilladsp_routing_graph(output);
    let routed_expected = routed_bass_management_declared(output);
    if routed_expected && graph.is_none() {
        anyhow::bail!(
            "CamillaDsp export found routed bass-management metadata but could not decode its graph"
        );
    }

    validate_camilladsp_global_plugins(output, routed_expected)?;
    let capabilities = camilladsp_capabilities();
    validate_export_input(output, &capabilities, sample_rate, graph.as_ref())?;
    if let Some(graph) = graph.as_ref().filter(|graph| !graph.routes.is_empty()) {
        validate_camilladsp_routing_graph(output, graph, sample_rate)?;
    }
    Ok(())
}

pub(super) fn validate_pipewire_input(
    output: &DspChainOutput,
    sample_rate: Option<f64>,
) -> anyhow::Result<()> {
    if let Some(plugin) = output.global_plugins.first() {
        anyhow::bail!(
            "PipeWire export does not yet support global plugin #0 ('{}'); use the \
             CamillaDSP export for routed or matrix DSP graphs",
            plugin.plugin_type
        );
    }

    let capabilities = ExportCapabilities {
        format: ExportFormat::PipeWire,
        supported_plugin_types: PIPEWIRE_PLUGIN_TYPES,
        supports_driver_branches: false,
        routed_stages: &[],
        normalize_identifier: normalize_export_identifier,
        validate_plugin: validate_pipewire_plugin,
    };
    validate_export_input(output, &capabilities, sample_rate, None)
}

pub(super) fn validate_serial_external_input(
    output: &DspChainOutput,
    format: ExportFormat,
) -> anyhow::Result<()> {
    if output.channels.is_empty() {
        anyhow::bail!("{format:?} export requires at least one channel");
    }
    if let Some(plugin) = output.global_plugins.first() {
        anyhow::bail!(
            "{format:?} export does not support global plugin #0 ('{}')",
            plugin.plugin_type
        );
    }

    let allowed: &[&str] = match format {
        ExportFormat::EqualizerApo | ExportFormat::RoonDsp => {
            &["gain", "delay", "eq", "convolution"]
        }
        ExportFormat::EasyEffects | ExportFormat::Wavelet => &["gain", "eq"],
        _ => anyhow::bail!("internal error: no serial validator for {format:?}"),
    };

    let mut channels: Vec<_> = output.channels.iter().collect();
    channels.sort_by(|a, b| a.0.cmp(b.0));
    for (channel_name, chain) in &channels {
        if chain.channel != **channel_name {
            anyhow::bail!(
                "{format:?} export channel map key '{}' does not match chain channel '{}'",
                channel_name,
                chain.channel
            );
        }
        if chain
            .drivers
            .as_ref()
            .is_some_and(|drivers| !drivers.is_empty())
        {
            anyhow::bail!(
                "{format:?} export cannot represent active-crossover driver branches for channel '{channel_name}'"
            );
        }
        let mut convolution_count = 0usize;
        let mut eq_filter_count = 0usize;
        for (plugin_index, plugin) in chain.plugins.iter().enumerate() {
            let context = format!(
                "channel '{channel_name}' plugin #{plugin_index} ('{}')",
                plugin.plugin_type
            );
            if !allowed.contains(&plugin.plugin_type.as_str()) {
                anyhow::bail!("{format:?} export does not support {context}");
            }
            let parameters = plugin.parameters.as_object().ok_or_else(|| {
                anyhow::anyhow!("{format:?} export requires object parameters on {context}")
            })?;
            match plugin.plugin_type.as_str() {
                "gain" => {
                    required_f64(parameters, "gain_db", &context)?;
                    optional_bool(parameters, "invert", &context)?;
                    if parameters.get("invert").and_then(|value| value.as_bool()) == Some(true) {
                        anyhow::bail!(
                            "{format:?} export cannot represent polarity inversion on {context}"
                        );
                    }
                }
                "delay" => {
                    let delay = required_f64(parameters, "delay_ms", &context)?;
                    if delay < 0.0 {
                        anyhow::bail!("{format:?} export requires non-negative delay on {context}");
                    }
                }
                "eq" => {
                    let filters = parameters
                        .get("filters")
                        .and_then(|value| value.as_array())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "{format:?} export requires an array field 'filters' on {context}"
                            )
                        })?;
                    eq_filter_count += filters.len();
                    for (index, filter) in filters.iter().enumerate() {
                        let filter_context = format!("{context}, filter #{index}");
                        let filter = filter.as_object().ok_or_else(|| {
                            anyhow::anyhow!(
                                "{format:?} export requires an object for {filter_context}"
                            )
                        })?;
                        let filter_type = required_str(filter, "filter_type", &filter_context)?;
                        if !matches!(
                            filter_type,
                            "peak"
                                | "lowshelf"
                                | "highshelf"
                                | "lowpass"
                                | "highpass"
                                | "highpassvariableq"
                                | "notch"
                                | "bandpass"
                                | "allpass"
                        ) {
                            anyhow::bail!(
                                "{format:?} export does not support filter type '{filter_type}' on {filter_context}"
                            );
                        }
                        if matches!(format, ExportFormat::RoonDsp) && filter_type == "allpass" {
                            anyhow::bail!(
                                "RoonDsp export cannot represent all-pass filter on {filter_context}"
                            );
                        }
                        let frequency = required_f64(filter, "freq", &filter_context)?;
                        let q = required_f64(filter, "q", &filter_context)?;
                        if frequency <= 0.0 || q <= 0.0 {
                            anyhow::bail!(
                                "{format:?} export requires positive frequency and Q on {filter_context}"
                            );
                        }
                        required_f64(filter, "db_gain", &filter_context)?;
                    }
                }
                "convolution" => {
                    convolution_count += 1;
                    let ir_file = required_str(parameters, "ir_file", &context)?;
                    if matches!(format, ExportFormat::RoonDsp) {
                        let path = std::path::Path::new(ir_file);
                        if path.is_absolute()
                            || path.components().any(|component| {
                                !matches!(component, std::path::Component::Normal(_))
                            })
                        {
                            anyhow::bail!(
                                "RoonDsp export requires a safe relative impulse path on {context}"
                            );
                        }
                        for key in parameters.keys() {
                            if key != "ir_file" {
                                anyhow::bail!(
                                    "RoonDsp export does not support convolution parameter '{key}' on {context}"
                                );
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        if matches!(format, ExportFormat::RoonDsp) && eq_filter_count > 20 {
            anyhow::bail!(
                "RoonDsp export supports at most 20 PEQ filters per channel; '{channel_name}' has {eq_filter_count}"
            );
        }
        if matches!(format, ExportFormat::RoonDsp) && convolution_count > 1 {
            anyhow::bail!(
                "RoonDsp export supports one convolution impulse per channel; '{channel_name}' has {convolution_count}"
            );
        }
    }

    if matches!(format, ExportFormat::EasyEffects | ExportFormat::Wavelet) && channels.len() > 1 {
        let canonical = serde_json::to_value(&channels[0].1.plugins)?;
        if channels.iter().skip(1).any(|(_, chain)| {
            serde_json::to_value(&chain.plugins).ok().as_ref() != Some(&canonical)
        }) {
            anyhow::bail!(
                "{format:?} export is system-wide and cannot preserve different per-channel DSP chains"
            );
        }
    }
    Ok(())
}

fn validate_pipewire_plugin(
    plugin: &PluginConfigWrapper,
    sample_rate: Option<f64>,
    context: &str,
) -> anyhow::Result<()> {
    let parameters = plugin.parameters.as_object().ok_or_else(|| {
        anyhow::anyhow!("PipeWire export requires an object for parameters on {context}")
    })?;

    match plugin.plugin_type.as_str() {
        "gain" => {
            let gain_db = required_f64(parameters, "gain_db", context)?;
            if !(-150.0..=150.0).contains(&gain_db) {
                anyhow::bail!("PipeWire export requires gain_db between -150 and 150 on {context}");
            }
            optional_bool(parameters, "invert", context)?;
        }
        "delay" => {
            let delay_ms = required_f64(parameters, "delay_ms", context)?;
            if delay_ms < 0.0 {
                anyhow::bail!("PipeWire export requires non-negative delay_ms on {context}");
            }
        }
        "crossover" => {
            let crossover_type = required_str(parameters, "type", context)?;
            if !matches!(
                crossover_type.to_ascii_lowercase().as_str(),
                "lr24"
                    | "linkwitzriley4"
                    | "linkwitz-riley-4"
                    | "lr48"
                    | "linkwitzriley8"
                    | "linkwitz-riley-8"
            ) {
                anyhow::bail!(
                    "PipeWire export supports LR24 and LR48 crossover plugins, not \
                     '{crossover_type}' on {context}"
                );
            }
            let output = required_str(parameters, "output", context)?;
            if !matches!(output, "low" | "high") {
                anyhow::bail!(
                    "PipeWire export requires crossover output 'low' or 'high' on {context}"
                );
            }
            validate_frequency(
                required_f64(parameters, "frequency", context)?,
                sample_rate,
                context,
            )?;
        }
        "eq" => {
            let filters = parameters
                .get("filters")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "PipeWire export requires an array field 'filters' on {context}"
                    )
                })?;
            for (filter_index, filter) in filters.iter().enumerate() {
                let filter_context = format!("{context}, filter #{filter_index}");
                let filter = filter.as_object().ok_or_else(|| {
                    anyhow::anyhow!("PipeWire export requires an object for {filter_context}")
                })?;
                let filter_type = required_str(filter, "filter_type", &filter_context)?;
                super::pipewire::pipewire_filter_label(filter_type)?;
                if filter.contains_key("topology") {
                    anyhow::bail!(
                        "PipeWire export does not support explicit EQ topology on {filter_context}"
                    );
                }
                validate_frequency(
                    required_f64(filter, "freq", &filter_context)?,
                    sample_rate,
                    &filter_context,
                )?;
                let q = required_f64(filter, "q", &filter_context)?;
                if q <= 0.0 {
                    anyhow::bail!("PipeWire export requires q > 0 on {filter_context}");
                }
                required_f64(filter, "db_gain", &filter_context)?;
            }
        }
        "convolution" => {
            let ir_file = required_str(parameters, "ir_file", context)?;
            if ir_file.trim().is_empty() {
                anyhow::bail!("PipeWire export requires a non-empty ir_file on {context}");
            }
        }
        unsupported => {
            anyhow::bail!("PipeWire export does not support plugin type '{unsupported}'")
        }
    }
    Ok(())
}

fn validate_export_input(
    output: &DspChainOutput,
    capabilities: &ExportCapabilities,
    sample_rate: Option<f64>,
    routing_graph: Option<&BassManagementRoutingGraph>,
) -> anyhow::Result<()> {
    if output.channels.is_empty() {
        anyhow::bail!(
            "{:?} export requires at least one channel",
            capabilities.format
        );
    }
    if let Some(sample_rate) = sample_rate
        && (!sample_rate.is_finite() || sample_rate <= 0.0)
    {
        anyhow::bail!(
            "{:?} export requires a positive finite sample rate, got {sample_rate}",
            capabilities.format
        );
    }
    if let Some(sample_rate) = sample_rate
        && (sample_rate.fract() != 0.0 || sample_rate > u32::MAX as f64)
    {
        anyhow::bail!(
            "{:?} export requires an integer sample rate representable as u32, got {sample_rate}",
            capabilities.format
        );
    }

    let routed = routing_graph.is_some_and(|graph| !graph.routes.is_empty());
    let mut normalized_channels = BTreeMap::new();
    let mut channels: Vec<_> = output.channels.iter().collect();
    channels.sort_by(|a, b| a.0.cmp(b.0));

    for (channel_name, chain) in channels {
        if chain.channel != *channel_name {
            anyhow::bail!(
                "{:?} export channel map key '{}' does not match chain channel '{}'",
                capabilities.format,
                channel_name,
                chain.channel
            );
        }

        let normalized = (capabilities.normalize_identifier)(channel_name);
        if let Some(existing) = normalized_channels.insert(normalized.clone(), channel_name) {
            anyhow::bail!(
                "{:?} export channel names '{}' and '{}' both normalize to '{normalized}'",
                capabilities.format,
                existing,
                channel_name
            );
        }

        if !capabilities.supports_driver_branches
            && chain
                .drivers
                .as_ref()
                .is_some_and(|drivers| !drivers.is_empty())
        {
            anyhow::bail!(
                "{:?} export cannot represent active-crossover driver branches for channel \
                 '{channel_name}' without changing them into a serial cascade",
                capabilities.format
            );
        }

        for (plugin_index, plugin) in chain.plugins.iter().enumerate() {
            let context = format!(
                "channel '{channel_name}' plugin #{plugin_index} ('{}')",
                plugin.plugin_type
            );
            if !capabilities
                .supported_plugin_types
                .contains(&plugin.plugin_type.as_str())
            {
                anyhow::bail!(
                    "{:?} export does not support {context}",
                    capabilities.format
                );
            }

            if routed {
                let stage = plugin
                    .parameters
                    .get("room_eq_stage")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{:?} routed export requires room_eq_stage on {context}",
                            capabilities.format
                        )
                    })?;
                if !capabilities.routed_stages.contains(&stage) {
                    anyhow::bail!(
                        "{:?} routed export does not recognize room_eq_stage '{stage}' on {context}",
                        capabilities.format
                    );
                }
            }

            (capabilities.validate_plugin)(plugin, sample_rate, &context)?;
        }
    }

    Ok(())
}

fn validate_camilladsp_global_plugins(
    output: &DspChainOutput,
    routed_expected: bool,
) -> anyhow::Result<()> {
    for (index, plugin) in output.global_plugins.iter().enumerate() {
        let is_bass_management_matrix = plugin.plugin_type == "matrix"
            && plugin
                .parameters
                .get("label")
                .and_then(|label| label.as_str())
                == Some("home_cinema_bass_management");
        if !routed_expected || !is_bass_management_matrix {
            anyhow::bail!(
                "CamillaDsp export does not support global plugin #{index} ('{}')",
                plugin.plugin_type
            );
        }
    }
    Ok(())
}

fn validate_camilladsp_plugin(
    plugin: &PluginConfigWrapper,
    sample_rate: Option<f64>,
    context: &str,
) -> anyhow::Result<()> {
    let parameters = plugin.parameters.as_object().ok_or_else(|| {
        anyhow::anyhow!("CamillaDsp export requires an object for parameters on {context}")
    })?;

    match plugin.plugin_type.as_str() {
        "gain" => {
            let gain_db = required_f64(parameters, "gain_db", context)?;
            if !(-150.0..=150.0).contains(&gain_db) {
                anyhow::bail!(
                    "CamillaDsp export requires gain_db between -150 and 150 on {context}"
                );
            }
            optional_bool(parameters, "invert", context)?;
        }
        "delay" => {
            let delay_ms = required_f64(parameters, "delay_ms", context)?;
            if delay_ms < 0.0 {
                anyhow::bail!("CamillaDsp export requires non-negative delay_ms on {context}");
            }
        }
        "crossover" => {
            let crossover_type = required_str(parameters, "type", context)?;
            let output = required_str(parameters, "output", context)?;
            camilladsp_crossover_filter_type(crossover_type, output)?;
            validate_frequency(
                required_f64(parameters, "frequency", context)?,
                sample_rate,
                context,
            )?;
        }
        "eq" => {
            let filters = parameters
                .get("filters")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "CamillaDsp export requires an array field 'filters' on {context}"
                    )
                })?;
            for (filter_index, filter) in filters.iter().enumerate() {
                let filter_context = format!("{context}, filter #{filter_index}");
                let filter = filter.as_object().ok_or_else(|| {
                    anyhow::anyhow!("CamillaDsp export requires an object for {filter_context}")
                })?;
                let filter_type = required_str(filter, "filter_type", &filter_context)?;
                if let Some(topology) = filter.get("topology") {
                    let topology = topology.as_str().unwrap_or("<non-string>");
                    anyhow::bail!(
                        "CamillaDsp export does not support EQ topology '{topology}' on \
                         {filter_context}"
                    );
                }
                if !matches!(
                    filter_type,
                    "peak"
                        | "lowshelf"
                        | "highshelf"
                        | "lowpass"
                        | "highpass"
                        | "highpassvariableq"
                        | "notch"
                        | "bandpass"
                        | "allpass"
                ) {
                    anyhow::bail!(
                        "CamillaDsp export does not support filter type '{filter_type}' on \
                         {filter_context}"
                    );
                }
                validate_frequency(
                    required_f64(filter, "freq", &filter_context)?,
                    sample_rate,
                    &filter_context,
                )?;
                let q = required_f64(filter, "q", &filter_context)?;
                if q <= 0.0 {
                    anyhow::bail!("CamillaDsp export requires q > 0 on {filter_context}");
                }
                required_f64(filter, "db_gain", &filter_context)?;
            }
        }
        "convolution" => {
            let ir_file = required_str(parameters, "ir_file", context)?;
            if ir_file.trim().is_empty() {
                anyhow::bail!("CamillaDsp export requires a non-empty ir_file on {context}");
            }
        }
        unsupported => {
            anyhow::bail!("CamillaDsp export does not support plugin type '{unsupported}'")
        }
    }
    Ok(())
}

fn validate_camilladsp_routing_graph(
    output: &DspChainOutput,
    graph: &BassManagementRoutingGraph,
    sample_rate: Option<f64>,
) -> anyhow::Result<()> {
    let (input_channels, output_channels) = routed_channel_names(output, graph);
    validate_channel_list(output, &input_channels, "input")?;
    validate_channel_list(output, &output_channels, "output")?;

    let mut destination_has_route = vec![false; output_channels.len()];
    for (route_index, route) in graph.routes.iter().enumerate() {
        let context = format!(
            "route #{route_index} ({} -> {})",
            route.source_channel, route.destination
        );
        validate_route_endpoint(
            route.source_index,
            &route.source_channel,
            &input_channels,
            "source",
            &context,
        )?;
        validate_route_endpoint(
            route.destination_index,
            &route.destination,
            &output_channels,
            "destination",
            &context,
        )?;
        destination_has_route[route.destination_index] = true;

        if route.high_pass_hz.is_some() && route.low_pass_hz.is_some() {
            anyhow::bail!("CamillaDsp export {context} cannot be both high-pass and low-pass");
        }
        if let Some((frequency, output)) = route
            .high_pass_hz
            .map(|frequency| (frequency, "high"))
            .or_else(|| route.low_pass_hz.map(|frequency| (frequency, "low")))
        {
            validate_frequency(frequency, sample_rate, &context)?;
            camilladsp_crossover_filter_type(&route.crossover_type, output)?;
        }
        if !route.delay_ms.is_finite() || route.delay_ms < 0.0 {
            anyhow::bail!("CamillaDsp export requires a non-negative finite delay on {context}");
        }
        if !route.gain_db.is_finite()
            || !route.gain_linear.is_finite()
            || !route.matrix_gain.is_finite()
        {
            anyhow::bail!("CamillaDsp export requires finite gain values on {context}");
        }
    }

    if let Some(index) = destination_has_route
        .iter()
        .position(|has_route| !has_route)
    {
        anyhow::bail!(
            "CamillaDsp export output channel '{}' has no incoming route",
            output_channels[index]
        );
    }
    Ok(())
}

fn validate_channel_list(
    output: &DspChainOutput,
    channels: &[String],
    kind: &str,
) -> anyhow::Result<()> {
    if channels.is_empty() {
        anyhow::bail!("CamillaDsp routed export requires at least one {kind} channel");
    }
    let mut seen = BTreeSet::new();
    for channel in channels {
        if !output.channels.contains_key(channel) {
            anyhow::bail!(
                "CamillaDsp routed export {kind} channel '{channel}' has no channel DSP chain"
            );
        }
        if !seen.insert(channel) {
            anyhow::bail!("CamillaDsp routed export repeats {kind} channel '{channel}'");
        }
    }
    Ok(())
}

fn validate_route_endpoint(
    index: usize,
    name: &str,
    channels: &[String],
    endpoint: &str,
    context: &str,
) -> anyhow::Result<()> {
    let Some(expected) = channels.get(index) else {
        anyhow::bail!(
            "CamillaDsp export {context} has {endpoint} index {index}, but only {} {endpoint} \
             channels exist",
            channels.len()
        );
    };
    if expected != name {
        anyhow::bail!(
            "CamillaDsp export {context} {endpoint} index {index} names '{expected}', not '{name}'"
        );
    }
    Ok(())
}

pub(super) fn routed_channel_names(
    output: &DspChainOutput,
    graph: &BassManagementRoutingGraph,
) -> (Vec<String>, Vec<String>) {
    let input_channels = if graph.input_channels.is_empty() {
        sorted_channels(output)
            .into_iter()
            .map(|(name, _)| name.clone())
            .collect()
    } else {
        graph.input_channels.clone()
    };
    let output_channels = if graph.output_channels.is_empty() {
        input_channels.clone()
    } else {
        graph.output_channels.clone()
    };
    (input_channels, output_channels)
}

fn routed_bass_management_declared(output: &DspChainOutput) -> bool {
    output
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.bass_management.as_ref())
        .and_then(|report| report.routing_graph.as_ref())
        .is_some_and(|graph| !graph.routes.is_empty())
        || output.global_plugins.iter().any(|plugin| {
            plugin.plugin_type == "matrix"
                && plugin
                    .parameters
                    .get("metadata")
                    .and_then(|metadata| metadata.get("routes"))
                    .and_then(|routes| routes.as_array())
                    .is_some_and(|routes| !routes.is_empty())
        })
}

fn required_f64(
    parameters: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    context: &str,
) -> anyhow::Result<f64> {
    let value = parameters
        .get(name)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| {
            anyhow::anyhow!("CamillaDsp export requires numeric field '{name}' on {context}")
        })?;
    if !value.is_finite() {
        anyhow::bail!("CamillaDsp export requires finite field '{name}' on {context}");
    }
    Ok(value)
}

fn required_str<'a>(
    parameters: &'a serde_json::Map<String, serde_json::Value>,
    name: &str,
    context: &str,
) -> anyhow::Result<&'a str> {
    parameters
        .get(name)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("CamillaDsp export requires string field '{name}' on {context}")
        })
}

fn optional_bool(
    parameters: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    context: &str,
) -> anyhow::Result<()> {
    if parameters
        .get(name)
        .is_some_and(|value| !value.is_boolean())
    {
        anyhow::bail!("CamillaDsp export requires boolean field '{name}' on {context}");
    }
    Ok(())
}

fn validate_frequency(
    frequency: f64,
    sample_rate: Option<f64>,
    context: &str,
) -> anyhow::Result<()> {
    if frequency <= 0.0 {
        anyhow::bail!("CamillaDsp export requires a positive frequency on {context}");
    }
    if let Some(sample_rate) = sample_rate
        && frequency >= sample_rate / 2.0
    {
        let nyquist = sample_rate / 2.0;
        anyhow::bail!(
            "CamillaDsp export frequency {frequency} Hz on {context} must be below the \
             {nyquist} Hz Nyquist frequency"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_manifest_rejects_duplicate_definitions() {
        let mut manifest = ExportArtifactManifest::new(ExportFormat::CamillaDsp);
        manifest
            .define_node(ExportNodeKind::Processor, "left_gain")
            .unwrap();
        let error = manifest
            .define_node(ExportNodeKind::Processor, "left_gain")
            .unwrap_err();
        assert!(error.to_string().contains("duplicate processor"));
    }

    #[test]
    fn artifact_manifest_rejects_dangling_references() {
        let mut manifest = ExportArtifactManifest::new(ExportFormat::CamillaDsp);
        manifest.reference_node(
            ExportNodeKind::Processor,
            "left_conv",
            "left channel pipeline",
        );
        let error = manifest.validate().unwrap_err();
        assert!(error.to_string().contains("missing processor 'left_conv'"));
    }

    #[test]
    fn artifact_manifest_rejects_out_of_range_channels() {
        let manifest = ExportArtifactManifest::new(ExportFormat::CamillaDsp);
        let error = manifest
            .reference_channel(2, 2, "left channel pipeline")
            .unwrap_err();
        assert!(error.to_string().contains("only 2 channels"));
    }

    #[test]
    fn artifact_manifest_rejects_unreferenced_definitions() {
        let mut manifest = ExportArtifactManifest::new(ExportFormat::CamillaDsp);
        manifest
            .define_node(ExportNodeKind::Processor, "left_gain")
            .unwrap();
        let error = manifest.validate().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unreferenced processor 'left_gain'")
        );
    }
}
