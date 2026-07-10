use super::super::types::PluginConfigWrapper;
use super::conformance::{ExportArtifactManifest, ExportNodeKind};
use super::misc::camilladsp_filter_type;
use std::fmt::Write as FmtWrite;

pub(super) fn write_camilladsp_pipeline_filter_step(
    out: &mut String,
    manifest: &mut ExportArtifactManifest,
    channel_index: usize,
    channel_count: usize,
    filter_names: &[String],
    context: &str,
) -> anyhow::Result<()> {
    manifest.reference_channel(channel_index, channel_count, context)?;
    writeln!(out, "- bypassed: null")?;
    writeln!(out, "  channels:")?;
    writeln!(out, "  - {channel_index}")?;
    writeln!(out, "  names:")?;
    for name in filter_names {
        manifest.reference_node(ExportNodeKind::Processor, name, context);
        writeln!(out, "  - {name}")?;
    }
    writeln!(out, "  type: Filter")?;
    Ok(())
}

pub(super) fn write_camilladsp_filters_for_plugins(
    out: &mut String,
    manifest: &mut ExportArtifactManifest,
    prefix: &str,
    plugins: &[PluginConfigWrapper],
) -> anyhow::Result<Vec<String>> {
    let mut names = Vec::new();
    let mut eq_idx = 0;
    let mut gain_idx = 0;
    let mut delay_idx = 0;
    let mut conv_idx = 0;
    let mut crossover_idx = 0;

    for plugin in plugins {
        match plugin.plugin_type.as_str() {
            "gain" => {
                let gain_db = required_f64(plugin, "gain_db")?;
                let inverted = plugin
                    .parameters
                    .get("invert")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let name = if gain_idx == 0 {
                    format!("{prefix}_gain")
                } else {
                    format!("{prefix}_gain_{gain_idx}")
                };
                manifest.define_node(ExportNodeKind::Processor, &name)?;
                writeln!(out, "  {name}:")?;
                writeln!(out, "    type: Gain")?;
                writeln!(out, "    parameters:")?;
                writeln!(out, "      gain: {gain_db:.2}")?;
                if inverted {
                    writeln!(out, "      inverted: true")?;
                }
                names.push(name);
                gain_idx += 1;
            }
            "delay" => {
                let delay_ms = required_f64(plugin, "delay_ms")?;
                let name = if delay_idx == 0 {
                    format!("{prefix}_delay")
                } else {
                    format!("{prefix}_delay_{delay_idx}")
                };
                manifest.define_node(ExportNodeKind::Processor, &name)?;
                writeln!(out, "  {name}:")?;
                writeln!(out, "    type: Delay")?;
                writeln!(out, "    parameters:")?;
                writeln!(out, "      delay: {delay_ms:.3}")?;
                writeln!(out, "      unit: ms")?;
                names.push(name);
                delay_idx += 1;
            }
            "crossover" => {
                let crossover_type = required_str(plugin, "type")?;
                let freq = required_f64(plugin, "frequency")?;
                let output = required_str(plugin, "output")?;
                let name = if crossover_idx == 0 {
                    format!("{prefix}_crossover")
                } else {
                    format!("{prefix}_crossover_{crossover_idx}")
                };
                write_camilladsp_crossover_filter(
                    out,
                    manifest,
                    &name,
                    crossover_type,
                    freq,
                    output,
                )?;
                names.push(name);
                crossover_idx += 1;
            }
            "eq" => {
                let filters = plugin
                    .parameters
                    .get("filters")
                    .and_then(|value| value.as_array())
                    .ok_or_else(|| anyhow::anyhow!("eq plugin is missing array field 'filters'"))?;
                for filter in filters {
                    let ft = filter
                        .get("filter_type")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| anyhow::anyhow!("eq filter is missing 'filter_type'"))?;
                    let freq = required_filter_f64(filter, "freq")?;
                    let q = required_filter_f64(filter, "q")?;
                    let gain = required_filter_f64(filter, "db_gain")?;
                    let name = format!("{prefix}_peq_{eq_idx}");

                    manifest.define_node(ExportNodeKind::Processor, &name)?;
                    writeln!(out, "  {name}:")?;
                    writeln!(out, "    type: Biquad")?;
                    writeln!(out, "    parameters:")?;
                    writeln!(out, "      type: {}", camilladsp_filter_type(ft))?;
                    writeln!(out, "      freq: {freq:.1}")?;
                    writeln!(out, "      q: {q:.4}")?;
                    match ft {
                        "lowpass" | "highpass" | "highpassvariableq" | "notch" | "bandpass"
                        | "allpass" => {}
                        _ => {
                            writeln!(out, "      gain: {gain:.2}")?;
                        }
                    }
                    names.push(name);
                    eq_idx += 1;
                }
            }
            "convolution" => {
                let ir_file = required_str(plugin, "ir_file")?;
                let name = if conv_idx == 0 {
                    format!("{prefix}_conv")
                } else {
                    format!("{prefix}_conv_{conv_idx}")
                };
                manifest.define_node(ExportNodeKind::Processor, &name)?;
                writeln!(out, "  {name}:")?;
                writeln!(out, "    type: Conv")?;
                writeln!(out, "    parameters:")?;
                writeln!(out, "      type: Wav")?;
                writeln!(out, "      filename: {}", serde_json::to_string(ir_file)?)?;
                names.push(name);
                conv_idx += 1;
            }
            unsupported => anyhow::bail!("unsupported CamillaDSP plugin type '{unsupported}'"),
        }
    }
    Ok(names)
}

pub(super) fn write_camilladsp_crossover_filter(
    out: &mut String,
    manifest: &mut ExportArtifactManifest,
    name: &str,
    crossover_type: &str,
    freq: f64,
    output: &str,
) -> anyhow::Result<()> {
    let (filter_type, order) = camilladsp_crossover_filter_type(crossover_type, output)?;
    manifest.define_node(ExportNodeKind::Processor, name)?;
    writeln!(out, "  {name}:")?;
    writeln!(out, "    type: BiquadCombo")?;
    writeln!(out, "    parameters:")?;
    writeln!(out, "      type: {filter_type}")?;
    writeln!(out, "      freq: {freq:.6}")?;
    writeln!(out, "      order: {order}")?;
    Ok(())
}

pub(super) fn write_camilladsp_delay_filter(
    out: &mut String,
    manifest: &mut ExportArtifactManifest,
    name: &str,
    delay_ms: f64,
) -> anyhow::Result<()> {
    manifest.define_node(ExportNodeKind::Processor, name)?;
    writeln!(out, "  {name}:")?;
    writeln!(out, "    type: Delay")?;
    writeln!(out, "    parameters:")?;
    writeln!(out, "      delay: {delay_ms:.3}")?;
    writeln!(out, "      unit: ms")?;
    Ok(())
}

pub(super) fn camilladsp_crossover_filter_type(
    crossover_type: &str,
    output: &str,
) -> anyhow::Result<(&'static str, u32)> {
    let suffix = match output.to_ascii_lowercase().as_str() {
        "low" | "lowpass" => "Lowpass",
        "high" | "highpass" => "Highpass",
        other => anyhow::bail!("Unsupported crossover output '{other}'"),
    };
    let (family, order) = match crossover_type.to_ascii_lowercase().as_str() {
        "lr24" | "lr4" | "linkwitzriley24" | "linkwitz-riley24" => ("LinkwitzRiley", 4),
        "lr48" | "lr8" | "linkwitzriley48" | "linkwitz-riley48" => ("LinkwitzRiley", 8),
        "butterworth12" | "bw12" => ("Butterworth", 2),
        "butterworth24" | "bw24" => ("Butterworth", 4),
        "linearphase" | "linear_phase" | "linear-phase" | "linearphasefir" | "fir" | "lpfir" => {
            anyhow::bail!("CamillaDSP external export does not support FIR crossover plugins yet")
        }
        other => anyhow::bail!("Unsupported crossover type '{other}' for CamillaDSP export"),
    };
    Ok((
        match (family, suffix) {
            ("LinkwitzRiley", "Lowpass") => "LinkwitzRileyLowpass",
            ("LinkwitzRiley", "Highpass") => "LinkwitzRileyHighpass",
            ("Butterworth", "Lowpass") => "ButterworthLowpass",
            ("Butterworth", "Highpass") => "ButterworthHighpass",
            _ => unreachable!(),
        },
        order,
    ))
}

fn required_f64(plugin: &PluginConfigWrapper, name: &str) -> anyhow::Result<f64> {
    plugin
        .parameters
        .get(name)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} plugin is missing numeric field '{name}'",
                plugin.plugin_type
            )
        })
}

fn required_str<'a>(plugin: &'a PluginConfigWrapper, name: &str) -> anyhow::Result<&'a str> {
    plugin
        .parameters
        .get(name)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} plugin is missing string field '{name}'",
                plugin.plugin_type
            )
        })
}

fn required_filter_f64(filter: &serde_json::Value, name: &str) -> anyhow::Result<f64> {
    filter
        .get(name)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| anyhow::anyhow!("eq filter is missing numeric field '{name}'"))
}
