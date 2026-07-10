use super::super::types::PluginConfigWrapper;
use super::misc::camilladsp_filter_type;
use std::fmt::Write as FmtWrite;

pub(super) fn write_camilladsp_pipeline_filter_step(
    out: &mut String,
    channel_index: usize,
    filter_names: &[String],
) -> anyhow::Result<()> {
    writeln!(out, "- bypassed: null")?;
    writeln!(out, "  channels:")?;
    writeln!(out, "  - {channel_index}")?;
    writeln!(out, "  names:")?;
    for name in filter_names {
        writeln!(out, "  - {name}")?;
    }
    writeln!(out, "  type: Filter")?;
    Ok(())
}

pub(super) fn write_camilladsp_filters_for_plugins(
    out: &mut String,
    prefix: &str,
    plugins: &[PluginConfigWrapper],
    _suffix: &str,
) -> anyhow::Result<()> {
    let mut eq_idx = 0;
    let mut gain_idx = 0;
    let mut delay_idx = 0;
    let mut conv_idx = 0;
    let mut crossover_idx = 0;

    for plugin in plugins {
        match plugin.plugin_type.as_str() {
            "gain" => {
                let gain_db = plugin
                    .parameters
                    .get("gain_db")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
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
                writeln!(out, "  {name}:")?;
                writeln!(out, "    type: Gain")?;
                writeln!(out, "    parameters:")?;
                writeln!(out, "      gain: {gain_db:.2}")?;
                if inverted {
                    writeln!(out, "      inverted: true")?;
                }
                gain_idx += 1;
            }
            "delay" => {
                let delay_ms = plugin
                    .parameters
                    .get("delay_ms")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let name = if delay_idx == 0 {
                    format!("{prefix}_delay")
                } else {
                    format!("{prefix}_delay_{delay_idx}")
                };
                writeln!(out, "  {name}:")?;
                writeln!(out, "    type: Delay")?;
                writeln!(out, "    parameters:")?;
                writeln!(out, "      delay: {delay_ms:.3}")?;
                writeln!(out, "      unit: ms")?;
                delay_idx += 1;
            }
            "crossover" => {
                let crossover_type = plugin
                    .parameters
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("LR24");
                let freq = plugin
                    .parameters
                    .get("frequency")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(80.0);
                let output = plugin
                    .parameters
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("high");
                let name = if crossover_idx == 0 {
                    format!("{prefix}_crossover")
                } else {
                    format!("{prefix}_crossover_{crossover_idx}")
                };
                write_camilladsp_crossover_filter(out, &name, crossover_type, freq, output)?;
                crossover_idx += 1;
            }
            "eq" => {
                if let Some(filters) = plugin.parameters.get("filters").and_then(|v| v.as_array()) {
                    for f in filters {
                        let ft = f
                            .get("filter_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("peak");
                        let freq = f.get("freq").and_then(|v| v.as_f64()).unwrap_or(1000.0);
                        let q = f.get("q").and_then(|v| v.as_f64()).unwrap_or(1.0);
                        let gain = f.get("db_gain").and_then(|v| v.as_f64()).unwrap_or(0.0);

                        writeln!(out, "  {prefix}_peq_{eq_idx}:")?;
                        writeln!(out, "    type: Biquad")?;
                        writeln!(out, "    parameters:")?;
                        writeln!(out, "      type: {}", camilladsp_filter_type(ft))?;
                        writeln!(out, "      freq: {freq:.1}")?;
                        writeln!(out, "      q: {q:.4}")?;
                        match ft {
                            "lowpass" | "highpass" | "notch" | "bandpass" | "allpass" => {}
                            _ => {
                                writeln!(out, "      gain: {gain:.2}")?;
                            }
                        }
                        eq_idx += 1;
                    }
                }
            }
            "convolution" => {
                if let Some(ir_file) = plugin.parameters.get("ir_file").and_then(|v| v.as_str()) {
                    let name = if conv_idx == 0 {
                        format!("{prefix}_conv")
                    } else {
                        format!("{prefix}_conv_{conv_idx}")
                    };
                    writeln!(out, "  {name}:")?;
                    writeln!(out, "    type: Conv")?;
                    writeln!(out, "    parameters:")?;
                    writeln!(out, "      type: Wav")?;
                    writeln!(out, "      filename: {ir_file}")?;
                    conv_idx += 1;
                }
            }
            _ => {} // Skip unsupported plugin types
        }
    }
    Ok(())
}

pub(super) fn write_camilladsp_crossover_filter(
    out: &mut String,
    name: &str,
    crossover_type: &str,
    freq: f64,
    output: &str,
) -> anyhow::Result<()> {
    let (filter_type, order) = camilladsp_crossover_filter_type(crossover_type, output)?;
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
    name: &str,
    delay_ms: f64,
) -> anyhow::Result<()> {
    writeln!(out, "  {name}:")?;
    writeln!(out, "    type: Delay")?;
    writeln!(out, "    parameters:")?;
    writeln!(out, "      delay: {delay_ms:.3}")?;
    writeln!(out, "      unit: ms")?;
    Ok(())
}

fn camilladsp_crossover_filter_type(
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
