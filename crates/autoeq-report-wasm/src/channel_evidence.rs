//! Renderer-neutral display of serialized per-channel acoustic evidence.
//!
//! These builders validate and present values already stored in the report.
//! They do not calculate STI, time-frequency transforms, or decay estimates.

// Rust guideline compliant 2026-02-21

use crate::report_document::{DiagnosticBlock, diagnostic_html};
use crate::report_table::ReportTable;

use serde_json::{Value, json};

const OCTAVES: [f64; 7] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
const MODULATIONS: [f64; 14] = [
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5,
];

struct GridData {
    times: Vec<f64>,
    freqs: Vec<f64>,
    rows: Vec<Vec<f64>>,
}

struct GridValidation {
    min_time: usize,
    max_time: usize,
    min_freq: usize,
    max_freq: usize,
    min_db: f64,
}

struct GridDisplay<'a> {
    title: &'a str,
    surface: bool,
    zmin: f64,
    zmax: f64,
}

/// Renders the serialized speech-transmission result, or its established
/// unavailable explanation when the stored evidence is absent or invalid.
pub fn speech_transmission_html(channel: &Value, label: &str) -> String {
    diagnostic_html(&speech_transmission(channel, label))
}

pub(crate) fn speech_transmission(channel: &Value, label: &str) -> Vec<DiagnosticBlock> {
    let mut blocks = vec![DiagnosticBlock::Heading(format!(
        "Speech Transmission Index — {label}"
    ))];
    let report = channel.get("speech_transmission").unwrap_or(&Value::Null);
    if !valid_sti(report) {
        blocks.push(DiagnosticBlock::Caption(
            "STI unavailable: requires a native measured room impulse response covering all seven octaves (125 Hz–8 kHz), with usable band energy and its complete decay. Synthesized EQ impulses do not supply this result.".into()
        ));
        return blocks;
    }
    let sti = number(report, "sti").unwrap_or_default();
    let rating = if sti < 0.3 {
        "Bad"
    } else if sti < 0.45 {
        "Poor"
    } else if sti < 0.6 {
        "Fair"
    } else if sti < 0.75 {
        "Good"
    } else {
        "Excellent"
    };
    let sample_rate = number(report, "sample_rate_hz").unwrap_or_default();
    let duration = number(report, "duration_s").unwrap_or_default();
    blocks.extend([
        DiagnosticBlock::Emphasis(format!("Full STI (IR-only): {sti:.3} — {rating}")),
        DiagnosticBlock::Paragraph(format!(
            "Indirect method using the IEC 60268-16:2020 full-STI model: 14 modulation frequencies in seven octave bands. Native capture: {} Hz, {duration:.3} s.",
            format_g(sample_rate)
        )),
        DiagnosticBlock::Caption(
            "No operational speech/noise levels, auditory masking or hearing-threshold corrections are applied. Arbitrary IR amplitude does not establish calibrated SPL. This result describes the measured capture, not a post-EQ prediction or IEC-certified instrument result. Verify full decay, measurement SNR and linear time-invariant behavior.".into()
        ),
    ]);
    let columns = std::iter::once("Modulation (Hz)".into())
        .chain(OCTAVES.map(|center| format!("{} Hz", format_g(center))))
        .collect();
    let matrix = report["modulation_transfer"]
        .as_array()
        .expect("validated STI");
    fn cells(row: &Value) -> impl Iterator<Item = String> {
        row.as_array()
            .expect("validated STI row")
            .iter()
            .map(|value| format!("{:.3}", value.as_f64().unwrap_or_default()))
    }
    let mut rows: Vec<Vec<String>> = MODULATIONS
        .into_iter()
        .zip(matrix)
        .map(|(frequency, row)| {
            std::iter::once(format_g(frequency))
                .chain(cells(row))
                .collect()
        })
        .collect();
    rows.push(
        std::iter::once("MTI".into())
            .chain(cells(&report["mti"]))
            .collect(),
    );
    blocks.push(DiagnosticBlock::Matrix(ReportTable {
        title: "Modulation transfer m-values by octave band".into(),
        note: String::new(),
        columns,
        rows,
    }));
    if let Some(warnings) = report.get("warnings").and_then(Value::as_array) {
        blocks.extend(
            warnings
                .iter()
                .filter_map(Value::as_str)
                .map(|warning| DiagnosticBlock::Caption(warning.into())),
        );
    }
    blocks
}

/// Builds the report HTML for the pre-correction waterfall when its complete
/// serialized grid and decay evidence satisfy the stored schema.
pub fn optimization_waterfall_html(waterfall: &Value, decays: &Value) -> Option<String> {
    optimization_waterfall(waterfall, decays).map(|blocks| diagnostic_html(&blocks))
}

pub(crate) fn optimization_waterfall(
    waterfall: &Value,
    decays: &Value,
) -> Option<Vec<DiagnosticBlock>> {
    if !valid_waterfall(waterfall, decays) {
        return None;
    }
    let band = waterfall["valid_band_hz"].as_array()?;
    let lo = band[0].as_f64()?;
    let hi = band[1].as_f64()?;
    let scope = waterfall["scope"].as_str()?;
    let times = numbers(&waterfall["times_ms"])?;
    let freqs = numbers(&waterfall["freqs_hz"])?;
    let rows = matrix(&waterfall["mags_db"])?;
    let resonances = decays["decays"].as_array()?;
    let mut blocks = vec![
        DiagnosticBlock::Heading("Measured room IR waterfall and resonance decay".into()),
        DiagnosticBlock::Paragraph(scope.into()),
        DiagnosticBlock::Paragraph(format!(
            "Supported band: {}–{} Hz. The grid is referenced to its own full-grid peak; colors and levels do not compare absolute output between channels.",
            format_g(lo),
            format_g(hi)
        )),
        DiagnosticBlock::Subheading("Pre-correction".into()),
    ];
    let highlights = worst_resonances(resonances, &freqs);
    let surface = grid_section(
        &freqs,
        &times,
        &rows,
        &highlights,
        GridDisplay {
            title: "Measured room IR waterfall",
            surface: true,
            zmin: -60.0,
            zmax: 0.0,
        },
    );
    blocks.push(DiagnosticBlock::Chart(surface));
    blocks.push(DiagnosticBlock::Table(resonance_table(resonances)));
    if let Some(chart) = resonance_figure(resonances, &freqs, &times, &rows) {
        blocks.push(DiagnosticBlock::Paragraph(
            "Highlighted: up to two longest fitted decays, then strongest 60 ms peak. Curves retain the full-grid peak reference; no fitted intercept is exported, so no extrapolated fit line is invented.".into()
        ));
        blocks.push(DiagnosticBlock::Chart(chart));
    }
    Some(blocks)
}

/// Builds the report HTML for the pre-correction wavelet when the stored
/// measured-IR evidence satisfies its serialized contract.
pub fn optimization_wavelet_html(wavelet: &Value) -> Option<String> {
    optimization_wavelet(wavelet).map(|blocks| diagnostic_html(&blocks))
}

pub(crate) fn optimization_wavelet(wavelet: &Value) -> Option<Vec<DiagnosticBlock>> {
    if !valid_wavelet(wavelet) {
        return None;
    }
    let freqs = numbers(&wavelet["freqs_hz"])?;
    let times = numbers(&wavelet["times_ms"])?;
    let frequency_rows = matrix(&wavelet["mags_db"])?;
    let mut time_major = vec![vec![0.0; freqs.len()]; times.len()];
    for (fi, row) in frequency_rows.iter().enumerate() {
        for (ti, value) in row.iter().enumerate() {
            time_major[ti][fi] = *value;
        }
    }
    let early_count = times
        .iter()
        .filter(|time| **time >= -1.0 && **time <= 15.0)
        .count();
    let section = grid_section(
        &freqs,
        &times,
        &time_major,
        &[],
        GridDisplay {
            title: "Measured room IR three-cycle wavelet heatmap",
            surface: false,
            zmin: -30.0,
            zmax: 0.0,
        },
    );
    Some(vec![
        DiagnosticBlock::Heading("Measured room IR three-cycle wavelet".into()),
        DiagnosticBlock::Paragraph("Blue = −30 dB, red = 0 dB relative to the channel’s own full wavelet-grid peak. Colors do not compare absolute output levels.".into()),
        DiagnosticBlock::Paragraph("Frequency runs left to right; time runs upward. Dense exports use 0.1 ms early-time sampling through 15 ms and 1 ms later sampling; the recorded time coordinates remain authoritative.".into()),
        DiagnosticBlock::Paragraph(format!("Three-cycle wavelet; first 15 ms ({early_count} exported time samples). Log-frequency x-axis, time y-axis. Colours retain the full-grid peak reference. The cyan dashed ridge marks peak energy arrival time per frequency. Display cells reflect the exported sampling; zoom cannot restore omitted samples.")),
        DiagnosticBlock::Chart(section),
    ])
}

fn valid_sti(report: &Value) -> bool {
    if !report.is_object()
        || report["method"] != "iec_60268_16_2020_indirect_ir_only_v1"
        || report["basis"] != "measured_room_ir"
        || number_array(&report["octave_centers_hz"]).as_deref() != Some(&OCTAVES[..])
        || number_array(&report["modulation_frequencies_hz"]).as_deref() != Some(&MODULATIONS[..])
        || !bounded_number(report.get("sti"), 0.0, 1.0)
        || !bounded_number(
            report.get("sample_rate_hz"),
            16_000.0 * 2.0_f64.sqrt() / 0.99,
            192_000.0,
        )
        || !number(report, "duration_s").is_some_and(|v| v.is_finite() && v > 0.0)
    {
        return false;
    }
    let Some(mti) = report["mti"].as_array() else {
        return false;
    };
    let Some(matrix) = report["modulation_transfer"].as_array() else {
        return false;
    };
    mti.len() == 7
        && mti
            .iter()
            .all(|value| bounded_number(Some(value), 0.0, 1.0))
        && matrix.len() == 14
        && matrix.iter().all(|row| {
            row.as_array().is_some_and(|row| {
                row.len() == 7
                    && row
                        .iter()
                        .all(|value| bounded_number(Some(value), 0.0, 1.0))
            })
        })
}

pub(crate) fn valid_waterfall(waterfall: &Value, decays: &Value) -> bool {
    if waterfall.get("basis") != Some(&json!("measured_room_ir"))
        || waterfall.get("method") != Some(&json!("hann_stft_waterfall_v1"))
        || waterfall.get("reference") != Some(&json!("full_grid_peak"))
        || number(waterfall, "window_ms") != Some(32.0)
        || number(waterfall, "hop_ms") != Some(2.0)
        || number(waterfall, "post_ms") != Some(500.0)
        || !valid_band(&waterfall["valid_band_hz"])
    {
        return false;
    }
    let Some(grid) = grid_values(waterfall, "times_ms", "freqs_hz") else {
        return false;
    };
    if !valid_grid(
        &grid,
        GridValidation {
            min_time: 2,
            max_time: 800,
            min_freq: 2,
            max_freq: 2049,
            min_db: -100.01,
        },
    ) || !grid.freqs.iter().all(|freq| {
        *freq
            >= waterfall["valid_band_hz"][0]
                .as_f64()
                .unwrap_or(f64::INFINITY)
            && *freq
                <= waterfall["valid_band_hz"][1]
                    .as_f64()
                    .unwrap_or(f64::NEG_INFINITY)
    }) {
        return false;
    }
    if decays.get("basis") != Some(&json!("measured_room_ir"))
        || decays.get("method") != Some(&json!("hann_stft_waterfall_v1"))
        || decays.get("reference") != Some(&json!("full_grid_peak"))
        || number(decays, "slice_ms") != Some(60.0)
    {
        return false;
    }
    let Some(resonances) = decays.get("decays").and_then(Value::as_array) else {
        return false;
    };
    resonances.len() <= 64
        && resonances.iter().all(|resonance| {
            let Some(frequency) = number(resonance, "freq_hz") else {
                return false;
            };
            let Some(level) = number(resonance, "level_db") else {
                return false;
            };
            resonance.is_object()
                && frequency >= grid.freqs[0]
                && frequency <= *grid.freqs.last().unwrap_or(&0.0)
                && (-100.01..=0.01).contains(&level)
                && resonance.get("decay_time_s").is_some_and(|value| {
                    value.is_null() || value.as_f64().is_some_and(|v| v.is_finite() && v > 0.0)
                })
        })
        && waterfall["scope"]
            .as_str()
            .is_some_and(|scope| !scope.is_empty())
}

fn valid_wavelet(wavelet: &Value) -> bool {
    if wavelet.get("basis") != Some(&json!("measured_room_ir"))
        || wavelet.get("method") != Some(&json!("complex_morlet_three_cycle_v1"))
        || wavelet.get("reference") != Some(&json!("full_grid_peak"))
        || number(wavelet, "cycles") != Some(3.0)
        || ![6.0, 48.0].contains(&number(wavelet, "freqs_per_octave").unwrap_or_default())
        || ![0.1, 1.0].contains(&number(wavelet, "hop_ms").unwrap_or_default())
        || wavelet["display_range_db"] != json!([-30.0, 0.0])
        || !valid_band(&wavelet["valid_band_hz"])
    {
        return false;
    }
    let Some(grid) = grid_values(wavelet, "times_ms", "freqs_hz") else {
        return false;
    };
    grid.times.len() >= 2
        && grid.times.len() <= 800
        && grid.freqs.len() >= 2
        && grid.freqs.len() <= 2049
        && grid.rows.len() == grid.freqs.len()
        && grid.rows.iter().all(|row| row.len() == grid.times.len())
        && grid.times.iter().all(|v| v.is_finite())
        && grid.freqs.iter().all(|v| v.is_finite() && *v > 0.0)
        && grid.times.windows(2).all(|pair| pair[0] < pair[1])
        && grid.freqs.windows(2).all(|pair| pair[0] < pair[1])
        && grid.times[0] >= -5.01
        && *grid.times.last().unwrap_or(&f64::INFINITY) <= 500.01
        && grid
            .rows
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (-30.01..=0.01).contains(v))
        && grid.freqs.iter().all(|freq| {
            *freq
                >= wavelet["valid_band_hz"][0]
                    .as_f64()
                    .unwrap_or(f64::INFINITY)
                && *freq
                    <= wavelet["valid_band_hz"][1]
                        .as_f64()
                        .unwrap_or(f64::NEG_INFINITY)
        })
}

fn valid_band(value: &Value) -> bool {
    value.as_array().is_some_and(|band| {
        band.len() == 2
            && band[0].as_f64().is_some_and(|v| v.is_finite() && v > 0.0)
            && band[1].as_f64().is_some_and(f64::is_finite)
            && band[0].as_f64().unwrap_or_default() < band[1].as_f64().unwrap_or_default()
    })
}

fn grid_values(value: &Value, time_key: &str, frequency_key: &str) -> Option<GridData> {
    Some(GridData {
        times: numbers(&value[time_key])?,
        freqs: numbers(&value[frequency_key])?,
        rows: matrix(&value["mags_db"])?,
    })
}

fn valid_grid(grid: &GridData, limits: GridValidation) -> bool {
    grid.times.len() >= limits.min_time
        && grid.times.len() <= limits.max_time
        && grid.freqs.len() >= limits.min_freq
        && grid.freqs.len() <= limits.max_freq
        && grid.rows.len() == grid.times.len()
        && grid.times.iter().all(|v| v.is_finite())
        && grid.freqs.iter().all(|v| v.is_finite() && *v > 0.0)
        && grid.times.windows(2).all(|pair| pair[0] < pair[1])
        && grid.freqs.windows(2).all(|pair| pair[0] < pair[1])
        && grid.times[0] >= -5.01
        && *grid.times.last().unwrap_or(&f64::INFINITY) <= 500.01
        && grid.rows.iter().all(|row| {
            row.len() == grid.freqs.len()
                && row
                    .iter()
                    .all(|v| v.is_finite() && *v >= limits.min_db && *v <= 0.01)
        })
}

fn grid_section(
    x: &[f64],
    y: &[f64],
    z: &[Vec<f64>],
    highlights: &[(usize, String, String)],
    display: GridDisplay<'_>,
) -> Value {
    let ymax = if display.surface {
        *y.last().unwrap_or(&15.0)
    } else {
        15.0_f64.min(*y.last().unwrap_or(&15.0))
    };
    let series = highlights
        .iter()
        .map(|(_, name, color)| {
            json!({"name":name,"x":[],"y":[],"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0})
        })
        .collect::<Vec<_>>();
    let mut section = json!({
        "kind":"grid",
        "figure":{
            "title":display.title,
            "x":{"label":"Frequency (Hz)","scale":"log","min":x[0],"max":20000.0_f64.min(*x.last().unwrap_or(&20000.0))},
            "y":{"label":"Time from broadband peak (ms)","scale":"linear","min":y[0].max(-1.0),"max":ymax},
            "y2":null,"series":series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
        },
        "grid":{"x":x,"y":y,"z":z,"surface":display.surface,"zmin":display.zmin,"zmax":display.zmax,"highlights":highlights.iter().map(|(index,_,_)| *index).collect::<Vec<_>>()},
        "tab":null
    });
    if display.surface {
        section["grid"]["colormap"] = json!("turbo");
        section["grid"]["show_surface"] = json!(false);
        section["grid"]["show_contours"] = json!(true);
    }
    section
}

fn resonance_table(resonances: &[Value]) -> ReportTable {
    ReportTable {
        title: "Detected 60 ms peaks and fitted 20–200 ms decay".into(),
        note: String::new(),
        columns: [
            "Frequency (Hz)",
            "Level (dB re own grid peak)",
            "Fitted decay (s)",
        ]
        .map(String::from)
        .into(),
        rows: resonances
            .iter()
            .map(|resonance| {
                let frequency = number(resonance, "freq_hz").unwrap_or_default();
                let level = number(resonance, "level_db").unwrap_or_default();
                let decay = resonance["decay_time_s"]
                    .as_f64()
                    .map_or_else(|| "unavailable".to_owned(), format_g);
                vec![
                    format_g_precision(frequency, 3),
                    format!("{level:.2}"),
                    decay,
                ]
            })
            .collect(),
    }
}

fn resonance_figure(
    resonances: &[Value],
    freqs: &[f64],
    times: &[f64],
    rows: &[Vec<f64>],
) -> Option<Value> {
    let selected = worst_resonances(resonances, freqs);
    if selected.is_empty() {
        return None;
    }
    let series = selected
        .iter()
        .map(|(index, name, color)| {
            let name = resonances
                .iter()
                .find(|row| number(row, "freq_hz").is_some_and(|f| format!("{f:.1} Hz") == *name))
                .and_then(|row| {
                    row["decay_time_s"]
                        .as_f64()
                        .map(|decay| format!("{name} — T60 {decay:.3} s"))
                })
                .unwrap_or_else(|| name.clone());
            let y = rows.iter().map(|row| json!(row[*index])).collect::<Vec<_>>();
            json!({"name":name,"x":times,"y":y,"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0})
        })
        .collect::<Vec<_>>();
    let section = json!({"kind":"figure","figure":{"title":"Pre-correction: decay of highlighted resonances","x":{"label":"Time from broadband peak (ms)","scale":"linear","min":0.0,"max":500.0},"y":{"label":"Level relative to full grid peak (dB)","scale":"linear","min":-60.0,"max":0.0},"y2":null,"series":series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true},"tab":null});
    Some(section)
}

fn worst_resonances(resonances: &[Value], freqs: &[f64]) -> Vec<(usize, String, String)> {
    let mut ranked = resonances.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        resonance_rank(right)
            .partial_cmp(&resonance_rank(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
        .into_iter()
        .take(2)
        .filter_map(|resonance| {
            let frequency = number(resonance, "freq_hz")?;
            let index = freqs
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    (*left - frequency)
                        .abs()
                        .partial_cmp(&(*right - frequency).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?
                .0;
            Some((
                index,
                format!("{frequency:.1} Hz"),
                resonance_color(frequency),
            ))
        })
        .collect()
}

fn resonance_rank(resonance: &Value) -> (bool, f64, f64) {
    let decay = resonance["decay_time_s"].as_f64();
    (
        decay.is_some(),
        decay.unwrap_or(0.0),
        number(resonance, "level_db").unwrap_or_default(),
    )
}

fn resonance_color(frequency: f64) -> String {
    let ratio = ((frequency.max(10.0) / 10.0).ln() / 30.0_f64.ln()).clamp(0.0, 1.0);
    let h = 0.5 + ratio / 3.0;
    let s = 0.65;
    let v = 0.8;
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - f64::from(i);
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8
    )
}

pub(crate) fn embedded_figure(section: &Value) -> String {
    let encoded = serde_json::to_string(section).unwrap_or_else(|_| "{}".into());
    format!(
        "<div class=\"report-figure\" data-section=\"{}\"></div>",
        escape_html(&encoded)
    )
}

fn escape_html(input: &str) -> String {
    input
        .chars()
        .map(|character| match character {
            '&' => "&amp;".to_owned(),
            '<' => "&lt;".to_owned(),
            '>' => "&gt;".to_owned(),
            '"' => "&quot;".to_owned(),
            '\'' => "&#x27;".to_owned(),
            _ => character.to_string(),
        })
        .collect()
}

fn numbers(value: &Value) -> Option<Vec<f64>> {
    value
        .as_array()?
        .iter()
        .map(|v| v.as_f64().filter(|n| n.is_finite()))
        .collect()
}

fn matrix(value: &Value) -> Option<Vec<Vec<f64>>> {
    value.as_array()?.iter().map(numbers).collect()
}

fn number_array(value: &Value) -> Option<Vec<f64>> {
    numbers(value)
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64().filter(|n| n.is_finite())
}

fn bounded_number(value: Option<&Value>, lo: f64, hi: f64) -> bool {
    value
        .and_then(Value::as_f64)
        .is_some_and(|n| n.is_finite() && n >= lo && n <= hi)
}

fn format_g(value: f64) -> String {
    format_g_precision(value, 6)
}

fn format_g_precision(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let rendered = format!("{value:.digits$e}", digits = precision.saturating_sub(1));
        let (mantissa, exponent) = rendered.split_once('e').unwrap_or((&rendered, "0"));
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        let exponent = exponent.parse::<i32>().unwrap_or_default();
        return format!("{mantissa}e{exponent:+03}");
    }
    let decimals = usize::try_from((precision as i32 - 1 - exponent).max(0)).unwrap_or_default();
    let rendered = format!("{value:.decimals$}");
    if rendered.contains('.') {
        rendered
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::{optimization_waterfall_html, optimization_wavelet_html, speech_transmission_html};
    use serde_json::{Value, json};

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../tests/fixtures/channel_evidence_python_goldens.json"
        ))
        .unwrap()
    }

    fn normalize_embedded_figure(html: &str) -> (String, Vec<Value>) {
        let mut normalized = String::new();
        let mut figures = Vec::new();
        let mut rest = html;
        while let Some(start) = rest.find("<div class=\"report-figure\" data-section=\"") {
            let (prefix, after_start) = rest.split_at(start);
            normalized.push_str(prefix);
            let marker = "<div class=\"report-figure\" data-section=\"";
            let body = &after_start[marker.len()..];
            let end = body.find("\"></div>").unwrap();
            let encoded = &body[..end];
            let decoded = encoded
                .replace("&quot;", "\"")
                .replace("&#x27;", "'")
                .replace("&gt;", ">")
                .replace("&lt;", "<")
                .replace("&amp;", "&");
            figures.push(normalize_numbers(serde_json::from_str(&decoded).unwrap()));
            normalized.push_str("<div class=\"report-figure\" data-section=\"FIGURE\"></div>");
            rest = &body[end + 8..];
        }
        normalized.push_str(rest);
        (normalized, figures)
    }

    fn normalize_numbers(value: Value) -> Value {
        match value {
            Value::Number(number) => number
                .as_f64()
                .and_then(serde_json::Number::from_f64)
                .map_or(Value::Null, Value::Number),
            Value::Array(values) => {
                Value::Array(values.into_iter().map(normalize_numbers).collect())
            }
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, normalize_numbers(value)))
                    .collect(),
            ),
            other => other,
        }
    }

    #[test]
    fn serialized_channel_evidence_matches_python_golden() {
        let fixture = fixture();
        let input = &fixture["input"];
        let expected = &fixture["expected"];
        assert_eq!(
            speech_transmission_html(input, input["label"].as_str().unwrap()),
            expected["speech_transmission_html"].as_str().unwrap()
        );

        let waterfall =
            optimization_waterfall_html(&input["waterfall"], &input["resonance_decays"]).unwrap();
        let expected_waterfall = expected["optimization_waterfall_html"].as_str().unwrap();
        let (waterfall_text, waterfall_figures) = normalize_embedded_figure(&waterfall);
        let (expected_waterfall_text, expected_waterfall_figures) =
            normalize_embedded_figure(expected_waterfall);
        assert_eq!(waterfall_text, expected_waterfall_text);
        assert_eq!(waterfall_figures, expected_waterfall_figures);

        let wavelet = optimization_wavelet_html(&input["wavelet"]).unwrap();
        let expected_wavelet = expected["optimization_wavelet_html"].as_str().unwrap();
        let (wavelet_text, wavelet_figures) = normalize_embedded_figure(&wavelet);
        let (expected_wavelet_text, expected_wavelet_figures) =
            normalize_embedded_figure(expected_wavelet);
        assert_eq!(wavelet_text, expected_wavelet_text);
        assert_eq!(wavelet_figures, expected_wavelet_figures);
    }

    #[test]
    fn invalid_serialized_evidence_keeps_established_unavailable_behavior() {
        assert!(optimization_waterfall_html(&json!({"method":"wrong"}), &Value::Null).is_none());
        assert!(optimization_wavelet_html(&json!({"method":"wrong"})).is_none());
        let html = speech_transmission_html(&json!({}), "L <R>");
        assert!(html.starts_with("<h3>Speech Transmission Index — L &lt;R&gt;</h3>"));
        assert!(html.contains("STI unavailable: requires a native measured room impulse response"));
    }
}
