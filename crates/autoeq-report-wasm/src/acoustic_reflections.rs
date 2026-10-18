//! Measured early-reflection evidence and early-to-late report views.
//!
//! This module presents reflection candidates already emitted from a measured
//! room impulse response. It validates their declared basis, method, band,
//! threshold, time ordering, and geometry before exposing values or plots.

// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};

const PENDING_REFLECTIONS: &str = "<p class=\"epa-footer\">1–8 kHz early reflections: pending roomeq field early_reflections from a measured room IR.</p>";
const PENDING_CELL_COLOR: &str =
    "<td style=\"background:#eee;color:#888\" title=\"needs roomeq field: ";

/// Builds validated measured-reflection summaries, tables, and report figures.
///
/// The result contains the validated source report, a typed level summary,
/// early-to-late ratio evidence, HTML cells, explanatory HTML, and standard
/// report figure sections. Invalid or missing evidence produces pending
/// values and no reflection figures. Response curves only add context when
/// their saved frequency and level arrays are valid; they are never used to
/// infer reflection candidates.
///
/// # Examples
///
/// ```
/// use autoeq_report_wasm::acoustic_reflections::reflection_report;
/// use serde_json::json;
///
/// let report = reflection_report(&json!({}), "Left", Some("L"));
/// assert!(report["validated_report"].is_null());
/// assert!(report["figures"].as_array().unwrap().is_empty());
/// ```
///
/// # Errors
///
/// Invalid and incomplete evidence is represented by null metrics and pending
/// HTML; it is not returned as a processing error.
///
/// # Panics
///
/// This function does not panic for malformed report values.
pub fn reflection_report(channel: &Value, label: &str, tab: Option<&str>) -> Value {
    let validated = validated_early_reflections(channel);
    let level = validated.and_then(reflection_level);
    let ratio = early_late_ratio_db(channel);
    json!({
        "validated_report": validated.cloned(),
        "early_reflection_level_db": level.map(|level| json!({
            "level_db": level.level_db,
            "upper_bound": level.upper_bound
        })),
        "early_reflection_level_cell": reflection_level_cell(level),
        "early_late_ratio_db": ratio,
        "early_late_ratio_cell": early_late_ratio_cell(ratio),
        "html": reflections_html(validated, label),
        "figures": reflection_figures(channel, validated, label, tab),
    })
}

#[derive(Debug, Clone, Copy)]
struct ReflectionLevel {
    level_db: f64,
    upper_bound: bool,
}

fn validated_early_reflections(channel: &Value) -> Option<&Value> {
    let report = channel.get("early_reflections")?;
    let bands = report.get("band_hz")?.as_array()?;
    if report.get("basis")?.as_str()? != "measured_room_ir"
        || report.get("method")?.as_str()? != "bandlimited_early_reflection_table_v1"
        || bands.len() != 2
        || number(bands.first())? != 1000.0
        || number(bands.get(1))? != 8000.0
        || number(report.get("threshold_dbfs"))? != -15.0
        || report
            .get("direct_reference")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        return None;
    }
    for side in ["pre", "post"] {
        let events = report.get(side)?.as_array()?;
        if events.len() > 64 {
            return None;
        }
        let mut last_time = 0.0;
        for event in events {
            if !event.is_object() {
                return None;
            }
            let gain = number(event.get("gain_dbfs"))?;
            let time = number(event.get("time_ms"))?;
            let distance = number(event.get("distance_cm"))?;
            let dip = number(event.get("first_dip_hz"))?;
            if !(-15.0..=0.0).contains(&gain)
                || !(time > last_time && time <= 15.0)
                || distance <= 0.0
                || dip <= 0.0
                || (distance - 34.3 * time).abs() > 0.1_f64.max(0.02 * 34.3 * time)
                || (dip - 500.0 / time).abs() > 0.5_f64.max(0.02 * 500.0 / time)
                || !valid_optional_nonnegative_number(event.get("ripple_db"))
            {
                return None;
            }
            last_time = time;
        }
    }
    Some(report)
}

fn reflection_level(report: &Value) -> Option<ReflectionLevel> {
    let post = report.get("post")?.as_array()?;
    if post.is_empty() {
        return Some(ReflectionLevel {
            level_db: -15.0,
            upper_bound: true,
        });
    }
    let level_db = post
        .iter()
        .filter_map(|event| number(event.get("gain_dbfs")))
        .max_by(f64::total_cmp)?;
    Some(ReflectionLevel {
        level_db,
        upper_bound: false,
    })
}

fn reflections_html(report: Option<&Value>, label: &str) -> String {
    let Some(report) = report else {
        return PENDING_REFLECTIONS.to_string();
    };
    let mut parts = vec![
        format!(
            "<div class=\"filters-section\"><h3>1–8 kHz early reflections: {}</h3>",
            crate::report_helpers::escape_html(label)
        ),
        format!(
            "<p>Candidate search: first 15 ms after the filtered direct peak, at or above −15 dB relative to direct sound. Before uses the measured room IR; after is predicted through the saved DSP chain, not a new measurement. Reference: {}.</p>",
            crate::report_helpers::escape_html(
                report.get("direct_reference").and_then(Value::as_str).unwrap_or("")
            )
        ),
        "<p>Candidate peaks are drawn over the measured 1–8 kHz energy-time envelope when the saved impulse response is available. Two-path first-dip and ripple values are estimates, not confirmed acoustic nulls.</p>".to_string(),
    ];
    for (side, title) in [("pre", "Before"), ("post", "Predicted after EQ")] {
        let events = array(report.get(side));
        parts.push(format!("<h4>{title} ({} candidates)</h4>", events.len()));
        if events.is_empty() {
            parts.push("<p>No candidates were emitted at or above −15 dB within the 15 ms search window. This does not mean the room has no reflections.</p>".to_string());
            continue;
        }
        parts.push("<table class=\"epa-table\"><thead><tr><th>Reflection</th><th>Gain (dB re direct)</th><th>Time (ms)</th><th>Extra path (cm)</th><th>First dip (Hz)</th><th>Comb ripple (dB p-p)</th></tr></thead><tbody>".to_string());
        for (index, event) in events.iter().enumerate() {
            let ripple = event.get("ripple_db").filter(|value| !value.is_null());
            let ripple_text = ripple
                .and_then(|value| number(Some(value)))
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "unbounded".to_string());
            let gain = number(event.get("gain_dbfs")).unwrap_or(0.0);
            let time = number(event.get("time_ms")).unwrap_or(0.0);
            let distance = number(event.get("distance_cm")).unwrap_or(0.0);
            let dip = number(event.get("first_dip_hz")).unwrap_or(0.0);
            parts.push(format!(
                "<tr><td>{}</td><td>{gain:.2}</td><td>{time:.2}</td><td>{distance:.1}</td><td>{dip:.1}</td><td>{ripple_text}</td></tr>",
                index + 1
            ));
        }
        parts.push("</tbody></table>".to_string());
    }
    parts.push("</div>".to_string());
    parts.concat()
}

fn reflection_level_cell(level: Option<ReflectionLevel>) -> String {
    let Some(level) = level else {
        return pending_cell("early_reflections");
    };
    let color = if level.level_db < -10.0 {
        "#2ecc71"
    } else if level.level_db <= -3.0 {
        "#f1c40f"
    } else {
        "#e74c3c"
    };
    let display = if level.upper_bound {
        format!("≤ {:.1}", level.level_db)
    } else {
        format!("{:+.1}", level.level_db)
    };
    format!(
        "<td style=\"background:{color}33\" title=\"strongest emitted post-correction 1–8 kHz reflection, dB relative to direct\">{display}</td>"
    )
}

fn early_late_ratio_db(channel: &Value) -> Option<f64> {
    let report = channel.get("early_late_curves")?;
    if report.get("method")?.as_str()? != "incoherent_band_energy"
        || report.get("reference")?.as_str()? != "full_peak_band"
        || report.get("smoothing")?.as_str()? != "third_octave"
        || number(report.get("split_ms"))? != 20.0
    {
        return None;
    }
    let full = report.get("full")?;
    let early = report.get("early")?;
    let late = report.get("late")?;
    if !full.is_object() || !early.is_object() || !late.is_object() {
        return None;
    }
    let frequencies = finite_positive_values(full.get("freq"))?;
    if frequencies.len() < 2
        || frequencies[0] > 1000.0
        || *frequencies.last()? < 8000.0
        || frequencies.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let early_frequencies = finite_positive_values(early.get("freq"))?;
    let late_frequencies = finite_positive_values(late.get("freq"))?;
    if early_frequencies != frequencies || late_frequencies != frequencies {
        return None;
    }
    let full_levels = finite_values(full.get("spl"))?;
    let early_levels = finite_values(early.get("spl"))?;
    let late_levels = finite_values(late.get("spl"))?;
    if full_levels.len() != frequencies.len()
        || early_levels.len() != frequencies.len()
        || late_levels.len() != frequencies.len()
    {
        return None;
    }
    let count = frequencies
        .iter()
        .filter(|frequency| (1000.0..=8000.0).contains(*frequency))
        .count();
    if count < 2 {
        return None;
    }
    let mut scaled_sum = 0.0;
    let mut compensation = 0.0;
    for ((frequency, early), late) in frequencies.iter().zip(early_levels).zip(late_levels) {
        if (1000.0..=8000.0).contains(frequency) {
            let difference = early - late;
            if !difference.is_finite() {
                return None;
            }
            let value = difference / count as f64;
            let adjusted = value - compensation;
            let next = scaled_sum + adjusted;
            compensation = (next - scaled_sum) - adjusted;
            scaled_sum = next;
        }
    }
    scaled_sum.is_finite().then_some(scaled_sum)
}

fn early_late_ratio_cell(ratio: Option<f64>) -> String {
    let Some(ratio) = ratio else {
        return pending_cell("early_late_curves");
    };
    let color = if ratio > 3.0 {
        "#2ecc71"
    } else if ratio >= 0.0 {
        "#f1c40f"
    } else {
        "#e74c3c"
    };
    format!(
        "<td style=\"background:{color}33\" title=\"arithmetic mean of early minus late dB at emitted 1–8 kHz third-octave centers\">{ratio:+.1}</td>"
    )
}

fn reflection_figures(
    channel: &Value,
    report: Option<&Value>,
    label: &str,
    tab: Option<&str>,
) -> Vec<Value> {
    let Some(report) = report else {
        return Vec::new();
    };
    let mut figures = Vec::new();
    for (side, title, color, waveform_key) in [
        ("pre", "Before", "#2864b4", "pre_ir"),
        ("post", "Predicted after EQ", "#c56816", "post_ir"),
    ] {
        let events = array(report.get(side));
        if events.is_empty() {
            continue;
        }
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut annotations = Vec::new();
        for (index, event) in events.iter().enumerate() {
            let time = number(event.get("time_ms")).unwrap_or(0.0);
            let gain = number(event.get("gain_dbfs")).unwrap_or(0.0);
            xs.extend([time, time, time]);
            ys.extend([Some(-40.0), Some(gain), None]);
            annotations.push(json!({"x": time, "y": gain + 1.0, "text": (index + 1).to_string()}));
        }
        let mut figure_series = Vec::new();
        if let Some(envelope) = etc_series(channel, waveform_key) {
            figure_series.push(envelope);
        }
        figure_series.push(series(title, xs, ys, Some(color), "solid"));
        figure_series.push(series(
            "Direct reference",
            vec![0.0, 0.0],
            vec![Some(-40.0), Some(0.0)],
            Some("#555555"),
            "solid",
        ));
        figure_series.push(series(
            "Detection threshold (−15 dB)",
            vec![0.0, 15.0],
            vec![Some(-15.0), Some(-15.0)],
            Some("#888888"),
            "dash",
        ));
        figures.push(figure_section(
            format!("{label}: {title} — early-reflection candidates over the 1–8 kHz envelope"),
            axis(
                "Time after direct peak (ms)",
                "linear",
                Some(0.0),
                Some(15.0),
            ),
            axis(
                "Level relative to direct sound (dB)",
                "linear",
                Some(-40.0),
                Some(3.0),
            ),
            figure_series,
            annotations,
            tab,
        ));
    }

    let mut curves = Vec::new();
    for (key, name, color) in [
        ("initial_curve", "Measured before EQ", "#2864b4"),
        ("final_curve", "Predicted after EQ", "#c56816"),
    ] {
        if let Some((frequencies, levels)) = valid_curve(channel.get(key)) {
            curves.push(series(
                name,
                frequencies,
                levels.into_iter().map(Some).collect(),
                Some(color),
                "solid",
            ));
        }
    }
    if !curves.is_empty() {
        let high = curves
            .iter()
            .flat_map(|item| item["y"].as_array().into_iter().flatten())
            .filter_map(Value::as_f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let top = (high / 5.0).ceil() * 5.0;
        let bottom = top - 50.0;
        if top.is_finite() && bottom.is_finite() {
            for (index, event) in array(report.get("post")).iter().enumerate() {
                let dip = number(event.get("first_dip_hz")).unwrap_or(0.0);
                curves.push(series(
                    &format!("Candidate {}: {dip:.1} Hz (estimated dip)", index + 1),
                    vec![dip, dip],
                    vec![Some(bottom), Some(top)],
                    Some("#888888"),
                    "dot",
                ));
            }
            figures.push(figure_section(
                format!("{label}: response and estimated reflection dips (not measured nulls)"),
                axis("Frequency (Hz)", "log", Some(20.0), Some(20000.0)),
                axis("SPL (dB)", "linear", Some(bottom), Some(top)),
                curves,
                Vec::new(),
                tab,
            ));
        }
    }
    figures
}

fn figure_section(
    title: String,
    x: Value,
    y: Value,
    series: Vec<Value>,
    annotations: Vec<Value>,
    tab: Option<&str>,
) -> Value {
    json!({
        "kind": "figure",
        "figure": {
            "title": title,
            "x": x,
            "y": y,
            "y2": null,
            "series": series,
            "hlines": [],
            "vlines": [],
            "xranges": [],
            "annotations": annotations,
            "legend": true
        },
        "tab": tab
    })
}

fn axis(label: &str, scale: &str, min: Option<f64>, max: Option<f64>) -> Value {
    json!({"label": label, "scale": scale, "min": min, "max": max})
}

/// Build the 1–8 kHz energy-time envelope of a saved IR waveform.
///
/// Display context for the candidate stems: FFT bandpass (half-octave
/// raised-cosine edges), Hilbert envelope, 1 ms smoothing, level relative
/// to the bandpassed direct peak. Returns `None` when the waveform is
/// missing, irregular, too short, or sampled too slowly for the band.
fn etc_series(channel: &Value, waveform_key: &str) -> Option<Value> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let waveform = channel.get(waveform_key)?.as_object()?;
    let time_ms = finite_values(waveform.get("time_ms"))?;
    let amplitude = finite_values(waveform.get("amplitude"))?;
    let count = time_ms.len();
    if count < 256 || amplitude.len() != count {
        return None;
    }
    let step_ms = (time_ms[count - 1] - time_ms[0]) / (count - 1) as f64;
    if !step_ms.is_finite() || step_ms <= 0.0 {
        return None;
    }
    let sample_rate = 1000.0 / step_ms;
    if sample_rate <= 16_000.0 {
        return None;
    }
    let gain = |frequency: f64| -> f64 {
        let smoothstep = |edge0: f64, edge1: f64, x: f64| {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        smoothstep(700.0, 1000.0, frequency) * (1.0 - smoothstep(8000.0, 11000.0, frequency))
    };
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(count);
    let inverse = planner.plan_fft_inverse(count);
    let mut spectrum: Vec<Complex<f64>> = amplitude
        .iter()
        .map(|&sample| Complex::new(sample, 0.0))
        .collect();
    forward.process(&mut spectrum);
    let mut bandpassed = spectrum.clone();
    for (index, value) in bandpassed.iter_mut().enumerate() {
        let frequency = (index.min(count - index)) as f64 * sample_rate / count as f64;
        *value *= gain(frequency);
    }
    let mut time_domain = bandpassed.clone();
    inverse.process(&mut time_domain);
    let scale = 1.0 / count as f64;
    let bandpassed: Vec<f64> = time_domain
        .iter()
        .map(|value| value.re * scale)
        .collect();
    // Analytic signal: keep DC/Nyquist, double positive frequencies.
    for (index, value) in spectrum.iter_mut().enumerate() {
        let frequency_index = index.min(count - index);
        let factor = if index == 0 || (count % 2 == 0 && index == count / 2) {
            1.0
        } else if index == frequency_index {
            2.0
        } else {
            0.0
        };
        *value *= gain((frequency_index as f64 * sample_rate) / count as f64) * factor;
    }
    inverse.process(&mut spectrum);
    let envelope: Vec<f64> = spectrum
        .iter()
        .map(|value| (value.re * scale).hypot(value.im * scale))
        .collect();
    let direct = envelope
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)?;
    let direct_window = ((sample_rate / 1000.0).round() as usize).max(1);
    let direct_peak = bandpassed[direct.saturating_sub(direct_window)
        ..=(direct + direct_window).min(count - 1)]
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f64, f64::max);
    if direct_peak <= 0.0 {
        return None;
    }
    // 1 ms boxcar smoothing of the envelope, via prefix sums.
    let half = direct_window / 2;
    let mut prefix = vec![0.0; count + 1];
    for (index, &value) in envelope.iter().enumerate() {
        prefix[index + 1] = prefix[index] + value;
    }
    let t_direct = time_ms[direct];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (offset, &time) in time_ms[direct..].iter().enumerate() {
        let index = direct + offset;
        let x = time - t_direct;
        if x > 15.0 {
            break;
        }
        let a = index.saturating_sub(half);
        let b = (index + half + 1).min(count);
        let smoothed = (prefix[b] - prefix[a]) / (b - a) as f64;
        xs.push(x);
        ys.push(Some((20.0 * (smoothed / direct_peak).log10()).clamp(-40.0, 3.0)));
    }
    if xs.len() < 16 || *xs.last()? < 5.0 {
        return None;
    }
    // Bound the display payload; the shape is smooth by construction.
    let stride = (xs.len() / 600).max(1);
    let xs: Vec<f64> = xs.iter().step_by(stride).copied().collect();
    let ys: Vec<Option<f64>> = ys.iter().step_by(stride).copied().collect();
    Some(series("1–8 kHz envelope", xs, ys, Some("#a0a7ad"), "solid"))
}

fn series(name: &str, x: Vec<f64>, y: Vec<Option<f64>>, color: Option<&str>, dash: &str) -> Value {
    json!({
        "name": name,
        "x": x,
        "y": y,
        "color": color,
        "width": 2.0,
        "dash": dash,
        "visible": true,
        "y_axis": 0
    })
}

fn valid_curve(curve: Option<&Value>) -> Option<(Vec<f64>, Vec<f64>)> {
    let curve = curve?.as_object()?;
    let frequencies = finite_positive_values(curve.get("freq"))?;
    let levels = finite_values(curve.get("spl"))?;
    if !(2..=65_536).contains(&frequencies.len())
        || levels.len() != frequencies.len()
        || frequencies.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    Some((frequencies, levels))
}

fn pending_cell(field: &str) -> String {
    format!(
        "{PENDING_CELL_COLOR}{}\">pending</td>",
        crate::report_helpers::escape_html(field)
    )
}

fn number(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    number.is_finite().then_some(number)
}

fn finite_values(value: Option<&Value>) -> Option<Vec<f64>> {
    let values = value?.as_array()?;
    values.iter().map(|value| number(Some(value))).collect()
}

fn finite_positive_values(value: Option<&Value>) -> Option<Vec<f64>> {
    let values = finite_values(value)?;
    values.iter().all(|value| *value > 0.0).then_some(values)
}

fn valid_optional_nonnegative_number(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(value) => number(Some(value)).is_some_and(|number| number >= 0.0),
    }
}

fn array(value: Option<&Value>) -> Vec<&Value> {
    value
        .and_then(Value::as_array)
        .map(|values| values.iter().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn event(gain: f64, time: f64) -> Value {
        json!({
            "gain_dbfs": gain,
            "time_ms": time,
            "distance_cm": 34.3 * time,
            "first_dip_hz": 500.0 / time,
            "ripple_db": 4.0
        })
    }

    fn report() -> Value {
        json!({
            "basis": "measured_room_ir",
            "method": "bandlimited_early_reflection_table_v1",
            "band_hz": [1000.0, 8000.0],
            "threshold_dbfs": -15.0,
            "direct_reference": "filtered direct peak",
            "pre": [event(-12.0, 5.0)],
            "post": [event(-8.0, 3.0)]
        })
    }

    #[test]
    fn reports_only_measured_valid_reflection_evidence() {
        let channel = json!({"early_reflections": report()});
        let output = reflection_report(&channel, "Left", Some("L"));
        assert_eq!(output["early_reflection_level_db"]["level_db"], -8.0);
        assert_eq!(output["early_reflection_level_db"]["upper_bound"], false);
        assert_eq!(output["figures"].as_array().unwrap().len(), 2);
        assert_eq!(output["figures"][0]["tab"], "L");

        let mut invalid = report();
        invalid["basis"] = json!("reconstructed_transfer_ir");
        let output = reflection_report(&json!({"early_reflections": invalid}), "Left", None);
        assert!(output["validated_report"].is_null());
        assert!(output["early_reflection_level_db"].is_null());
        assert!(output["figures"].as_array().unwrap().is_empty());
        assert!(output["html"].as_str().unwrap().contains("pending"));
    }

    #[test]
    fn candidate_figure_includes_etc_envelope_when_ir_is_saved() {
        // Decaying 2 kHz burst IR: direct at ~5 ms, sampled at 48 kHz.
        let count = 4800;
        let (time_ms, amplitude): (Vec<f64>, Vec<f64>) = (0..count)
            .map(|index| {
                let t = index as f64 / 48000.0;
                let envelope = if index < 240 {
                    0.0
                } else {
                    (-(t - 0.005) * 300.0).exp()
                };
                (
                    t * 1000.0,
                    envelope * (2.0 * std::f64::consts::PI * 2000.0 * t).sin(),
                )
            })
            .unzip();
        let channel = json!({
            "early_reflections": report(),
            "pre_ir": {"time_ms": time_ms, "amplitude": amplitude},
        });
        let output = reflection_report(&channel, "L", None);
        let figures = output["figures"].as_array().unwrap();
        let series = figures[0]["figure"]["series"].as_array().unwrap();
        assert_eq!(series.len(), 4, "{series:?}");
        assert_eq!(series[0]["name"], "1–8 kHz envelope");
        let xs = series[0]["x"].as_array().unwrap();
        let ys = series[0]["y"].as_array().unwrap();
        assert!(xs.len() > 16 && xs.len() == ys.len());
        assert!(
            xs.iter()
                .all(|x| (0.0..=15.0).contains(&x.as_f64().unwrap()))
        );
        assert!(
            ys.iter()
                .all(|y| y.as_f64().is_some_and(|v| (-40.0..=3.0).contains(&v)))
        );
        // Without a saved waveform the figure keeps its three series.
        let bare = reflection_report(&json!({"early_reflections": report()}), "L", None);
        assert_eq!(
            bare["figures"][0]["figure"]["series"].as_array().unwrap().len(),
            3
        );
    }

    #[test]
    fn no_emitted_candidates_are_a_bound_not_a_zero_level() {
        let mut evidence = report();
        evidence["post"] = json!([]);
        let output = reflection_report(&json!({"early_reflections": evidence}), "L", None);
        assert_eq!(output["early_reflection_level_db"]["level_db"], -15.0);
        assert_eq!(output["early_reflection_level_db"]["upper_bound"], true);
        assert!(
            output["early_reflection_level_cell"]
                .as_str()
                .unwrap()
                .contains("≤ -15.0")
        );
    }

    #[test]
    fn geometry_order_and_booleans_are_validated() {
        let mut evidence = report();
        evidence["pre"][0]["distance_cm"] = json!(250.0);
        assert!(validated_early_reflections(&json!({"early_reflections": evidence})).is_none());
        let mut evidence = report();
        evidence["pre"][0]["gain_dbfs"] = json!(true);
        assert!(validated_early_reflections(&json!({"early_reflections": evidence})).is_none());
    }

    #[test]
    fn early_late_ratio_requires_matching_full_peak_band_curves() {
        let channel = json!({"early_late_curves": {
            "method": "incoherent_band_energy",
            "reference": "full_peak_band",
            "smoothing": "third_octave",
            "split_ms": 20.0,
            "full": {"freq": [1000.0, 2000.0, 4000.0, 8000.0], "spl": [0.0, 0.0, 0.0, 0.0]},
            "early": {"freq": [1000.0, 2000.0, 4000.0, 8000.0], "spl": [-1.0, -1.0, -1.0, -1.0]},
            "late": {"freq": [1000.0, 2000.0, 4000.0, 8000.0], "spl": [-7.0, -7.0, -7.0, -7.0]}
        }});
        let output = reflection_report(&channel, "L", None);
        assert_eq!(output["early_late_ratio_db"], 6.0);
        assert!(
            output["early_late_ratio_cell"]
                .as_str()
                .unwrap()
                .contains("+6.0")
        );
        let mut invalid = channel.clone();
        invalid["early_late_curves"]["reference"] = json!("each_segment_peak");
        assert!(early_late_ratio_db(&invalid).is_none());
    }

    #[test]
    fn response_figure_escapes_no_values_and_validates_curve_shapes() {
        let channel = json!({
            "early_reflections": report(),
            "initial_curve": {"freq": [1000.0, 2000.0], "spl": [80.0, 81.0]},
            "final_curve": {"freq": [1000.0, 2000.0], "spl": [79.0, 80.0]}
        });
        let output = reflection_report(&channel, "<Left>", None);
        assert_eq!(output["figures"].as_array().unwrap().len(), 3);
        assert!(output["html"].as_str().unwrap().contains("&lt;Left&gt;"));
        assert_eq!(
            output["figures"][1]["figure"]["series"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let invalid_curve = json!({"freq": [1000.0, 500.0], "spl": [80.0, 81.0]});
        assert!(valid_curve(Some(&invalid_curve)).is_none());
    }
}
