//! Renders saved playback, psychoacoustic, and FIR timing evidence.
//!
//! These sections display serialized evidence without repeating measurement or
//! payload verification. Playback status requires the caller's binding result
//! from the authoritative bundle verifier.

// Rust guideline compliant 2026-02-21

use crate::report_document::{
    DiagnosticBlock, DiagnosticIntroduction, DiagnosticSectionStyle, EpaDelta, EpaScoreRow,
    NoticeTone, ReportNotice, ResonanceCell, ResonanceEvidence, ResonanceRow, TableStyle,
    diagnostic_html, diagnostic_section, section_fragment, table_section,
};
use crate::report_helpers::{
    channel_sort_key, escape_html, format_ms, numeric_value, python_string,
};
use crate::report_table::{ReportTable, write_table_html};
use serde_json::{Map, Value, json};

const OK: &str = "#2ecc71";
const WARN: &str = "#f1c40f";
const BAD: &str = "#e74c3c";
const EMPTY_CELL: &str = "<td style=\"color:#999\">-</td>";

fn populated(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn number(value: &Value) -> Option<f64> {
    numeric_value(value).filter(|value| value.is_finite())
}

fn status_box(notice: &ReportNotice) -> String {
    let color = match notice.tone {
        NoticeTone::Success => OK,
        NoticeTone::Warning => WARN,
        NoticeTone::Error => BAD,
    };
    let mut html = format!(
        "<section class=\"playback-status-box\" role=\"note\" style=\"flex:1;min-width:260px;border:2px solid {color};padding:16px\"><h2>{}</h2>",
        escape_html(&notice.title)
    );
    for detail in &notice.paragraphs {
        html.push_str(&format!("<p>{}</p>", escape_html(detail)));
    }
    html.push_str("</section>\n");
    html
}

fn playback_status(request: &Value) -> String {
    playback_notices_html(&playback_notices(request))
}

pub(crate) fn playback_notices(request: &Value) -> Vec<ReportNotice> {
    let metadata = &request["metadata"];
    let acceptance = &metadata["correction_acceptance"];
    let outcome = acceptance["outcome"].as_str();
    let eligible = outcome == Some("unchanged")
        || (outcome == Some("accepted")
            && acceptance["accepted"] == true
            && acceptance["decision"] == "accepted");
    let verified = request["binding_verified"] == true;
    let approved = eligible && verified;
    let label = request["label"].as_str().unwrap_or_default();
    let prefix = if label.is_empty() {
        String::new()
    } else {
        format!("{label} — ")
    };
    let title = format!(
        "{prefix}Saved DSP playback eligibility: {}",
        outcome
            .filter(|value| !value.is_empty())
            .unwrap_or("unverified")
    );
    let (eligibility_color, detail) = match outcome {
        Some("accepted" | "unchanged") if eligible => (
            NoticeTone::Success,
            format!("Recorded verdict: {}.", outcome.unwrap_or_default()),
        ),
        Some("accepted" | "unchanged") => (
            NoticeTone::Error,
            format!(
                "Recorded verdict is contradictory: outcome '{}' without matching accepted/decision flags.",
                outcome.unwrap_or_default()
            ),
        ),
        Some("rejected") => (NoticeTone::Error, "Recorded verdict: rejected.".into()),
        None => (
            NoticeTone::Warning,
            "No recorded acceptance verdict (legacy or unverified output).".into(),
        ),
        Some(value) => (
            NoticeTone::Warning,
            format!("Recorded verdict: {value} — no playback approval."),
        ),
    };
    let mut eligibility = vec![detail];
    let violations: Vec<_> = acceptance["violations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .collect();
    if !violations.is_empty() {
        eligibility.push(format!("Recorded violations: {}.", violations.join("; ")));
    }
    let binding_title = format!(
        "{prefix}Delivered payload: {}",
        if verified { "verified" } else { "not verified" }
    );
    let mut playback = vec![if approved {
        "Approved for playback under the declared contract.".to_string()
    } else {
        "Not approved for playback. These curves are diagnostic only.".to_string()
    }];
    if approved {
        playback.push("Recorded validation is evidence for the saved graph and its declared playback contract, not an independent replay by this report.".into());
    }
    let policy = &metadata["effective_config"]["optimizer"]["finalization"];
    let mut peaks = std::collections::BTreeMap::new();
    peaks.insert(
        "default",
        policy
            .get("default_input_peak")
            .and_then(number)
            .unwrap_or(1.0),
    );
    if let Some(limits) = policy["input_peak_limits"].as_object() {
        for (name, value) in limits {
            if let Some(value) = number(value) {
                peaks.insert(name.as_str(), value);
            } else {
                peaks.remove(name.as_str());
            }
        }
    }
    let reduced: Vec<_> = peaks
        .into_iter()
        .filter(|(_, value)| *value > 0.0 && *value < 1.0)
        .collect();
    if !reduced.is_empty() {
        let values = reduced
            .iter()
            .map(|(name, peak)| format!("{name}: {:.2} dBFS", 20.0 * peak.log10()))
            .collect::<Vec<_>>()
            .join(", ");
        playback.push(format!("Conditional on enforced input-peak ceilings ({values}). This report does not enforce them; full-scale input safety is not established."));
    }
    let limited: Vec<_> = metadata["stage_outcomes"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|stage| stage["checks"].as_array().into_iter().flatten())
        .filter_map(|check| {
            check["id"]
                .as_str()?
                .strip_prefix("runtime_limiter_physical_output:")
        })
        .collect();
    if !limited.is_empty() {
        playback.push(format!("Runtime limiter required on physical outputs: {}. Response curves describe small-signal playback below limiting. Loud bass peaks may be reduced dynamically. Protection is sample-peak, not a true-peak or loudspeaker-excursion guarantee. Do not bypass the limiter or add downstream gain.", limited.join(", ")));
    }
    let playback_color = if !approved {
        NoticeTone::Error
    } else if !reduced.is_empty() || !limited.is_empty() {
        NoticeTone::Warning
    } else {
        NoticeTone::Success
    };
    vec![
        ReportNotice {
            title,
            tone: eligibility_color,
            paragraphs: eligibility,
        },
        ReportNotice {
            title: binding_title,
            tone: if verified {
                NoticeTone::Success
            } else {
                NoticeTone::Warning
            },
            paragraphs: vec![
                request["binding_reason"]
                    .as_str()
                    .unwrap_or("Binding evidence unavailable.")
                    .to_string(),
            ],
        },
        ReportNotice {
            title: format!("{prefix}Playback approval"),
            tone: playback_color,
            paragraphs: playback,
        },
    ]
}

pub(crate) fn playback_notices_html(notices: &[ReportNotice]) -> String {
    let mut html = "<section class=\"playback-status\" role=\"group\" aria-label=\"Playback verdict\" style=\"display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;border:0;padding:0\">".to_string();
    for notice in notices {
        html.push_str(&status_box(notice));
    }
    html.push_str("</section>\n");
    html
}

fn labelled_heading(title: &str, label: &str) -> String {
    if label.is_empty() {
        title.to_string()
    } else {
        format!("{title} — {label}")
    }
}

fn waveform_status_html(data: &Value, label: &str) -> String {
    waveform_status(data, label).map_or_else(String::new, |table| render_waveform_status(&table))
}

pub(crate) fn render_waveform_status(table: &ReportTable) -> String {
    let mut html = format!(
        "<section class=\"waveform-status\"><h2>{}</h2><p>{}</p>",
        escape_html(&table.title),
        escape_html(&table.note),
    );
    if !table.columns.is_empty() {
        write_table_html(&mut html, table, "", "");
    }
    html.push_str("</section>");
    html
}

fn waveform_status(data: &Value, label: &str) -> Option<ReportTable> {
    let title = labelled_heading("Waveform-view availability", label);
    let unavailable = |reason: &str| {
        Some(ReportTable {
            title: title.clone(),
            note: format!("Unavailable: {reason}."),
            columns: Vec::new(),
            rows: Vec::new(),
        })
    };
    if !data.is_object() {
        return unavailable("malformed report data");
    }
    let metadata = &data["metadata"];
    if metadata.is_null() {
        return None;
    }
    if !metadata.is_object() {
        return unavailable("malformed waveform metadata");
    }
    // The legacy Python renderer treats an absent field as an old report with
    // no waveform-view stage. An explicit null remains malformed evidence.
    let Some(outcomes) = metadata.get("stage_outcomes") else {
        return None;
    };
    if !outcomes.is_array()
        || outcomes
            .as_array()
            .is_some_and(|items| items.iter().any(|item| !item.is_object()))
    {
        return unavailable("malformed stage outcomes");
    }
    let stages: Vec<_> = outcomes
        .as_array()
        .into_iter()
        .flatten()
        .filter(|stage| stage["stage"] == "waveform_views")
        .collect();
    if stages.is_empty() {
        return None;
    }
    if stages.len() != 1 {
        return unavailable("conflicting waveform diagnostics");
    }
    let channels = &data["channels"];
    let Some(channel_map) = channels.as_object() else {
        return unavailable("malformed waveform channel inventory");
    };
    if channel_map.values().any(|channel| !channel.is_object()) {
        return unavailable("malformed waveform channel inventory");
    }
    let checks = &stages[0]["checks"];
    let Some(checks) = checks.as_array() else {
        return unavailable("malformed waveform checks");
    };
    if checks.iter().any(|check| !check.is_object()) {
        return unavailable("malformed waveform checks");
    }
    let expected: std::collections::BTreeSet<_> = channel_map
        .keys()
        .flat_map(|channel| [format!("pre_ir:{channel}"), format!("post_ir:{channel}")])
        .collect();
    let ids: Vec<_> = checks
        .iter()
        .filter_map(|check| check["id"].as_str())
        .collect();
    let unique: std::collections::BTreeSet<_> = ids.iter().copied().collect();
    if ids.len() != checks.len()
        || ids.len() != expected.len()
        || unique.len() != ids.len()
        || unique
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != expected.iter().map(String::as_str).collect()
    {
        return unavailable("waveform checks do not cover each channel and view exactly once");
    }
    let mut sorted = checks.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|check| check["id"].as_str().unwrap_or_default());
    let mut rows = Vec::new();
    for check in sorted {
        let id = check["id"].as_str().unwrap_or_default();
        let Some((view, channel)) = id.split_once(':') else {
            return unavailable("waveform checks do not cover each channel and view exactly once");
        };
        let available = !channel_map[channel][view].is_null();
        let passed = check["passed"].as_bool();
        let state = match passed {
            Some(value) if available != value => {
                "Unavailable: recorded status and saved waveform disagree".to_string()
            }
            Some(true) => {
                let measured = view == "pre_ir"
                    && channel_map[channel]["t60_octaves"]["basis"] == "measured_room_ir";
                if measured {
                    "Measured room impulse response imported".to_string()
                } else if view == "post_ir" {
                    "Calculated impulse response saved — prediction after DSP".to_string()
                } else {
                    "Impulse response reconstructed from the input frequency response".to_string()
                }
            }
            Some(false) => format!(
                "Unavailable: {}",
                check["diagnostic"]
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .unwrap_or("no reason recorded")
            ),
            None => "Unavailable: recorded status and saved waveform disagree".to_string(),
        };
        rows.push(vec![channel.into(), view.into(), state]);
    }
    Some(ReportTable {
        title,
        note: "An impulse response shows amplitude over time. Before DSP uses the imported room IR when supplied; otherwise it is reconstructed from the input response. After DSP is a calculated prediction, not a new microphone measurement. Available means the waveform was saved and can be plotted.".into(),
        columns: ["Channel", "View", "Status / reason"].map(String::from).into(),
        rows,
    })
}

fn finite(value: &Value) -> Option<f64> {
    value.as_f64().filter(|value| value.is_finite())
}

fn known(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|text| !text.trim().is_empty() && !text.trim().eq_ignore_ascii_case("unknown"))
}

fn clock_config(data: &Value) -> &Value {
    if data["speakers"].is_object() {
        data
    } else {
        data.get("metadata")
            .and_then(|value| value.get("effective_config"))
            .unwrap_or(&Value::Null)
    }
}

fn capture_reason(source: &Value) -> Option<String> {
    if !source.is_object() {
        return Some("capture provenance is unavailable".into());
    }
    let provenance = &source["provenance"];
    let capture = &provenance["capture"];
    if !capture.is_object() {
        return Some("per-microphone clock bounds are unavailable".into());
    }
    let measurements = source["measurements"].as_array();
    let takes = capture["takes"].as_array();
    let (Some(measurements), Some(takes)) = (measurements, takes) else {
        return Some("clock evidence does not identify every microphone".into());
    };
    if !(2..=4).contains(&takes.len()) || takes.len() != measurements.len() {
        return Some("clock evidence does not identify every microphone".into());
    }
    let geometry = capture["geometry"].as_str();
    if !matches!(geometry, Some("spread" | "compact")) {
        return Some("session geometry is unavailable".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut reference: Option<&str> = None;
    for take in takes {
        if !take.is_object() {
            return Some("invalid microphone clock record".into());
        }
        let mic = take["microphone_id"].as_str().unwrap_or("unknown");
        let prefix = format!("{mic}: ");
        if !known(&take["microphone_id"])
            || !ids.insert(mic.to_string())
            || !known(&take["device_id"])
            || !known(&take["calibration_id"])
        {
            return Some(format!(
                "{prefix}microphone, device, or calibration identity is invalid"
            ));
        }
        let position = take["position_m"].as_array();
        let uncertainty = finite(&take["position_uncertainty_mm"]);
        if position.is_none_or(|position| {
            position.len() != 3 || position.iter().any(|v| finite(v).is_none())
        }) || uncertainty
            .is_none_or(|value| value < 0.0 || (geometry == Some("compact") && value > 1.0))
        {
            return Some(format!("{prefix}microphone geometry is invalid"));
        }
        if finite(&take["gain_db"]).is_none()
            || !matches!(
                take["calibration_orientation"].as_str(),
                Some("on_axis" | "ninety_degrees")
            )
        {
            return Some(format!(
                "{prefix}gain or calibration orientation is invalid"
            ));
        }
        if take["correction_applied"] != "resampled" || take["preserves_acoustic_delay"] != true {
            return Some(format!(
                "{prefix}common-clock acoustic-delay evidence is unavailable"
            ));
        }
        if take["quality_passed"] != true {
            return Some(format!("{prefix}measurement quality is pending or failed"));
        }
        let skew = finite(&take["skew_ppm"]);
        if finite(&take["offset_samples"]).is_none()
            || skew.is_none_or(|value| value.abs() > 5000.0)
        {
            return Some(format!("{prefix}clock fit is invalid"));
        }
        let bound = finite(&take["residual_uncertainty_us"]);
        if bound.is_none_or(|value| value < 0.0 || value >= 50.0) {
            return Some(format!(
                "{prefix}residual bound is unavailable or not below 50 µs"
            ));
        }
        let declared = take["timing_reference_id"].as_str();
        if declared.is_none_or(|value| {
            !known(&take["timing_reference_id"]) || reference.is_some_and(|old| old != value)
        }) {
            return Some(format!(
                "{prefix}microphones do not share a timing reference"
            ));
        }
        reference = declared;
    }
    if reference != provenance["timing_reference_id"].as_str() {
        return Some("source and microphone timing references disagree".into());
    }
    if !matches!(
        provenance["capture_kind"].as_str(),
        Some("stationary_ir" | "direct_sound")
    ) {
        return Some("source contains magnitude-only acquisition evidence".into());
    }
    None
}

/// Returns the first persisted clock-gate failure for a requested output pair.
///
/// Channel-to-source resolution uses the same saved `system.speakers` mapping as
/// the capture-view renderer. A missing source, duplicate channel/source, or a
/// failed per-source clock check returns a reason; callers must not sum phase
/// coherently when this returns `Some`.
#[must_use]
pub fn coherent_capture_clock_reason(data: &Value, channels: [&str; 2]) -> Option<String> {
    if channels[0].is_empty() || channels[1].is_empty() || channels[0] == channels[1] {
        return Some("two distinct output channels are required".into());
    }
    let config = clock_config(data);
    let Some(sources) = config["speakers"].as_object() else {
        return Some("capture source inventory is unavailable".into());
    };
    let mappings = config["system"]["speakers"].as_object();
    let mut source_ids = std::collections::BTreeSet::new();
    for channel in channels {
        let source_id = mappings
            .and_then(|items| items.get(channel))
            .and_then(Value::as_str)
            .unwrap_or(channel);
        if !source_ids.insert(source_id) {
            return Some(format!(
                "output channels resolve to the same capture source: {source_id}"
            ));
        }
        let Some(source) = sources.get(source_id) else {
            return Some(format!("capture provenance is unavailable for {source_id}"));
        };
        if let Some(reason) = capture_reason(source) {
            return Some(format!("{source_id}: {reason}"));
        }
    }
    None
}

fn capture_clock_qa(data: &Value) -> Option<ReportTable> {
    let config = clock_config(data);
    let speakers = config["speakers"].as_object()?;
    let mut rows = Vec::new();
    for (source_name, source) in speakers {
        if !source.is_object() {
            continue;
        }
        let capture = &source["provenance"]["capture"];
        if !capture.is_object() {
            continue;
        }
        let reason = capture_reason(source);
        let takes = capture["takes"].as_array();
        let fallback = vec![Value::Object(Map::new())];
        for take in takes.filter(|items| !items.is_empty()).unwrap_or(&fallback) {
            if !take.is_object() {
                continue;
            }
            let number_text = |name: &str| {
                finite(&take[name])
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "unavailable".into())
            };
            let geometry = capture["geometry"].as_str().unwrap_or("unknown");
            let microphone = take["microphone_id"].as_str().unwrap_or("unknown");
            let device = take["device_id"].as_str().unwrap_or("unknown");
            let correction = take["correction_applied"].as_str().unwrap_or("unknown");
            let status = reason
                .as_deref()
                .unwrap_or("Clock gate passed; phase validity still required");
            rows.push(vec![
                source_name.to_string(),
                geometry.into(),
                microphone.into(),
                device.into(),
                number_text("offset_samples"),
                number_text("skew_ppm"),
                number_text("residual_uncertainty_us"),
                correction.into(),
                status.into(),
            ]);
        }
    }
    (!rows.is_empty()).then(|| ReportTable {
        title: "Capture clock QA".into(),
        note: "Every participating microphone needs a finite residual bound below 50 µs, a common reference, and passed measurement quality. Missing evidence uses magnitude only. Bounds assume linear clocks and correctly isolated timing markers.".into(),
        columns: ["Source", "Geometry", "Microphone", "Device", "Offset (samples)",
            "Skew (ppm)", "Residual (µs)", "Correction", "Coherent-use status"]
            .map(str::to_owned).to_vec(),
        rows,
    })
}

pub(crate) fn render_capture_clock_qa(table: &ReportTable) -> String {
    let mut html = format!(
        "<section class=\"capture-clock-qa\"><h2>{}</h2><p>{}</p>",
        escape_html(&table.title),
        escape_html(&table.note),
    );
    write_table_html(&mut html, table, "summary-table", "");
    html.push_str("</section>");
    html
}

fn capture_clock_qa_html(data: &Value) -> String {
    capture_clock_qa(data)
        .as_ref()
        .map(render_capture_clock_qa)
        .unwrap_or_default()
}

/// Renders serialized captured-arrival evidence without estimating new directions.
pub fn capture_reflections_html(data: &Value) -> String {
    section_fragment(capture_reflections_section(data))
        .expect("capture reflection evidence is serialized by AutoEQ")
}

fn capture_reflections_section(data: &Value) -> Value {
    let config = clock_config(data);
    let channels = data
        .get("channels")
        .and_then(Value::as_object)
        .or_else(|| config.get("speakers").and_then(Value::as_object));
    let Some(channels) = channels else {
        return Value::Null;
    };
    let system_speakers = config["system"]["speakers"].as_object();
    let sources = config["speakers"].as_object();
    let mut blocks = Vec::new();
    for channel in channels.keys() {
        let source_id = system_speakers
            .and_then(|speakers| speakers.get(channel))
            .and_then(Value::as_str)
            .unwrap_or(channel);
        let source = sources
            .and_then(|items| items.get(source_id))
            .unwrap_or(&Value::Null);
        let report = &source["provenance"]["capture"]["reflection_report"];
        if !report.is_object() {
            continue;
        }
        let mut events = Vec::new();
        let direct = report.get("direct_sound").filter(|event| event.is_object());
        if let Some(direct) = direct {
            events.push((direct, true));
        }
        if let Some(reflections) = report.get("early_reflections").and_then(Value::as_array) {
            events.extend(
                reflections
                    .iter()
                    .take(64)
                    .filter(|event| event.is_object())
                    .map(|event| (event, false)),
            );
        }
        let mut rows = Vec::new();
        let mut plot_series = Vec::new();
        for (index, (event, is_direct)) in events.into_iter().enumerate() {
            let relative = finite(&event["relative_ms"]);
            let level = finite(&event["level_db"]);
            let mut reason = capture_reflection_direction_reason(source, event);
            if report["source_id"] != source_id {
                reason = Some("arrival source identity does not match this channel".into());
            }
            if relative.is_none_or(|value| !(0.0..=80.0).contains(&value)) || level.is_none() {
                reason = Some("finite early-arrival time/level unavailable".into());
            }
            let label = if is_direct {
                "Direct candidate".to_string()
            } else {
                format!(
                    "Reflection candidate {}",
                    index + 1 - usize::from(direct.is_some())
                )
            };
            let has_direction = reason.is_none();
            let mut status = reason
                .clone()
                .unwrap_or_else(|| "Conditional measured direction".into());
            if has_direction {
                let direction = &event["direction"];
                let x = finite(&direction[0]).unwrap_or(0.0);
                let y = finite(&direction[1]).unwrap_or(0.0);
                let z = finite(&direction[2]).unwrap_or(0.0);
                let horizontal = x.hypot(y);
                let azimuth = y.atan2(x).to_degrees();
                let elevation = z.clamp(-1.0, 1.0).asin().to_degrees();
                if horizontal <= 1e-6 {
                    status = format!(
                        "Conditional vertical arrival; azimuth undefined, elevation {elevation:.1}°"
                    );
                } else {
                    status.push_str(&format!(
                        "; azimuth {azimuth:.1}°, elevation {elevation:.1}°"
                    ));
                    if !is_direct {
                        let time = relative.unwrap_or_default();
                        let level = level.unwrap_or_default();
                        plot_series.push(json!({
                            "name":format!("{label}: {level:.1} dB"),
                            "x":[azimuth,azimuth],
                            "y":[0.0,time],
                            "color":"#56b4e9",
                            "width":2.0,
                            "dash":"solid",
                            "visible":true,
                            "y_axis":0
                        }));
                    }
                }
            }
            if let Some(issues) = event.get("issues").and_then(Value::as_array) {
                let issues = issues
                    .iter()
                    .map(python_string)
                    .collect::<Vec<_>>()
                    .join("; ");
                if !issues.is_empty() {
                    status.push_str("; ");
                    status.push_str(&issues);
                }
            }
            let time_text = relative
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "—".into());
            let level_text = level
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "—".into());
            rows.push(vec![label, time_text, level_text, status]);
        }
        let chart = (!plot_series.is_empty()).then(|| json!({
                "kind":"figure",
                "figure":{
                    "title":"Conditional arrival azimuth and delay",
                    "x":{"label":"Azimuth (degrees)","scale":"linear","min":-180,"max":180},
                    "y":{"label":"Delay from direct candidate (ms)","scale":"linear","min":0,"max":80},
                    "y2":null,"series":plot_series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
                },
                "tab":null
            }));
        let notes = report
            .get("issues")
            .and_then(Value::as_array)
            .map(|issues| {
                issues
                    .iter()
                    .map(python_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default();
        blocks.push(DiagnosticBlock::CaptureArrival {
            table: ReportTable {
                title: format!("{channel}: early-arrival candidates"),
                note: notes,
                columns: ["Event", "Relative ms", "Level dB", "Evidence"]
                    .map(str::to_owned)
                    .to_vec(),
                rows,
            },
            chart,
        });
    }
    if blocks.is_empty() {
        return Value::Null;
    }
    let mut section = diagnostic_section(blocks);
    section["introduction"] = json!(DiagnosticIntroduction {
        title: "Measured arrival evidence".into(),
        style: DiagnosticSectionStyle::Plain,
        text: "Energy-based arrival candidates and conditional plane-wave directions. Microphone phase and source distance are not independently verified; fit residuals are not angular uncertainty bounds.".into(),
    });
    section
}

fn capture_reflection_direction_reason(source: &Value, event: &Value) -> Option<String> {
    if let Some(reason) = capture_reason(source) {
        return Some(reason);
    }
    let capture = &source["provenance"]["capture"];
    if capture["geometry"] != "compact" {
        return Some("array directions require compact geometry".into());
    }
    if event["mirror_ambiguous"] != false {
        return Some("planar mirror ambiguity is unresolved".into());
    }
    let direction = event["direction"].as_array();
    let Some(direction) = direction.filter(|values| values.len() == 3) else {
        return Some("finite unit direction unavailable".into());
    };
    let Some(vector) = direction.iter().map(finite).collect::<Option<Vec<_>>>() else {
        return Some("finite unit direction unavailable".into());
    };
    let norm = vector.iter().map(|value| value * value).sum::<f64>();
    if (norm - 1.0).abs() > 1e-3 {
        return Some("finite unit direction unavailable".into());
    }
    if finite(&event["residual_samples"]).is_none_or(|value| !(0.0..=0.5).contains(&value)) {
        return Some("plane-wave fit residual unavailable or too large".into());
    }
    let band = event["band_hz"].as_array();
    if band.is_none_or(|values| {
        values.len() != 2
            || finite(&values[0]).is_none_or(|value| value <= 0.0)
            || finite(&values[1]).is_none_or(|value| value <= finite(&values[0]).unwrap_or(0.0))
            || finite(&values[1]).is_some_and(|value| value > 3000.0)
    }) {
        return Some("supported direction-analysis band unavailable".into());
    }
    let mut uncertainties = capture["takes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|take| finite(&take["residual_uncertainty_us"]))
        .collect::<Vec<_>>();
    uncertainties.sort_by(f64::total_cmp);
    uncertainties.reverse();
    let relative_us = uncertainties
        .first()
        .zip(uncertainties.get(1))
        .map(|(a, b)| a + b);
    if relative_us.is_none_or(|value| value <= 0.0)
        || finite(&band.unwrap()[1])
            .is_some_and(|value| value > 25_000.0 / relative_us.unwrap_or(1.0))
    {
        return Some(
            "relative timing uncertainty exceeds the direction-analysis band limit".into(),
        );
    }
    None
}

/// Renders the all-speaker resonance table from already validated saved grids.
pub fn resonance_summary_html(data: &Value) -> String {
    resonance_summary(data)
        .map(|block| diagnostic_html(&[block]))
        .unwrap_or_default()
}

fn resonance_summary(data: &Value) -> Option<DiagnosticBlock> {
    let channels = data["channels"].as_object()?;
    let mut entries = Vec::<(String, Option<&[Value]>)>::new();
    for (name, channel) in channels {
        let valid_drivers = channel["drivers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|driver| driver.is_object() && driver["measured_acoustics"].is_object())
            .collect::<Vec<_>>();
        if valid_drivers.is_empty() {
            entries.push((name.clone(), decay_modes(channel)));
        } else {
            for driver in valid_drivers {
                let driver_name = driver
                    .get("name")
                    .map(python_string)
                    .unwrap_or_else(|| "driver".into());
                let acoustic = &driver["measured_acoustics"];
                entries.push((format!("{name} / {driver_name}"), decay_modes(acoustic)));
            }
        }
    }
    let columns = entries
        .iter()
        .filter_map(|(_, modes)| modes.map(<[Value]>::len))
        .max()?
        .max(1);
    let rows = entries
        .into_iter()
        .map(|(name, modes)| {
            let evidence = match modes {
                None => ResonanceEvidence::Unavailable,
                Some([]) => ResonanceEvidence::NoCandidates,
                Some(modes) => {
                    let mut candidates = modes.iter().collect::<Vec<_>>();
                    candidates.sort_by(|left, right| {
                        finite(&left["freq_hz"])
                            .unwrap_or_default()
                            .total_cmp(&finite(&right["freq_hz"]).unwrap_or_default())
                    });
                    ResonanceEvidence::Candidates(
                        candidates
                            .into_iter()
                            .map(|mode| {
                                let frequency = finite(&mode["freq_hz"]).unwrap_or_default();
                                let level = finite(&mode["level_db"]).unwrap_or_default();
                                let decay = finite(&mode["decay_time_s"])
                                    .map(|value| format!("{value:.3} s"))
                                    .unwrap_or_else(|| "fit unavailable".into());
                                ResonanceCell {
                                    lines: [
                                        format!("{frequency:.1} Hz"),
                                        format!("{level:.2} dB"),
                                        decay,
                                    ],
                                    color: resonance_color(frequency),
                                }
                            })
                            .collect(),
                    )
                }
            };
            ResonanceRow { name, evidence }
        })
        .collect();
    Some(DiagnosticBlock::ResonanceSummary { columns, rows })
}

fn decay_modes(channel: &Value) -> Option<&[Value]> {
    let decays = &channel["resonance_decays"];
    if !crate::channel_evidence::valid_waterfall(&channel["waterfall"], decays) {
        return None;
    }
    decays["decays"].as_array().map(Vec::as_slice)
}

fn resonance_color(frequency: f64) -> u32 {
    let ratio = ((frequency.max(10.0) / 10.0).ln() / 30.0_f64.ln()).clamp(0.0, 1.0);
    let h = 0.5 + ratio / 3.0;
    let s = 0.65;
    let v = 0.8;
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f64;
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
    let [r, g, b] = [r, g, b].map(|value| (value * 255.0).round() as u32);
    (r << 16) | (g << 8) | b
}

pub(crate) fn embedded_report_figure(section: &Value) -> String {
    let encoded = serde_json::to_string(section).unwrap_or_else(|_| "{}".into());
    format!(
        "<div class=\"report-figure\" data-section=\"{}\"></div>",
        escape_html(&encoded)
    )
}

/// Renders capture verification views only after the native host confirms the
/// frozen comparison payload bindings supplied in `verification`.
pub fn capture_views_html(report: &Value, verification: &Value) -> String {
    section_fragment(capture_views_section(report, verification).0)
        .expect("capture comparisons are serialized by AutoEQ")
}

pub(crate) fn capture_views_section(report: &Value, verification: &Value) -> (Value, bool) {
    use DiagnosticBlock::{EvidenceList, Heading, MajorHeading, Paragraph, VerificationStatus};
    let section = |blocks| {
        let mut section = diagnostic_section(blocks);
        section["introduction"] = json!(DiagnosticIntroduction {
            title: "Capture acceptance diagnostics".into(),
            text: String::new(),
            style: DiagnosticSectionStyle::CaptureViews,
        });
        section
    };
    let unavailable = |reason: &str| {
        (
            section(vec![Paragraph(format!("Unavailable: {reason}"))]),
            false,
        )
    };
    if !report.is_object() {
        return unavailable("'Value' object has no attribute 'get'");
    }
    let mut complete = true;
    let mut blocks = vec![Paragraph("These plots do not establish safety, improved room damping, or listener benefit. Hashes check artifact consistency, not operator declarations or hardware authenticity.".into())];
    let status = report.get("status").filter(|value| !value.is_null());
    if let Some(status) = status {
        let status = status.as_str().unwrap_or_default();
        if !matches!(
            status,
            "accepted" | "unchanged" | "rejected" | "insufficient_evidence"
        ) {
            return unavailable("unsupported verification status");
        }
        let count = report["verified_captures"].as_u64();
        let exit_code = report["exit_code"].as_i64();
        let (Some(count), Some(exit_code)) = (count, exit_code) else {
            return unavailable("invalid verification capture count or exit code");
        };
        blocks.push(VerificationStatus {
            status: status.into(),
            count,
            exit_code,
        });
        if !report["detail"].is_null() {
            let Some(detail) = report["detail"].as_str() else {
                return unavailable("invalid verification detail");
            };
            blocks.push(Paragraph(format!("Recorded detail: {detail}")));
        }
    } else {
        blocks.push(Paragraph(
            "Recorded capture verification status unavailable in this legacy report.".into(),
        ));
    }
    let graph = report["graph_id"].as_str();
    let comparisons = report["comparisons"].as_array();
    let (Some(graph), Some(comparisons)) = (graph, comparisons) else {
        return unavailable("missing comparison identity or entries");
    };
    if graph.is_empty() || comparisons.is_empty() {
        return unavailable("missing comparison identity or entries");
    }
    if verification["graph_id"] != graph {
        return unavailable(
            "verification candidate graph does not match the saved optimization graph",
        );
    }
    let binding_rows = verification["comparisons"].as_array();
    let Some(binding_rows) = binding_rows else {
        return unavailable("capture-view payload verification is unavailable");
    };
    if binding_rows.len() != comparisons.len() {
        return unavailable(
            "capture-view payload verification does not cover each comparison exactly once",
        );
    }
    let mut t60_groups = Vec::<(String, String, Value)>::new();
    for (index, comparison) in comparisons.iter().enumerate() {
        let views = &comparison["capture_views"];
        if views.is_null() {
            complete = false;
            blocks.push(Paragraph(
                "Unavailable: legacy comparison has no capture views.".into(),
            ));
            continue;
        }
        let binding = binding_rows.get(index);
        let stored_binding = &comparison["capture_views_binding"];
        if binding.is_none_or(|binding| {
            binding["index"].as_u64() != Some(index as u64)
                || binding["verified"] != true
                || binding["source"] != comparison["source"]
                || binding["seat"] != comparison["seat"]
                || binding["payload_sha256"] != stored_binding["sha256"]
                || binding["payload_sha256"].as_str().is_none_or(str::is_empty)
        }) || stored_binding["algorithm"] != "sha256-json-typed-v1"
            || stored_binding["graph_identity"] != graph
            || views["candidate_graph_id"] != graph
            || views["source"] != comparison["source"]
            || views["seat"] != comparison["seat"]
        {
            return unavailable("capture view payload or graph/source/seat binding changed");
        }
        let Some(source) = views["source"].as_str() else {
            return unavailable("invalid capture identity or scope");
        };
        let Some(seat) = views["seat"].as_str() else {
            return unavailable("invalid capture identity or scope");
        };
        let Some(baseline) = views["baseline_graph_id"].as_str() else {
            return unavailable("invalid capture identity or scope");
        };
        let Some(scope) = views["scope"].as_str() else {
            return unavailable("invalid capture identity or scope");
        };
        let kind = views["evidence_kind"].as_str().unwrap_or_default();
        if !matches!(
            kind,
            "unavailable" | "synthetic_capture_pair" | "operator_declared_recorded_capture_pair"
        ) {
            return unavailable("unsupported capture evidence kind");
        }
        blocks.push(MajorHeading(format!("{source} / {seat}")));
        blocks.push(Paragraph(format!(
            "Matched IR evidence: {kind}; baseline {baseline}; candidate {graph}"
        )));
        blocks.push(Paragraph(scope.into()));
        if !views["ir_step"].is_null() {
            let view = &views["ir_step"];
            if view["provenance"]["graph_identity"] != graph {
                return unavailable("IR graph identity mismatch");
            }
            let Some(reference) = view["common_reference"].as_str() else {
                return unavailable("missing IR sample-zero reference");
            };
            blocks.push(Paragraph(format!(
                "Common sample-zero reference: {reference}"
            )));
            match capture_trace_figure(view, "ir", "raw sample units", "IR") {
                Ok(chart) => blocks.push(chart),
                Err(reason) => return unavailable(&reason),
            }
            match capture_trace_figure(view, "step", "raw sample units", "STEP") {
                Ok(chart) => blocks.push(chart),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["etc"].is_null() {
            let etc = &views["etc"];
            if etc["provenance"]["graph_identity"] != graph
                || etc["method"] != "hann_analytic_octave_linear_v1"
            {
                return unavailable("ETC graph identity or analysis method mismatch");
            }
            if etc["window_ms"] != json!([0.0, 40.0])
                || etc["bands"].as_array().is_none_or(|bands| bands.len() != 4)
                || etc["bands"].as_array().is_some_and(|bands| {
                    bands
                        .iter()
                        .map(|band| band["center_hz"].as_f64())
                        .collect::<Vec<_>>()
                        != [Some(500.0), Some(1000.0), Some(2000.0), Some(4000.0)]
                })
            {
                return unavailable("ETC band/window contract mismatch");
            }
            let Some(etc_scope) = etc["scope"].as_str() else {
                return unavailable("missing ETC scope");
            };
            blocks.push(Heading("Matched octave ETC".into()));
            blocks.push(Paragraph(etc_scope.into()));
            blocks.push(Paragraph(format!(
                "Analysis half-support by band (ms): {}; display floor: {} dB.",
                python_json(&etc["filter_half_support_ms"]),
                python_json(&etc["display_floor_db"]),
            )));
            for band in etc["bands"].as_array().unwrap() {
                let trace = json!({"times_ms":band["times_ms"],"pre_etc":band["pre_db"],"post_etc":band["post_db"]});
                let label = format!("ETC {} Hz", python_number(&band["center_hz"]));
                match capture_trace_figure(
                    &trace,
                    "etc",
                    "dB relative to the same baseline-band peak",
                    &label,
                ) {
                    Ok(chart) => blocks.push(chart),
                    Err(reason) => return unavailable(&reason),
                }
            }
        }
        if !views["decay"].is_null() {
            match capture_decay_blocks(&views["decay"], graph) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["octave_t60"].is_null() {
            match capture_octave_t60_blocks(&views["octave_t60"]) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
            let band = &views["valid_band_hz"];
            let sample_rate = finite(&views["sample_rate_hz"]);
            let stimulus = views["stimulus_hash"].as_str();
            let settings = &views["settings"];
            if kind != "unavailable"
                && settings.is_object()
                && band.as_array().is_some_and(|values| {
                    values.len() == 2
                        && finite(&values[0]).is_some_and(|value| value > 0.0)
                        && finite(&values[1])
                            .is_some_and(|value| value > finite(&values[0]).unwrap_or(0.0))
                        && views["octave_t60"]["valid_band_hz"]
                            .as_array()
                            .is_some_and(|other| values == other)
                })
                && stimulus.is_some_and(|value| !value.is_empty())
                && sample_rate.is_some_and(|value| value > 0.0)
            {
                let key = format!(
                    "{}|{}|{}|{}|{}|{}|{}",
                    views["seat"],
                    kind,
                    views["baseline_graph_id"],
                    stimulus.unwrap_or_default(),
                    sample_rate.unwrap_or_default(),
                    json_python_separators(settings),
                    band
                );
                t60_groups.push((key, source.to_string(), views["octave_t60"].clone()));
            }
        }
        if !views["early_reflections"].is_null() {
            match capture_early_reflections_blocks(&views["early_reflections"]) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["early_late_curves"].is_null() {
            match capture_early_late_blocks(&views["early_late_curves"]) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["waterfall"].is_null() {
            match capture_waterfall_blocks(&views["waterfall"]) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["wavelet"].is_null() {
            match capture_wavelet_blocks(&views["wavelet"]) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        if !views["ambient_noise"].is_null() {
            match capture_noise_blocks(&views["ambient_noise"], graph) {
                Ok(contents) => blocks.extend(contents),
                Err(reason) => return unavailable(&reason),
            }
        }
        let unavailable_rows = views["unavailable"].as_object();
        let Some(unavailable_rows) = unavailable_rows else {
            return unavailable("'unavailable'");
        };
        let mut unavailable_items = Vec::new();
        for (key, reason) in unavailable_rows {
            let Some(reason) = reason.as_str() else {
                return unavailable("invalid capture unavailability reason");
            };
            unavailable_items.push(format!("{key} unavailable: {reason}"));
        }
        blocks.push(EvidenceList(unavailable_items));
    }
    blocks.extend(capture_room_mean_t60_blocks(&t60_groups));
    (section(blocks), complete)
}

fn capture_trace_figure(
    view: &Value,
    prefix: &str,
    units: &str,
    title: &str,
) -> Result<DiagnosticBlock, String> {
    let times = view["times_ms"]
        .as_array()
        .ok_or("missing raw trace arrays")?;
    let before = view[format!("pre_{prefix}")]
        .as_array()
        .ok_or("missing raw trace arrays")?;
    let after = view[format!("post_{prefix}")]
        .as_array()
        .ok_or("missing raw trace arrays")?;
    if times.len() < 2
        || times.len() != before.len()
        || times.len() != after.len()
        || times.len() > 65_536
    {
        return Err("unaligned or oversized raw traces".into());
    }
    let Some(x) = finite_values(times) else {
        return Err("invalid raw trace value".into());
    };
    let Some(y0) = finite_values(before) else {
        return Err("invalid raw trace value".into());
    };
    let Some(y1) = finite_values(after) else {
        return Err("invalid raw trace value".into());
    };
    if x[0] != 0.0 || x.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("invalid common time axis".into());
    }
    let title_json = json!({"kind":"figure","figure":{
        "title":title,
        "x":{"label":"Time (ms)","scale":"linear","min":0,"max":times[times.len()-1]},
        "y":{"label":units,"scale":"linear","min":null,"max":null},
        "y2":null,
        "series":[
            {"name":"Baseline","x":x,"y":y0,"color":"#2864b4","width":2.0,"dash":"solid","visible":true,"y_axis":0},
            {"name":"Candidate","x":x,"y":y1,"color":"#c56816","width":2.0,"dash":"solid","visible":true,"y_axis":0}
        ],"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
    },"tab":null});
    Ok(DiagnosticBlock::CaptureChart(title_json))
}

fn capture_early_reflections_blocks(view: &Value) -> Result<Vec<DiagnosticBlock>, String> {
    if view["method"] != "bandlimited_early_reflection_table_v1"
        || !numeric_array_equals(&view["band_hz"], &[1000.0, 8000.0])
        || finite(&view["threshold_dbfs"]) != Some(-15.0)
        || !["pre_direct_sample", "post_direct_sample"]
            .into_iter()
            .all(|key| view[key].as_u64().is_some())
    {
        return Err("invalid early reflection analysis contract".into());
    }
    let scope = view["scope"]
        .as_str()
        .ok_or("invalid early reflection analysis contract")?;
    let direct_reference = view["direct_reference"]
        .as_str()
        .ok_or("invalid early reflection analysis contract")?;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, Subheading, Table};
    let mut blocks = vec![
        Heading("Early reflection candidates (1–8 kHz)".into()),
        Paragraph(scope.into()),
        Paragraph(format!(
            "{direct_reference}; reflections below −15 dB relative to direct are omitted. Filtered direct samples: baseline {}, candidate {}; table times are relative to each direct peak.",
            view["pre_direct_sample"].as_u64().unwrap_or_default(),
            view["post_direct_sample"].as_u64().unwrap_or_default(),
        )),
    ];
    for (key, label, color) in [
        ("pre", "Baseline", "#2864b4"),
        ("post", "Candidate", "#c56816"),
    ] {
        let events = view[key]
            .as_array()
            .filter(|events| events.len() <= 64)
            .ok_or("invalid early reflection event list")?;
        let mut last_time = 0.0;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for event in events {
            if !event.is_object() {
                return Err("invalid early reflection event".into());
            }
            let gain = finite(&event["gain_dbfs"]).ok_or("invalid early reflection values")?;
            let time = finite(&event["time_ms"]).ok_or("invalid early reflection values")?;
            let distance =
                finite(&event["distance_cm"]).ok_or("invalid early reflection values")?;
            let first_dip =
                finite(&event["first_dip_hz"]).ok_or("invalid early reflection values")?;
            let ripple = if event["ripple_db"].is_null() {
                None
            } else {
                Some(finite(&event["ripple_db"]).ok_or("invalid early reflection values")?)
            };
            if time <= last_time
                || time > 15.0
                || gain < -15.0
                || distance <= 0.0
                || first_dip <= 0.0
                || ripple.is_some_and(|value| value < 0.0)
            {
                return Err("invalid early reflection values".into());
            }
            last_time = time;
            x.extend([time, time, time]);
            y.extend([json!(-40.0), json!(gain), Value::Null]);
        }
        blocks.push(Subheading(format!("{label} ({} candidates)", events.len())));
        if !events.is_empty() {
            let figure = json!({
                "kind":"figure","figure":{
                    "title":format!("{label}: early-reflection candidates"),
                    "x":{"label":"Time after direct peak (ms)","scale":"linear","min":0,"max":15},
                    "y":{"label":"Level relative to direct (dB)","scale":"linear","min":-40,"max":3},
                    "y2":null,
                    "series":[{"name":label,"x":x,"y":y,"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0}],
                    "hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
                },"tab":null
            });
            blocks.push(CaptureChart(figure));
        }
        let rows = events
            .iter()
            .enumerate()
            .map(|(index, event)| {
                let ripple = finite(&event["ripple_db"])
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "unbounded".into());
                let gain = finite(&event["gain_dbfs"]).unwrap_or_default();
                let time = finite(&event["time_ms"]).unwrap_or_default();
                let distance = finite(&event["distance_cm"]).unwrap_or_default();
                let dip = finite(&event["first_dip_hz"]).unwrap_or_default();
                vec![
                    (index + 1).to_string(),
                    format!("{gain:.2}"),
                    format!("{time:.2}"),
                    format!("{distance:.1}"),
                    format!("{dip:.1}"),
                    ripple,
                ]
            })
            .collect();
        blocks.push(Table(ReportTable {
            title: String::new(),
            note: String::new(),
            columns: [
                "Reflection",
                "Gain (dB re direct)",
                "Time (ms)",
                "Extra path (cm)",
                "First dip (Hz)",
                "Comb ripple (dB p-p)",
            ]
            .map(str::to_owned)
            .to_vec(),
            rows,
        }));
    }
    Ok(blocks)
}

fn capture_early_late_blocks(view: &Value) -> Result<Vec<DiagnosticBlock>, String> {
    let band = view["valid_band_hz"].as_array();
    if view["method"] != "incoherent_band_energy"
        || view["reference"] != "full_peak_band"
        || view["smoothing"] != "third_octave"
        || finite(&view["split_ms"]) != Some(20.0)
        || !matches!(
            view["direct_reference"].as_str(),
            Some("broadband envelope peak" | "120 Hz lowpass envelope peak")
        )
        || band.is_none_or(|band| {
            band.len() != 2
                || finite(&band[0]).is_none_or(|value| value <= 0.0)
                || finite(&band[1]).is_none_or(|value| value <= finite(&band[0]).unwrap_or(0.0))
        })
        || view["scope"].as_str().is_none_or(str::is_empty)
    {
        return Err("invalid capture early/late analysis contract".into());
    }
    let band = band.unwrap();
    let scope = view["scope"].as_str().unwrap_or_default();
    let direct_reference = view["direct_reference"].as_str().unwrap_or_default();
    let mut common_freq: Option<Vec<f64>> = None;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, Subheading};
    let mut blocks = vec![
        Heading("Captured IR early and late energy".into()),
        Paragraph(scope.into()),
        Paragraph(format!(
            "20 ms split from {direct_reference}; full, early, and late use one peak-band reference within each capture. Baseline and candidate levels have separate references."
        )),
    ];
    for (side, label) in [("pre", "Baseline"), ("post", "Candidate")] {
        let record = &view[side];
        if record["direct_sample"].as_u64().is_none() {
            return Err("invalid capture early/late direct sample".into());
        }
        let curves = ["full", "early", "late"].map(|key| &record[key]);
        if curves.iter().any(|curve| !curve.is_object()) {
            return Err("missing capture early/late curves".into());
        }
        let frequencies = curves[0]["freq"]
            .as_array()
            .ok_or("invalid capture early/late band support")?;
        if frequencies.len() < 2 || frequencies.len() > 21 {
            return Err("invalid capture early/late band support".into());
        }
        let freq = finite_values(frequencies).ok_or("invalid capture early/late band support")?;
        if freq.iter().any(|value| *value <= 0.0)
            || freq.windows(2).any(|pair| pair[0] >= pair[1])
            || freq.iter().any(|value| {
                value / 2.0_f64.powf(1.0 / 6.0) < finite(&band[0]).unwrap_or_default() - 1e-6
                    || value * 2.0_f64.powf(1.0 / 6.0) > finite(&band[1]).unwrap_or_default() + 1e-6
            })
            || common_freq.as_ref().is_some_and(|old| old != &freq)
        {
            return Err("invalid capture early/late band support".into());
        }
        common_freq = Some(freq.clone());
        let mut levels = Vec::new();
        for curve in curves {
            if curve["freq"] != Value::Array(frequencies.clone()) {
                return Err("invalid capture early/late levels".into());
            }
            let values = curve["spl"]
                .as_array()
                .ok_or("invalid capture early/late levels")?;
            let values = finite_values(values).ok_or("invalid capture early/late levels")?;
            if values.len() != freq.len()
                || values.iter().any(|value| !(-120.01..=0.01).contains(value))
            {
                return Err("invalid capture early/late levels".into());
            }
            levels.push(values);
        }
        if levels[0]
            .iter()
            .zip(&levels[1])
            .any(|(full, early)| *full + 1e-6 < *early)
            || levels[0]
                .iter()
                .zip(&levels[2])
                .any(|(full, late)| *full + 1e-6 < *late)
        {
            return Err(
                "capture early/late full level below a part: incoherent energy sum violated".into(),
            );
        }
        blocks.push(Subheading(label.into()));
        blocks.push(Paragraph(format!(
            "Direct sample: {}.",
            record["direct_sample"].as_u64().unwrap_or_default()
        )));
        let colors = ["#202020", "#2864b4", "#c56816"];
        let names = ["Full", "Early", "Late"];
        let series = names.iter().enumerate().map(|(index, name)| json!({
            "name":name,"x":freq,"y":levels[index],"color":colors[index],"width":2.0,"dash":"solid","visible":true,"y_axis":0
        })).collect::<Vec<_>>();
        let figure = json!({"kind":"figure","figure":{
            "title":format!("{label}: early vs late energy"),
            "x":{"label":"Frequency (Hz)","scale":"log","min":null,"max":null},
            "y":{"label":"Level vs full peak band (dB)","scale":"linear","min":null,"max":null},"y2":null,
            "series":series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
        },"tab":null});
        blocks.push(CaptureChart(figure));
        let early_late = freq
            .iter()
            .zip(&levels[1])
            .zip(&levels[2])
            .filter_map(|((frequency, early), late)| {
                (1000.0..=8000.0)
                    .contains(frequency)
                    .then_some(early - late)
            })
            .collect::<Vec<_>>();
        if freq[0] <= 1000.0 && *freq.last().unwrap_or(&0.0) >= 8000.0 && early_late.len() >= 2 {
            let ratio = early_late.iter().sum::<f64>() / early_late.len() as f64;
            if ratio.is_finite() {
                blocks.push(Paragraph(format!("1–8 kHz mean early minus late: {ratio:+.2} dB (arithmetic mean of third-octave differences).")));
            }
        } else {
            blocks.push(Paragraph(
                "1–8 kHz early/late ratio unavailable: full band support is missing.".into(),
            ));
        }
    }
    Ok(blocks)
}

fn capture_octave_t60_blocks(view: &Value) -> Result<Vec<DiagnosticBlock>, String> {
    let centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
    let band = view["valid_band_hz"]
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or("invalid octave T60 analysis contract")?;
    let lo = finite(&band[0]).ok_or("invalid octave T60 analysis contract")?;
    let hi = finite(&band[1]).ok_or("invalid octave T60 analysis contract")?;
    let threshold = finite(&view["min_r2"]).ok_or("invalid octave T60 analysis contract")?;
    if view["method"] != "octave_schroeder_t30_t20_v1"
        || !(0.0..=1.0).contains(&threshold)
        || threshold == 0.0
        || lo <= 0.0
        || hi <= lo
    {
        return Err("invalid octave T60 analysis contract".into());
    }
    for side in ["pre", "post"] {
        let rows = view[side]
            .as_array()
            .filter(|rows| rows.len() == centers.len())
            .ok_or("incomplete octave T60 bands")?;
        for (center, row) in centers.iter().zip(rows) {
            let valid = row["valid"]
                .as_bool()
                .ok_or("invalid octave T60 fit quality")?;
            let r2 = finite(&row["r2"]).ok_or("invalid octave T60 fit quality")?;
            let value = finite(&row["t60_s"]);
            let reason = row["reason"].as_str();
            if row["centre_hz"].as_i64() != Some(*center as i64)
                || !(0.0..=1.0).contains(&r2)
                || (valid
                    && (value.is_none_or(|value| value <= 0.0)
                        || !matches!(row["fit_range"].as_str(), Some("T20" | "T30"))
                        || r2 < threshold
                        || reason.is_none_or(|reason| !reason.is_empty())))
                || (!valid && (!row["t60_s"].is_null() || reason.is_none_or(str::is_empty)))
            {
                return Err(if valid {
                    "invalid accepted octave T60 fit"
                } else {
                    "invalid rejected octave T60 fit"
                }
                .into());
            }
        }
    }
    let scope = view["scope"]
        .as_str()
        .ok_or("invalid octave T60 analysis contract")?;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, Table};
    let mut blocks = vec![
        Heading("Capture IR octave T60 diagnostic".into()),
        Paragraph(scope.into()),
        Paragraph(format!(
            "Declared supported band: {}–{} Hz; minimum fit R²: {}. No invalid or EDT-only band is plotted; gaps are unavailable fits.",
            python_number(&band[0]),
            python_number(&band[1]),
            python_number(&view["min_r2"]),
        )),
    ];
    let valid_values = ["pre", "post"].iter().any(|side| {
        view[*side]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|row| row["valid"] == true))
    });
    if valid_values {
        let mut series = Vec::new();
        for (side, label, color) in [
            ("pre", "Baseline", "#3089c5"),
            ("post", "Candidate", "#c56816"),
        ] {
            let rows = view[side].as_array().unwrap();
            let octave_edges = (0..centers.len())
                .map(|index| {
                    if index == 0 {
                        centers[index] as f64 / 2.0_f64.sqrt()
                    } else {
                        ((centers[index - 1] * centers[index]) as f64).sqrt()
                    }
                })
                .chain(std::iter::once(
                    centers[centers.len() - 1] as f64 * 2.0_f64.sqrt(),
                ))
                .collect::<Vec<_>>();
            let mut x = Vec::with_capacity(rows.len() * 2);
            let mut y = Vec::with_capacity(rows.len() * 2);
            for (index, row) in rows.iter().enumerate() {
                let value = if row["valid"] == true {
                    row["t60_s"].clone()
                } else {
                    Value::Null
                };
                x.extend([json!(octave_edges[index]), json!(octave_edges[index + 1])]);
                y.extend([value.clone(), value]);
            }
            series.push(json!({"name":label,"x":x,"y":y,"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0}));
        }
        blocks.push(CaptureChart(json!({"kind":"figure","figure":{"title":"Baseline: measured octave-band T60","x":{"label":"Frequency (Hz)","scale":"log","min":20,"max":20000},"y":{"label":"T60 (s)","scale":"linear","min":0,"max":null},"y2":null,"series":series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true},"tab":null})));
    }
    let rows = centers
        .iter()
        .enumerate()
        .map(|(index, center)| {
            let pre = &view["pre"][index];
            let post = &view["post"][index];
            let cell = |row: &Value| {
                if row["valid"] == true {
                    format!("{:.3}", finite(&row["t60_s"]).unwrap_or_default())
                } else {
                    format!("n/a: {}", row["reason"].as_str().unwrap_or_default())
                }
            };
            let fit = |row: &Value| {
                format!(
                    "{}; R²={:.3}",
                    row["fit_range"].as_str().unwrap_or("n/a"),
                    finite(&row["r2"]).unwrap_or_default(),
                )
            };
            vec![
                center.to_string(),
                cell(pre),
                cell(post),
                fit(pre),
                fit(post),
            ]
        })
        .collect();
    blocks.push(Table(ReportTable {
        title: "Per-band T30/T20 estimates from matched IR captures".into(),
        note: String::new(),
        columns: [
            "Hz",
            "Baseline T60 (s)",
            "Candidate T60 (s)",
            "Baseline fit",
            "Candidate fit",
        ]
        .map(str::to_owned)
        .to_vec(),
        rows,
    }));
    Ok(blocks)
}

fn capture_room_mean_t60_blocks(groups: &[(String, String, Value)]) -> Vec<DiagnosticBlock> {
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, Table};
    let mut blocks = Vec::new();
    let mut keys = Vec::<String>::new();
    for (key, _, _) in groups {
        if !keys.contains(key) {
            keys.push(key.clone());
        }
    }
    let centers = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
    for key in keys {
        let entries = groups
            .iter()
            .filter(|(candidate, _, _)| *candidate == key)
            .collect::<Vec<_>>();
        let sources = entries
            .iter()
            .map(|(_, source, _)| source.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if sources.len() < 2 || sources.len() != entries.len() {
            continue;
        }
        let mut averages = [[(None, 0usize); 9]; 2];
        for (side_index, average_side) in averages.iter_mut().enumerate() {
            let side = if side_index == 0 { "pre" } else { "post" };
            for band_index in 0..9 {
                let values = entries
                    .iter()
                    .filter_map(|(_, _, view)| {
                        let row = &view[side][band_index];
                        (row["valid"] == true)
                            .then(|| finite(&row["t60_s"]))
                            .flatten()
                    })
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    let mean = values.iter().map(|v| *v / values.len() as f64).sum::<f64>();
                    average_side[band_index] = (Some(mean), values.len());
                }
            }
        }
        let seat = key.split('|').next().unwrap_or_default();
        let kind = key.split('|').nth(1).unwrap_or_default();
        let rate = key.split('|').nth(4).unwrap_or("0");
        blocks.push(Heading(format!(
            "Capture room mean octave T60: {}",
            seat.trim_matches('"')
        )));
        blocks.push(Paragraph(format!(
            "{}; {} distinct sources with matching declared settings, {rate} Hz sample rate, and capture band ({}). Means include accepted T30/T20 fits only. Counts vary by band. This advisory average does not establish a passive-room damping change.",
            kind.trim_matches('"'), sources.len(), sources.iter().cloned().collect::<Vec<_>>().join(", "),
        )));
        if averages.iter().flatten().any(|(value, _)| value.is_some()) {
            let mut series = Vec::new();
            for (side_index, label, color) in
                [(0, "Baseline", "#2864b4"), (1, "Candidate", "#c56816")]
            {
                series.push(json!({"name":label,"x":centers,"y":averages[side_index].iter().map(|(v,_)| v.map(Value::from).unwrap_or(Value::Null)).collect::<Vec<_>>(),"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0}));
            }
            blocks.push(CaptureChart(json!({"kind":"figure","figure":{"title":"Capture room mean octave T60","x":{"label":"Frequency (Hz)","scale":"log","min":null,"max":null},"y":{"label":"T60 (s)","scale":"linear","min":null,"max":null},"y2":null,"series":series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true},"tab":null})));
        }
        let rows = centers
            .iter()
            .enumerate()
            .map(|(index, center)| {
                let (pre, pre_n) = averages[0][index];
                let (post, post_n) = averages[1][index];
                let pre = pre
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into());
                let post = post
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into());
                vec![
                    center.to_string(),
                    pre,
                    pre_n.to_string(),
                    post,
                    post_n.to_string(),
                ]
            })
            .collect();
        blocks.push(Table(ReportTable {
            title: "Accepted source fits; n is the number of contributing sources".into(),
            note: String::new(),
            columns: [
                "Hz",
                "Baseline mean (s)",
                "Baseline n",
                "Candidate mean (s)",
                "Candidate n",
            ]
            .map(str::to_owned)
            .to_vec(),
            rows,
        }));
    }
    blocks
}

fn capture_decay_blocks(view: &Value, graph: &str) -> Result<Vec<DiagnosticBlock>, String> {
    if view["provenance"]["graph_identity"] != graph
        || view["method"] != "finite_window_octave_schroeder_v1"
    {
        return Err("decay graph identity or analysis method mismatch".into());
    }
    let bands = view["bands"]
        .as_array()
        .filter(|bands| !bands.is_empty())
        .ok_or("missing decay bands")?;
    let scope = view["scope"].as_str().ok_or("invalid decay scope")?;
    use DiagnosticBlock::{Heading, Paragraph};
    let mut blocks = vec![
        Heading("Matched octave decay".into()),
        Paragraph(scope.into()),
        Paragraph(format!(
            "Finite-window T20 slope extrapolation, not passive-room RT. Noise-window and fit budgets are operator declarations: {}",
            python_repr(&view["settings"])
        )),
    ];
    for band in bands {
        let center = finite(&band["center_hz"]).ok_or("unsupported decay octave")?;
        if ![63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0].contains(&center) {
            return Err("unsupported decay octave".into());
        }
        let trace = json!({"times_ms":band["times_ms"],"pre_decay":band["pre_db"],"post_decay":band["post_db"]});
        blocks.push(capture_trace_figure(
            &trace,
            "decay",
            "dB relative to the same baseline-band integrated energy",
            &format!(
                "Common-reference decay {} Hz",
                python_number(&band["center_hz"])
            ),
        )?);
        if !band["post_normalized_db"].is_null() {
            let normalized = json!({"times_ms":band["times_ms"],"pre_decay":band["pre_normalized_db"],"post_decay":band["post_normalized_db"]});
            blocks.push(capture_trace_figure(
                &normalized,
                "decay",
                "dB relative to each capture’s own initial band energy",
                &format!("Normalized decay {} Hz", python_number(&band["center_hz"])),
            )?);
        } else {
            blocks.push(Paragraph(
                "Candidate normalized decay unavailable: no nonzero energy reference.".into(),
            ));
        }
        for prefix in ["pre", "post"] {
            let fit = &band[format!("{prefix}_fit")];
            if fit.is_null() {
                let reason = band[format!("{prefix}_fit_unavailable")]
                    .as_str()
                    .unwrap_or("unspecified");
                blocks.push(Paragraph(format!("{prefix} fit unavailable: {reason}")));
            } else {
                let seconds = finite(&fit["extrapolated_60_db_seconds"])
                    .filter(|v| *v > 0.0)
                    .ok_or("invalid decay fit")?;
                let r2 = finite(&fit["r_squared"])
                    .filter(|v| (0.0..=1.0).contains(v))
                    .ok_or("invalid decay fit")?;
                blocks.push(Paragraph(format!("{prefix} finite-window extrapolation: {seconds:.4} s; R²={r2:.4}; fit diagnostics: {}", python_json(fit))));
            }
        }
    }
    Ok(blocks)
}

fn capture_waterfall_blocks(view: &Value) -> Result<Vec<DiagnosticBlock>, String> {
    if view["method"] != "hann_stft_waterfall_v1"
        || view["reference"] != "each_full_grid_peak"
        || finite(&view["window_ms"]) != Some(32.0)
        || finite(&view["hop_ms"]) != Some(2.0)
        || finite(&view["post_ms"]) != Some(500.0)
    {
        return Err("invalid capture waterfall contract".into());
    }
    let band = view["valid_band_hz"]
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or("invalid capture waterfall contract")?;
    let lo = finite(&band[0]).ok_or("invalid capture waterfall contract")?;
    let hi = finite(&band[1]).ok_or("invalid capture waterfall contract")?;
    if lo <= 0.0 || hi <= lo {
        return Err("invalid capture waterfall contract".into());
    }
    let scope = view["scope"]
        .as_str()
        .ok_or("invalid capture waterfall scope")?;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, Subheading, Table};
    let mut blocks = vec![
        Heading("Captured IR waterfall and resonance decay".into()),
        Paragraph(scope.into()),
        Paragraph(format!(
            "Declared supported band: {}–{} Hz. Each grid is referenced to its own full-grid peak; colors and levels do not compare absolute output between captures.",
            python_number(&band[0]),
            python_number(&band[1]),
        )),
    ];
    for (side, label) in [("pre", "Baseline"), ("post", "Candidate")] {
        let grid = &view[side]["grid"];
        let times = grid["times_ms"]
            .as_array()
            .ok_or("invalid capture waterfall samples")?;
        let freqs = grid["freqs_hz"]
            .as_array()
            .ok_or("invalid capture waterfall samples")?;
        let rows = grid["mags_db"]
            .as_array()
            .ok_or("invalid capture waterfall samples")?;
        if times.len() < 2
            || times.len() > 100
            || freqs.len() < 2
            || freqs.len() > 64
            || rows.len() != times.len()
        {
            return Err("invalid capture waterfall samples".into());
        }
        let t = finite_values(times).ok_or("invalid capture waterfall samples")?;
        let f = finite_values(freqs).ok_or("invalid capture waterfall samples")?;
        let mut z = Vec::new();
        for row in rows {
            let values = row
                .as_array()
                .filter(|v| v.len() == f.len())
                .and_then(|v| finite_values(v))
                .ok_or("invalid capture waterfall samples")?;
            if values.iter().any(|v| !(-100.01..=0.01).contains(v)) {
                return Err("invalid capture waterfall samples".into());
            }
            z.push(row.clone());
        }
        if t.windows(2).any(|p| p[0] >= p[1])
            || f.windows(2).any(|p| p[0] >= p[1])
            || t[0] < -5.01
            || *t.last().unwrap() > 500.01
            || f.iter().any(|v| *v < lo || *v > hi)
        {
            return Err("invalid capture waterfall samples".into());
        }
        let resonances = view[side]["resonances"]
            .as_array()
            .filter(|v| v.len() <= 64)
            .ok_or("invalid capture resonance list")?;
        for r in resonances {
            let rf = finite(&r["freq_hz"]).ok_or("invalid capture resonance values")?;
            let level = finite(&r["level_db"]).ok_or("invalid capture resonance values")?;
            if rf < f[0]
                || rf > *f.last().unwrap()
                || !(-100.01..=0.01).contains(&level)
                || (!r["decay_time_s"].is_null()
                    && finite(&r["decay_time_s"]).is_none_or(|v| v <= 0.0))
            {
                return Err("invalid capture resonance values".into());
            }
        }
        let mut ranked = resonances.iter().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, left), (_, right)| {
            let rank = |row: &Value| {
                (
                    !row["decay_time_s"].is_null(),
                    finite(&row["decay_time_s"]).unwrap_or(0.0),
                    finite(&row["level_db"]).unwrap_or(0.0),
                )
            };
            let left = rank(left);
            let right = rank(right);
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| right.2.total_cmp(&left.2))
        });
        let highlights = ranked
            .iter()
            .take(2)
            .filter_map(|(_, row)| {
                let frequency = finite(&row["freq_hz"])?;
                f.iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        (*left - frequency)
                            .abs()
                            .total_cmp(&(*right - frequency).abs())
                    })
                    .map(|(index, _)| json!(index))
            })
            .collect::<Vec<_>>();
        let highlight_series = ranked.iter().take(2).map(|(_, row)| {
            let frequency = finite(&row["freq_hz"]).unwrap_or_default();
            json!({"name":format!("{frequency:.1} Hz"),"x":[],"y":[],"color":format!("#{:06x}", resonance_color(frequency)),"width":2.0,"dash":"solid","visible":true,"y_axis":0})
        }).collect::<Vec<_>>();
        let grid_figure = json!({
            "kind":"grid",
            "figure":{
                "title":"Captured IR waterfall",
                "x":{"label":"Frequency (Hz)","scale":"log","min":freqs[0],"max":if *f.last().unwrap_or(&20000.0) > 20000.0 { json!(20000) } else { freqs[freqs.len()-1].clone() }},
                "y":{"label":"Time from broadband peak (ms)","scale":"linear","min":if t[0] < -1.0 { json!(-1) } else { times[0].clone() },"max":times[times.len()-1]},
                "y2":null,"series":highlight_series,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
            },
            "grid":{"x":freqs,"y":times,"z":z,"surface":true,"zmin":-60,"zmax":0,"highlights":highlights,"colormap":"turbo","show_surface":false,"show_contours":true},
            "tab":null
        });
        blocks.push(Subheading(label.into()));
        blocks.push(CaptureChart(grid_figure));
        let rows = resonances
            .iter()
            .map(|resonance| {
                let freq = finite(&resonance["freq_hz"]).unwrap_or_default();
                let level = finite(&resonance["level_db"]).unwrap_or_default();
                let decay = finite(&resonance["decay_time_s"])
                    .map(|value| python_general(value, 3))
                    .unwrap_or_else(|| "unavailable".into());
                vec![python_general(freq, 3), format!("{level:.2}"), decay]
            })
            .collect();
        blocks.push(Table(ReportTable {
            title: "Detected 60 ms peaks and fitted 20–200 ms decay".into(),
            note: String::new(),
            columns: [
                "Frequency (Hz)",
                "Level (dB re own grid peak)",
                "Fitted decay (s)",
            ]
            .map(str::to_owned)
            .to_vec(),
            rows,
        }));
        if !ranked.is_empty() {
            blocks.push(Paragraph("Highlighted: up to two longest fitted decays, then strongest 60 ms peak. Curves retain the full-grid peak reference; no fitted intercept is exported, so no extrapolated fit line is invented.".into()));
            let selected = ranked.iter().take(2).map(|(_, row)| {
                let frequency = finite(&row["freq_hz"]).unwrap_or_default();
                let label = finite(&row["decay_time_s"])
                    .map(|decay| format!("{frequency:.1} Hz — T60 {decay:.3} s"))
                    .unwrap_or_else(|| format!("{frequency:.1} Hz"));
                let color = format!("#{:06x}", resonance_color(frequency));
                let index = f.iter().enumerate().min_by(|(_, left), (_, right)| {
                    (*left - frequency).abs().total_cmp(&(*right - frequency).abs())
                }).map(|(index, _)| index).unwrap_or_default();
                let values = z
                    .iter()
                    .map(|row| json!(finite(&row[index]).unwrap_or_default()))
                    .collect::<Vec<_>>();
                json!({"name":label,"x":t,"y":values,"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0})
            }).collect::<Vec<_>>();
            let figure = json!({"kind":"figure","figure":{"title":format!("{label}: decay of highlighted resonances"),"x":{"label":"Time from broadband peak (ms)","scale":"linear","min":0,"max":500},"y":{"label":"Level relative to full grid peak (dB)","scale":"linear","min":-60,"max":0},"y2":null,"series":selected,"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true},"tab":null});
            blocks.push(CaptureChart(figure));
        }
    }
    Ok(blocks)
}

fn capture_wavelet_blocks(view: &Value) -> Result<Vec<DiagnosticBlock>, String> {
    if view["method"] != "complex_morlet_three_cycle_v1"
        || view["reference"] != "each_full_grid_peak"
        || finite(&view["cycles"]) != Some(3.0)
        || finite(&view["freqs_per_octave"]) != Some(6.0)
        || finite(&view["hop_ms"]) != Some(1.0)
        || !numeric_array_equals(&view["display_range_db"], &[-30.0, 0.0])
    {
        return Err("invalid capture wavelet contract".into());
    }
    let band = view["valid_band_hz"]
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or("invalid capture wavelet contract")?;
    let lo = finite(&band[0]).ok_or("invalid capture wavelet contract")?;
    let hi = finite(&band[1]).ok_or("invalid capture wavelet contract")?;
    if lo <= 0.0 || hi <= lo {
        return Err("invalid capture wavelet contract".into());
    }
    let scope = view["scope"]
        .as_str()
        .ok_or("invalid capture wavelet scope")?;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph};
    let mut blocks = vec![
        Heading("Captured IR three-cycle wavelet".into()),
        Paragraph(scope.into()),
        Paragraph("Blue = −30 dB, red = 0 dB relative to each capture’s own full wavelet-grid peak. Colors do not compare absolute output levels.".into()),
    ];
    for (side, _label) in [("pre", "Baseline"), ("post", "Candidate")] {
        let grid = &view[side];
        let freqs = grid["freqs_hz"]
            .as_array()
            .ok_or("invalid capture wavelet samples")?;
        let times = grid["times_ms"]
            .as_array()
            .ok_or("invalid capture wavelet samples")?;
        let rows = grid["mags_db"]
            .as_array()
            .ok_or("invalid capture wavelet samples")?;
        if freqs.len() < 2
            || freqs.len() > 64
            || times.len() < 2
            || times.len() > 100
            || rows.len() != freqs.len()
        {
            return Err("invalid capture wavelet samples".into());
        }
        let f = finite_values(freqs).ok_or("invalid capture wavelet samples")?;
        let t = finite_values(times).ok_or("invalid capture wavelet samples")?;
        let mut z = Vec::new();
        for row in rows {
            let v = row
                .as_array()
                .filter(|v| v.len() == t.len())
                .and_then(|v| finite_values(v))
                .ok_or("invalid capture wavelet samples")?;
            if v.iter().any(|x| !(-30.01..=0.01).contains(x)) {
                return Err("invalid capture wavelet samples".into());
            }
            z.push(v);
        }
        if f.windows(2).any(|p| p[0] >= p[1])
            || t.windows(2).any(|p| p[0] >= p[1])
            || f.iter().any(|v| *v < lo || *v > hi)
            || t[0] < -5.01
            || *t.last().unwrap() > 500.01
        {
            return Err("invalid capture wavelet samples".into());
        }
        let early_count = t
            .iter()
            .filter(|time| **time >= -1.0 && **time <= 15.0)
            .count();
        blocks.push(Paragraph(format!("Three-cycle wavelet; first 15 ms ({early_count} exported time samples). Log-frequency x-axis, time y-axis. Colours retain the full-grid peak reference. The cyan dashed ridge marks peak energy arrival time per frequency. Display cells reflect the exported sampling; zoom cannot restore omitted samples.")));
        let time_major = (0..times.len())
            .map(|time| {
                (0..freqs.len())
                    .map(|frequency| rows[frequency][time].clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let grid_figure = json!({
            "kind":"grid",
            "figure":{
                "title":"Captured three-cycle wavelet heatmap",
                "x":{"label":"Frequency (Hz)","scale":"log","min":freqs[0],"max":freqs[freqs.len()-1]},
                "y":{"label":"Time from broadband peak (ms)","scale":"linear","min":if t[0] < -1.0 { json!(-1) } else { times[0].clone() },"max":15},
                "y2":null,"series":[],"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
            },
            "grid":{"x":freqs,"y":times,"z":time_major,"surface":false,"zmin":-30,"zmax":0,"highlights":[]},
            "tab":null
        });
        blocks.push(CaptureChart(grid_figure));
    }
    Ok(blocks)
}

fn capture_noise_blocks(view: &Value, graph: &str) -> Result<Vec<DiagnosticBlock>, String> {
    let analysis = &view["analysis"];
    if analysis["method"] != "periodic_hann_welch_pressure_v1"
        || analysis["spectrum"]["provenance"]["graph_identity"] != graph
        || !matches!(
            view["evidence_kind"].as_str(),
            Some("synthetic_noise_capture" | "operator_declared_recorded_noise_capture")
        )
    {
        return Err("noise graph, evidence kind, or analysis method mismatch".into());
    }
    let freqs = analysis["bin_freqs_hz"]
        .as_array()
        .ok_or("invalid calibrated noise PSD")?;
    let powers = analysis["pressure_psd_pa2_per_hz"]
        .as_array()
        .ok_or("invalid calibrated noise PSD")?;
    if freqs.is_empty() || freqs.len() != powers.len() || freqs.len() > 32768 {
        return Err("invalid calibrated noise PSD".into());
    }
    let x = finite_values(freqs).ok_or("invalid calibrated noise PSD")?;
    let p = finite_values(powers).ok_or("invalid calibrated noise PSD")?;
    if x.iter().any(|v| *v <= 0.0)
        || x.windows(2).any(|w| w[0] >= w[1])
        || p.iter().any(|v| *v < 0.0)
    {
        return Err("invalid calibrated noise PSD".into());
    }
    let kind = view["evidence_kind"]
        .as_str()
        .ok_or("invalid calibrated noise evidence")?;
    let conditions = view["conditions"]
        .as_str()
        .ok_or("invalid calibrated noise conditions")?;
    let scope = analysis["scope"]
        .as_str()
        .ok_or("invalid calibrated noise scope")?;
    use DiagnosticBlock::{CaptureChart, Heading, Paragraph, PreformattedDisclosure, Table};
    let mut blocks = vec![
        Heading("Calibrated ambient noise".into()),
        Paragraph(format!("{kind}; {conditions}")),
        Paragraph(scope.into()),
        Paragraph(format!(
            "Observation: {} s; bin spacing: {} Hz; Hann ENBW: {} Hz. Bin spacing is not resolving power. No self-noise subtraction.",
            python_json(&analysis["duration_seconds"]),
            python_json(&analysis["bin_spacing_hz"]),
            python_json(&analysis["window_enbw_hz"]),
        )),
        PreformattedDisclosure {
            title: "Numeric calibration and acquisition limitations".into(),
            text: serde_json::to_string_pretty(&analysis["calibration"]).unwrap_or_default(),
        },
    ];
    let levels = p
        .iter()
        .map(|v| {
            if *v > 0.0 {
                Some(10.0 * (v.log10() - (20e-6_f64.powi(2)).log10()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let positive = levels.iter().any(Option::is_some);
    if positive {
        let figure = json!({"kind":"figure","figure":{"title":"Calibrated ambient pressure PSD","x":{"label":"Frequency (Hz)","scale":"log","min":null,"max":null},"y":{"label":"dB re (20 µPa)²/Hz","scale":"linear","min":null,"max":null},"y2":null,"series":[{"name":"Pressure PSD","x":x,"y":levels,"color":"#2864b4","width":2.0,"dash":"solid","visible":true,"y_axis":0}],"hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true},"tab":null});
        blocks.push(CaptureChart(figure));
    } else {
        blocks.push(Paragraph("No finite noise SPL: all estimated supported powers are zero. This is not proof of a noiseless room.".into()));
    }
    let spectrum = &analysis["spectrum"];
    let centers = spectrum["freqs"]
        .as_array()
        .ok_or("invalid noise octave arrays")?;
    let db = spectrum["noise_spl_db"]
        .as_array()
        .ok_or("invalid noise octave arrays")?;
    let edges = analysis["octave_edges_hz"]
        .as_array()
        .ok_or("invalid noise octave arrays")?;
    let bins = analysis["octave_bin_centers_hz"]
        .as_array()
        .ok_or("invalid noise octave arrays")?;
    if centers.len() != db.len()
        || centers.len() != edges.len()
        || centers.len() != bins.len()
        || centers.len() > 10
    {
        return Err("invalid noise octave arrays".into());
    }
    let mut rows = Vec::new();
    for i in 0..centers.len() {
        let level = finite(&db[i]).ok_or("invalid noise octave arrays")?;
        let edge = edges[i]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or("invalid noise octave support")?;
        let extent = bins[i]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or("invalid noise octave support")?;
        let elo = finite(&edge[0]).ok_or("invalid noise octave support")?;
        let ehi = finite(&edge[1]).ok_or("invalid noise octave support")?;
        let blo = finite(&extent[0]).ok_or("invalid noise octave support")?;
        let bhi = finite(&extent[1]).ok_or("invalid noise octave support")?;
        if finite(&centers[i]).is_none_or(|c| c <= 0.0)
            || elo <= 0.0
            || elo >= ehi
            || blo < elo
            || blo > bhi
            || bhi >= ehi
        {
            return Err("invalid noise octave support".into());
        }
        rows.push(vec![
            python_general(finite(&centers[i]).unwrap_or_default(), 6),
            format!("{}–{}", python_general(elo, 4), python_general(ehi, 4)),
            format!("{}–{}", python_general(blo, 4), python_general(bhi, 4)),
            python_general(level, 4),
        ]);
    }
    blocks.push(Table(ReportTable {
        title: "FFT-bin-summed nominal octave levels, unweighted dB SPL re 20 µPa; not certified octave filters".into(),
        note: String::new(),
        columns: ["Center (Hz)", "Nominal band (Hz)", "Summed bin centers (Hz)", "dB SPL"].map(str::to_owned).to_vec(),
        rows,
    }));
    if let Some(unavailable) = analysis["unavailable_bands"].as_object() {
        for (band, reason) in unavailable {
            blocks.push(Paragraph(format!(
                "Noise {band} Hz unavailable: {}",
                python_string(reason)
            )));
        }
    }
    Ok(blocks)
}

fn finite_values(values: &[Value]) -> Option<Vec<f64>> {
    values.iter().map(finite).collect()
}

fn numeric_array_equals(value: &Value, expected: &[f64]) -> bool {
    value.as_array().is_some_and(|values| {
        values.len() == expected.len()
            && values
                .iter()
                .zip(expected)
                .all(|(value, expected)| finite(value) == Some(*expected))
    })
}

fn python_json(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".into(),
        _ => value.to_string(),
    }
}

fn python_number(value: &Value) -> String {
    finite(value)
        .map(|number| {
            if number.fract() == 0.0 {
                format!("{number:.0}")
            } else {
                number.to_string()
            }
        })
        .unwrap_or_else(|| python_json(value))
}

fn python_general(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let text = format!("{:.*e}", precision.saturating_sub(1), value);
        if let Some((mantissa, exp)) = text.split_once('e') {
            let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
            let exponent = exp.parse::<i32>().unwrap_or_default();
            return format!("{mantissa}e{exponent:+03}");
        }
        text
    } else {
        let decimals = precision.saturating_sub(1 + exponent.max(0) as usize);
        let text = format!("{value:.decimals$}");
        if decimals == 0 {
            text
        } else {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&Value::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn json_python_separators(value: &Value) -> String {
    let compact = serde_json::to_string(value).unwrap_or_else(|_| "null".into());
    let mut result = String::with_capacity(compact.len() + compact.len() / 8);
    let mut in_string = false;
    let mut escaped = false;
    for character in compact.chars() {
        if in_string {
            result.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
        } else {
            if character == '"' {
                in_string = true;
            }
            result.push(character);
            if matches!(character, ',' | ':') {
                result.push(' ');
            }
        }
    }
    result
}

fn acceptance_magnitude_figure(view: &Value, label: &str) -> Result<Value, String> {
    let frequencies = view["freqs"]
        .as_array()
        .ok_or("magnitude traces are missing")?;
    let pre = view["pre_db"]
        .as_array()
        .ok_or("magnitude traces are missing")?;
    let post = view["post_db"]
        .as_array()
        .ok_or("magnitude traces are missing")?;
    if frequencies.len() < 2
        || frequencies.len() != pre.len()
        || frequencies.len() != post.len()
        || frequencies.len() > 100_000
    {
        return Err("magnitude traces are empty, unaligned, or oversized".into());
    }
    let numeric = |values: &[Value]| -> Result<Vec<f64>, String> {
        values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|number| number.is_finite())
                    .ok_or_else(|| "magnitude traces contain invalid values".into())
            })
            .collect()
    };
    let x = numeric(frequencies)?;
    let before = numeric(pre)?;
    let after = numeric(post)?;
    if x[0] <= 0.0 || x.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("magnitude frequency axis is invalid".into());
    }
    let mut all_levels = before.iter().chain(&after);
    let first = all_levels
        .next()
        .copied()
        .ok_or("magnitude traces are missing")?;
    let (low, high) = all_levels.fold((first, first), |(lo, hi), value| {
        (lo.min(*value), hi.max(*value))
    });
    if !(high - low).max(1.0).is_finite() {
        return Err("magnitude range overflow".into());
    }
    let axis =
        |label: &str, scale: &str| json!({"label":label,"scale":scale,"min":null,"max":null});
    let series = |name: &str, values: &[f64], color: &str| json!({"name":name,"x":x,"y":values,"color":color,"width":2.0,"dash":"solid","visible":true,"y_axis":0});
    Ok(json!({
        "kind":"figure",
        "figure":{
            "title":label,
            "x":axis("Frequency (Hz)","log"),
            "y":axis("Magnitude (dB)","linear"),
            "y2":null,
            "series":[series("Before",&before,"#2864b4"),series("Retained prediction after",&after,"#c56816")],
            "hlines":[],"vlines":[],"xranges":[],"annotations":[],"legend":true
        },
        "tab":null
    }))
}

fn acceptance_views_html(request: &Value) -> String {
    section_fragment(acceptance_views_section(request))
        .expect("acceptance views are serialized by AutoEQ")
}

fn acceptance_views_section(request: &Value) -> Value {
    use DiagnosticBlock::{AcceptanceChart, EvidenceList, Heading, Paragraph};
    let data = &request["data"];
    let label = request["label"].as_str().unwrap_or_default();
    let section = |blocks: Vec<DiagnosticBlock>| {
        let mut section = diagnostic_section(blocks);
        section["introduction"] = json!(DiagnosticIntroduction {
            title: labelled_heading("Acceptance diagnostics", label),
            text: String::new(),
            style: DiagnosticSectionStyle::AcceptanceViews,
        });
        section
    };
    let unavailable = |reason: &str| section(vec![Paragraph(format!("Unavailable: {reason}"))]);
    let evidence = &data["correction_decisions"]["acceptance_evidence"];
    if evidence.is_null() {
        return unavailable(
            "no production acceptance views were recorded (legacy or unfinalized output).",
        );
    }
    if request["binding_verified"] != true {
        return unavailable(
            request["binding_reason"]
                .as_str()
                .unwrap_or("Payload binding unavailable."),
        );
    }
    if request["acceptance_evidence_verified"] != true {
        return unavailable(
            request["acceptance_evidence_reason"]
                .as_str()
                .unwrap_or("Acceptance-view payload binding is unavailable."),
        );
    }
    let graph_identity = request["binding_graph_identity"]
        .as_str()
        .unwrap_or_default();
    if graph_identity.is_empty() {
        return unavailable("Payload binding has no graph identity.");
    }
    if evidence["version"] != "acceptance-evidence-v1" {
        return unavailable("unsupported acceptance-evidence version");
    }
    let payload = &evidence["payload"];
    let channels = payload["channels"].as_object();
    let data_channels = data["channels"].as_object();
    let (Some(channels), Some(data_channels)) = (channels, data_channels) else {
        return unavailable("acceptance diagnostics do not cover the delivered channel set");
    };
    if channels.keys().collect::<std::collections::BTreeSet<_>>()
        != data_channels
            .keys()
            .collect::<std::collections::BTreeSet<_>>()
    {
        return unavailable("acceptance diagnostics do not cover the delivered channel set");
    }
    let mut blocks = vec![
        Paragraph("Retained predictions and serialized DSP controls—not recorded playback, physical safety certification, or demonstrated listener benefit. Existing measurement conditioning is retained; additional smoothing cannot recover raw detail.".into()),
        Paragraph(format!("Graph: {graph_identity}")),
    ];
    for (channel, entry) in channels {
        if entry["source_id"].as_str() != Some(channel)
            || entry["evidence_kind"] != "retained_prediction"
        {
            return unavailable("unsupported source or evidence classification");
        }
        blocks.push(Heading(channel.clone()));
        let seat = entry
            .get("seat_provenance")
            .and_then(Value::as_str)
            .unwrap_or("unavailable");
        blocks.push(Paragraph(format!("Seat provenance: {seat}")));
        let Some(bundle) = entry.get("bundle").filter(|value| value.is_object()) else {
            return unavailable("acceptance bundle is missing or malformed");
        };
        let Some(settings) = bundle.get("settings").filter(|value| value.is_object()) else {
            return unavailable("acceptance settings are missing or malformed");
        };
        let rate = python_string(&settings["sample_rate_hz"]);
        let Some(normalization) = settings["normalization"].as_str() else {
            return unavailable("acceptance normalization is missing or malformed");
        };
        let Some(window) = settings["window"].as_str() else {
            return unavailable("acceptance window is missing or malformed");
        };
        blocks.push(Paragraph(format!(
            "Rate: {rate} Hz; levels: {normalization}; window: {window}"
        )));
        for (key, title, view) in [
            (
                "magnitude",
                "General magnitude (additional 1/12-octave smoothing)",
                &bundle["magnitude"],
            ),
            (
                "fine_magnitude",
                "Fine magnitude (additional 1/24-octave smoothing)",
                &entry["fine_magnitude"],
            ),
        ] {
            if !view.is_null() {
                if view["provenance"]["graph_identity"] != graph_identity {
                    return unavailable(&format!("{key} graph provenance mismatch"));
                }
                match acceptance_magnitude_figure(view, title) {
                    Ok(chart) => blocks.push(AcceptanceChart(chart)),
                    Err(error) => return unavailable(&error),
                }
            }
        }
        let disposition = &bundle["disposition"];
        if !disposition.is_null() {
            if disposition["graph_identity"] != graph_identity {
                return unavailable("DSP disposition graph identity mismatch");
            }
            blocks.push(Paragraph(
                "Serialized DSP disposition (not calibrated output capability):".into(),
            ));
            let controls = disposition["channels"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|control| {
                    format!(
                        "{}: gain {} dB; delay {} ms; inverted {}",
                        python_string(&control["channel"]),
                        python_string(&control["gain_db"]),
                        python_string(&control["delay_ms"]),
                        python_string(&control["inverted"]),
                    )
                })
                .collect();
            blocks.push(EvidenceList(controls));
        }
        let Some(inventory) = entry.get("unavailable").and_then(Value::as_object) else {
            return unavailable("acceptance unavailable-view inventory is missing or malformed");
        };
        blocks.push(Paragraph("Unavailable evidence:".into()));
        blocks.push(EvidenceList(
            inventory
                .iter()
                .map(|(view, reason)| format!("{view}: {}", python_string(reason)))
                .collect(),
        ));
        blocks.push(Paragraph(
            "No complete acoustic/headroom acceptance is established by these diagnostics.".into(),
        ));
    }
    section(blocks)
}

fn evidence_band(value: &Value) -> Option<(f64, f64)> {
    let band = value.as_array()?;
    if band.len() != 2 {
        return None;
    }
    let low = finite(&band[0])?;
    let high = finite(&band[1])?;
    (low > 0.0 && low < high).then_some((low, high))
}

fn format_general(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= precision as i32 {
        let formatted = format!("{:.*e}", precision.saturating_sub(1), value);
        let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        let parsed = exponent.parse::<i32>().unwrap_or(0);
        let sign = if parsed >= 0 { '+' } else { '-' };
        format!("{mantissa}e{sign}{:02}", parsed.unsigned_abs())
    } else {
        let decimals = precision.saturating_sub((exponent + 1).max(0) as usize);
        format!("{:.*}", decimals, value)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn evidence_band_label(band: (f64, f64)) -> String {
    for precision in [6, 10, 15, 17] {
        let low = format_general(band.0, precision);
        let high = format_general(band.1, precision);
        if low != high {
            return format!("{low}–{high} Hz");
        }
    }
    format!(
        "{}–{} Hz",
        format_general(band.0, 17),
        format_general(band.1, 17)
    )
}

fn quantity(value: &Value, unit: &str) -> Option<String> {
    finite(value).map(|number| {
        format!("{} {unit}", format_general(number, 6))
            .trim()
            .to_string()
    })
}

fn verification_items(metadata: &Value) -> Vec<String> {
    let comparisons = metadata
        .get("playback_comparisons")
        .filter(|value| value.as_array().is_some_and(|items| !items.is_empty()))
        .unwrap_or(&metadata["verification"]);
    let Some(comparisons) = comparisons.as_array().filter(|items| !items.is_empty()) else {
        return Vec::new();
    };
    let kind_text = [
        ("stationary_ir", "acoustic capture"),
        (
            "spatial_magnitude",
            "acoustic magnitude capture (no timing reference)",
        ),
        ("direct_sound", "acoustic direct-sound capture"),
        (
            "simulated_backend",
            "exported-backend simulation (not an acoustic recording)",
        ),
        ("unknown", "uncharacterized capture"),
    ];
    let mut items = Vec::new();
    for comparison in comparisons {
        if !comparison.is_object() {
            continue;
        }
        let kind = kind_text
            .iter()
            .find(|(value, _)| comparison["capture_kind"] == *value)
            .map(|(_, label)| *label)
            .unwrap_or("uncharacterized capture");
        let get_or = |name: &str, fallback: &str| {
            comparison[name]
                .as_str()
                .filter(|value| !value.is_empty())
                .unwrap_or(fallback)
                .to_string()
        };
        let baseline = get_or("baseline_graph_identity", "identity missing");
        let candidate = get_or("candidate_graph_identity", "identity missing");
        let stimulus = get_or("stimulus_hash", "stimulus hash missing");
        let assessed = comparison["baseline_graph_identity"]
            .as_str()
            .is_some_and(|value| !value.trim().is_empty())
            && comparison["candidate_graph_identity"]
                .as_str()
                .is_some_and(|value| !value.trim().is_empty())
            && baseline != candidate
            && comparison["stimulus_hash"]
                .as_str()
                .is_some_and(|value| !value.trim().is_empty());
        let verdict = if assessed {
            "recorded"
        } else {
            "unassessed: insufficient evidence, no promotion"
        };
        let seats = comparison["seat_ids"]
            .as_array()
            .map(|items| items.iter().map(python_string).collect::<Vec<_>>())
            .unwrap_or_default();
        let seat_text = if seats.is_empty() {
            String::new()
        } else {
            format!(" Seats: {}.", seats.join(", "))
        };
        items.push(format!(
            "Playback check ({kind}; {}): {verdict}. Source {}.{seat_text} Baseline {baseline}; candidate {candidate}; stimulus {stimulus}.",
            get_or("processing_state", "unstated signal level"),
            get_or("source_id", "unstated source")
        ));
    }
    if metadata["listening_evidence"].is_object() {
        let listening = &metadata["listening_evidence"];
        items.push(format!(
            "Recorded listening outcome: {} (protocol {}). A modeled perceptual score is not a listener result; an inconclusive or unassessed outcome is not listening benefit.",
            listening["result"].as_str().filter(|value| !value.is_empty()).unwrap_or("unassessed"),
            listening["protocol_id"].as_str().filter(|value| !value.is_empty()).unwrap_or("unstated protocol")
        ));
    }
    items
}

fn measurement_conditioning_history(stage: &Value) -> Vec<String> {
    let names = [
        (
            "source_overlap_alignment",
            "Aligned source grids within shared support",
        ),
        (
            "source_spatial_power_rms",
            "Averaged spatial magnitudes in the power domain (no averaged phase)",
        ),
        (
            "source_coherent_pressure_mean",
            "Computed a declared coherent pressure mean",
        ),
        (
            "roomeq_dense_grid_conditioning",
            "Executed the dense-grid conditioning policy",
        ),
    ];
    let mut details = Vec::new();
    for check in stage["checks"].as_array().into_iter().flatten() {
        if !check.is_object() || check["passed"] != true {
            continue;
        }
        let parsed = check["diagnostic"]
            .as_str()
            .and_then(|diagnostic| serde_json::from_str::<Value>(diagnostic).ok());
        let Some(payload) =
            parsed.filter(|payload| payload.is_object() && payload["receipt"].is_object())
        else {
            continue;
        };
        let Some(entries) = payload["receipt"]["entries"].as_array() else {
            continue;
        };
        let channel = payload["channel"].as_str().unwrap_or("unknown channel");
        if payload["source_snapshot_binding"] == "verified_parsed_curve_snapshot" {
            details.push(format!("Measurement preparation for {channel}: native response identities match the frozen input snapshot in source order; this does not authenticate the original recording."));
        }
        if entries.is_empty() {
            details.push(format!("Measurement preparation for {channel}: no numerical loading change recorded; this does not describe later processing."));
        }
        for entry in entries {
            if !entry.is_object() {
                continue;
            }
            let operation = entry["operation"].as_str().unwrap_or("unknown operation");
            let mut description = names
                .iter()
                .find(|(known, _)| *known == operation)
                .map(|(_, text)| *text)
                .map(str::to_string)
                .unwrap_or_else(|| format!("Recorded operation: {operation}"));
            if let Some(band) = evidence_band(&entry["parameters"]["declared_valid_band_hz"]) {
                description.push_str(&format!(
                    "; declared usable band {}",
                    evidence_band_label(band)
                ));
            }
            if !entry["parameters"]["input_bins"].is_null()
                && !entry["parameters"]["output_bins"].is_null()
            {
                description.push_str(&format!(
                    "; {} → {} bins",
                    python_string(&entry["parameters"]["input_bins"]),
                    python_string(&entry["parameters"]["output_bins"])
                ));
            }
            details.push(format!(
                "Measurement preparation for {channel}: {description}."
            ));
        }
    }
    if !details.is_empty() {
        details.push("Measurement preparation is partial recorded history, not proof of capture/calibration validity or complete conditioning lineage.".into());
    }
    details
}

fn output_loss_table(quality: &Value) -> Option<ReportTable> {
    let entries = quality["useful_output"].as_array()?;
    let mut rows = Vec::new();
    for entry in entries {
        if !entry.is_object() {
            continue;
        }
        let scope = format!(
            "{} / {} / seat {}",
            entry["logical_input"]
                .as_str()
                .filter(|v| !v.is_empty())
                .unwrap_or("Unknown input"),
            entry["partition"]
                .as_str()
                .filter(|v| !v.is_empty())
                .unwrap_or("unknown partition"),
            entry
                .get("seat_index")
                .map(python_string)
                .unwrap_or_else(|| "?".into())
        );
        let mean = quantity(&entry["mean_level_change_db"], "dB");
        let unexplained = quantity(&entry["unexplained_loss_rms_db"], "dB");
        let worst = quantity(&entry["worst_unexplained_loss_db"], "dB");
        let worst_hz = number(&entry["worst_loss_frequency_hz"]);
        let mut detail = format!(
            "Mean level change: {}. Unexplained loss (RMS): {}.",
            mean.as_deref().unwrap_or("unknown"),
            unexplained.as_deref().unwrap_or("unknown")
        );
        if let Some(worst) = worst {
            detail.push_str(&format!(" Worst unexplained loss: {worst}"));
            if let Some(hz) = worst_hz {
                detail.push_str(&format!(" at {hz} Hz."));
            } else {
                detail.push('.');
            }
        }
        detail.push_str(" Raw output evidence; display normalization cannot hide this loss.");
        rows.push(vec![scope, detail]);
    }
    (!rows.is_empty()).then(|| ReportTable {
        title: "Recorded raw output loss (unnormalized)".into(),
        note: String::new(),
        columns: ["Seat", "Recorded loss"].map(str::to_owned).to_vec(),
        rows,
    })
}

fn k4_frequency_text(record: &Value) -> String {
    if let Some(band) = evidence_band(&record["frequency_band_hz"]) {
        return evidence_band_label(band);
    }
    finite(&record["filter_center_hz"])
        .filter(|value| *value > 0.0)
        .map(|value| format!("{} Hz (filter center)", format_general(value, 6)))
        .unwrap_or_else(|| "Frequency unavailable".into())
}

fn k4_scope_text(record: &Value) -> String {
    let logical = record["logical_input"]
        .as_str()
        .filter(|value| !value.is_empty())
        .unwrap_or("Unknown input");
    let physical = record["physical_output"]
        .as_str()
        .filter(|value| !value.is_empty())
        .unwrap_or("unknown output");
    let mut scope = format!("{logical} / {physical}");
    let seats = record["seat_refs"]
        .as_array()
        .into_iter()
        .flatten()
        .map(python_string)
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    if !seats.is_empty() {
        scope.push_str(&format!(" / seat {}", seats.join(", ")));
    }
    let measurements = record["measurement_refs"]
        .as_array()
        .into_iter()
        .flatten()
        .map(python_string)
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    if !measurements.is_empty() {
        scope.push_str(&format!(" (measured: {})", measurements.join(", ")));
    }
    scope
}

fn k4_decision_text(record: &Value) -> String {
    let action = match record["action"].as_str() {
        Some("equalize") => "Equalize",
        Some("phase_correct") => "Phase correction",
        Some("gain_adjust") => "Gain adjustment",
        Some("reroute") => "Reroute",
        Some("prune") => "Prune",
        _ => "Recorded action unavailable",
    };
    let status = match record["status"].as_str() {
        Some("applied") => "Applied",
        Some("already_acceptable") => "Already acceptable (no correction needed)",
        Some("insufficient_evidence") => "Insufficient evidence (no correction applied)",
        Some("outside_scope") => "Outside scope (no correction applied)",
        Some("constrained") => "Constrained (partial correction only; remainder withheld)",
        Some("reverted") => "Reverted (attempted correction rolled back; not delivered)",
        Some("unresolved") => "Unresolved (no decision reached)",
        Some("advisory") => "Advisory nomination only (nothing applied or removed)",
        _ => "Recorded status unavailable",
    };
    if record["stage"] == "final" {
        format!("{action} — {status} (final)")
    } else {
        format!("{action} — {status} (provisional; not a delivery claim)")
    }
}

fn k4_reason_text(record: &Value) -> String {
    let reason_names = [
        ("serialized_final_gain", "Gain present in the delivered DSP"),
        (
            "final_electrical_headroom",
            "Attenuation for the configured digital headroom ceiling and any declared physical-drive limits",
        ),
        (
            "final_channel_level_alignment",
            "Final channel-level alignment",
        ),
        (
            "branch_gain_not_acoustic_benefit_or_net_parallel_gain",
            "Scalar gain on this input/output branch; not net parallel-path gain or measured acoustic benefit",
        ),
        (
            "joint_array_proposal_reverted",
            "Joint subwoofer gain/delay proposal reverted",
        ),
        (
            "joint_shared_eq_proposal_reverted",
            "Shared subwoofer EQ proposal reverted",
        ),
        (
            "stage_history_not_final_route_acceptance",
            "This is stage history, not approval of later routing or recorded playback",
        ),
    ];
    let codes = record["reason_codes"]
        .as_array()
        .into_iter()
        .flatten()
        .map(python_string)
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    let labels = codes
        .iter()
        .map(|code| {
            reason_names
                .iter()
                .find(|(known, _)| *known == code)
                .map(|(_, label)| *label)
                .unwrap_or(code)
        })
        .collect::<Vec<_>>();
    let mut reason = if labels.is_empty() {
        "No reason recorded.".into()
    } else {
        format!("Recorded reason: {}.", labels.join(", "))
    };
    reason.push_str(&format!(
        " Confidence: {}.",
        record["confidence"]
            .as_str()
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
    ));
    let quantity_list = |key: &str| -> String {
        record[key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|item| {
                let value = quantity(&item["value"], item["unit"].as_str().unwrap_or(""))?;
                let name = item["name"]
                    .as_str()
                    .filter(|name| !name.trim().is_empty())?;
                Some(format!("{}: {}", name.trim(), value))
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let observed = quantity_list("observed");
    if !observed.is_empty() {
        reason.push_str(&format!(" Observed: {observed}."));
    }
    let limits = quantity_list("limits");
    if !limits.is_empty() {
        reason.push_str(&format!(" Limit: {limits}."));
    }
    let evidence = record["evidence_refs"]
        .as_array()
        .into_iter()
        .flatten()
        .map(python_string)
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    if evidence.is_empty() {
        reason.push_str(" Evidence: unavailable.");
    } else {
        reason.push_str(&format!(" Evidence: {}.", evidence.join(", ")));
    }
    for (key, template) in [
        (
            "related_decision_ids",
            "Linked record: {}. Both the applied partial correction and its constrained remainder stay visible.",
        ),
        (
            "supersedes_ids",
            "Supersedes: {}. The superseded attempt is not delivered correction.",
        ),
    ] {
        let linked = record[key]
            .as_array()
            .into_iter()
            .flatten()
            .map(python_string)
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>();
        if !linked.is_empty() {
            reason.push(' ');
            reason.push_str(&template.replace("{}", &linked.join(", ")));
        }
    }
    reason
}

fn correction_ledger(
    data: &Value,
    verified: bool,
    binding_reason: &str,
) -> (Vec<DiagnosticBlock>, Vec<String>) {
    let ledger = &data["correction_decisions"];
    let Some(decisions) = ledger["decisions"]
        .as_array()
        .filter(|items| !items.is_empty())
    else {
        return (Vec::new(), Vec::new());
    };
    let metadata = &data["metadata"];
    let acceptance = &metadata["correction_acceptance"];
    let delivery_blocked =
        acceptance["decision"] == "identity_fallback" || acceptance["outcome"] == "rejected";
    let delivered = metadata["delivered_graph_identity"]
        .as_str()
        .filter(|value| !value.trim().is_empty());
    let final_records = decisions
        .iter()
        .filter(|record| record.is_object() && record["stage"] == "final")
        .collect::<Vec<_>>();
    let identities = final_records
        .iter()
        .filter_map(|record| {
            record["final_graph_identity"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
        })
        .collect::<std::collections::BTreeSet<_>>();
    let disagree = identities.len() > 1;
    let payload_identity = ledger["payload_binding"]["graph_identity"]
        .as_str()
        .unwrap_or("");
    let binding_invalid = !verified
        || disagree
        || final_records
            .iter()
            .any(|record| record["final_graph_identity"].as_str() != Some(payload_identity))
        || delivered.is_some_and(|value| value != payload_identity);
    let mut rows = Vec::new();
    let mut history = Vec::new();
    for record in decisions {
        if !record.is_object() {
            history.push(
                "Unverified decision record: unreadable entry; not delivered correction.".into(),
            );
            continue;
        }
        let Some(decision_id) = record["decision_id"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
        else {
            history.push(
                "Unverified decision record: missing decision ID; not delivered correction.".into(),
            );
            continue;
        };
        let scope = k4_scope_text(record);
        let frequency = k4_frequency_text(record);
        let mut decision = k4_decision_text(record);
        let mut reason = k4_reason_text(record);
        if record["stage"] != "final" {
            history.push(format!("Provisional record {decision_id} ({scope}; {frequency}; {decision}): {reason} Final reconciled records above take precedence."));
            continue;
        }
        let identity = record["final_graph_identity"].as_str().unwrap_or("");
        if delivery_blocked
            && matches!(
                record["status"].as_str(),
                Some("applied" | "already_acceptable" | "constrained")
            )
        {
            decision.push_str(" (not delivered correction)");
            reason
                .push_str(" Final rejection or identity fallback overrides this candidate record.");
        }
        let row_verified = verified
            && identity == payload_identity
            && !identity.trim().is_empty()
            && !disagree
            && delivered.is_none_or(|value| identity == value);
        if !row_verified {
            decision.push_str(" (unverified; not a delivery claim)");
            reason.push(' ');
            reason.push_str(binding_reason);
            if identity.trim().is_empty() {
                reason.push_str(" Unverified: final record without a delivered-graph identity is not a delivery claim.");
            } else if disagree {
                reason.push_str(" Unverified: final records disagree on the delivered-graph identity; the explanation cannot be bound to one delivered graph.");
            } else if delivered.is_some_and(|value| identity != value) || verified {
                reason.push_str(" Unverified: record identity does not match the recorded delivered graph; not displayed as applied.");
            }
        }
        rows.push(vec![scope, frequency, decision, reason]);
    }
    if rows.is_empty() && history.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let statuses: [(&str, &[&str]); 3] = [
        ("applied", &["applied", "already_acceptable", "constrained"]),
        ("reverted", &["reverted"]),
        (
            "withheld",
            &[
                "insufficient_evidence",
                "outside_scope",
                "unresolved",
                "advisory",
            ],
        ),
    ];
    let set_labels = [
        ("applied", "applied delivery claims"),
        ("reverted", "reverted (rolled back; not delivered)"),
        ("provisional", "provisional history (not delivery claims)"),
        ("withheld", "withheld (final rows without delivery)"),
    ];
    let mut set_ids: std::collections::BTreeMap<&str, Vec<String>> =
        ["applied", "reverted", "provisional", "withheld"]
            .into_iter()
            .map(|key| (key, Vec::new()))
            .collect();
    let mut set_reasons: std::collections::BTreeMap<&str, Vec<String>> =
        ["applied", "reverted", "provisional", "withheld"]
            .into_iter()
            .map(|key| (key, Vec::new()))
            .collect();
    for record in decisions {
        if !record.is_object() {
            set_ids
                .get_mut("provisional")
                .unwrap()
                .push("unreadable entry".into());
            continue;
        }
        let id = record["decision_id"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("missing decision ID")
            .to_string();
        let bucket = if record["stage"] != "final" {
            "provisional"
        } else {
            statuses
                .iter()
                .find(|(_, allowed)| allowed.iter().any(|status| record["status"] == *status))
                .map(|(bucket, _)| *bucket)
                .unwrap_or("withheld")
        };
        let bucket = if (delivery_blocked || binding_invalid) && bucket == "applied" {
            "withheld"
        } else {
            bucket
        };
        set_ids.get_mut(bucket).unwrap().push(id);
        if let Some(codes) = record["reason_codes"].as_array() {
            for code in codes
                .iter()
                .map(python_string)
                .filter(|value| !value.trim().is_empty())
            {
                let reasons = set_reasons.get_mut(bucket).unwrap();
                if !reasons.contains(&code) {
                    reasons.push(code);
                }
            }
        }
    }
    let summary_rows = set_labels
        .iter()
        .map(|(bucket, label)| {
            let ids = set_ids[*bucket].join(", ");
            let ids = if ids.is_empty() { "none" } else { &ids };
            let reasons = &set_reasons[*bucket];
            let why = if reasons.is_empty() {
                " No reason recorded.".into()
            } else {
                format!(" Reasons: {}.", reasons.join(", "))
            };
            format!("{} {label} ({ids}).{why}", set_ids[*bucket].len())
        })
        .collect();
    let mut blocks = vec![
        DiagnosticBlock::Heading("Recorded final correction decisions".into()),
        DiagnosticBlock::Subheading("Final acceptance sets:".into()),
        DiagnosticBlock::AcceptanceSets(summary_rows),
        DiagnosticBlock::Emphasis(binding_reason.into()),
    ];
    if disagree {
        blocks.push(DiagnosticBlock::Emphasis("Final records disagree on the delivered-graph identity; this explanation is unverified and nothing below is shown as delivered.".into()));
    }
    if rows.is_empty() {
        blocks.push(DiagnosticBlock::Paragraph("No final reconciled decisions recorded; provisional history below is not delivered correction.".into()));
    } else {
        blocks.push(DiagnosticBlock::DecisionTable(ReportTable {
            title: String::new(),
            note: String::new(),
            columns: ["Channel / output / seat", "Frequency", "Decision", "Why"]
                .map(str::to_owned)
                .to_vec(),
            rows,
        }));
    }
    (blocks, history)
}

fn correction_explanation_html(
    data: &Value,
    label: &str,
    binding_verified: bool,
    binding_reason: &str,
) -> String {
    section_fragment(correction_explanation_section(
        data,
        label,
        binding_verified,
        binding_reason,
    ))
    .expect("correction explanation is serialized by AutoEQ")
}

fn correction_explanation_section(
    data: &Value,
    label: &str,
    binding_verified: bool,
    binding_reason: &str,
) -> Value {
    let metadata = &data["metadata"];
    let acceptance = &metadata["correction_acceptance"];
    let quality = &acceptance["acoustic_quality"];
    let optimizer = &metadata["effective_config"]["optimizer"];
    let requested = if optimizer["correction_band"].is_null() {
        evidence_band(&json!([optimizer["min_freq"], optimizer["max_freq"]]))
    } else {
        evidence_band(&json!([
            optimizer["correction_band"]["min_hz"],
            optimizer["correction_band"]["max_hz"]
        ]))
    };
    let recorded = evidence_band(&quality["correction_band_hz"]);
    let mut observation = evidence_band(
        &optimizer["min_freq"]
            .as_f64()
            .zip(optimizer["max_freq"].as_f64())
            .map(|(a, b)| json!([a, b]))
            .unwrap_or(Value::Null),
    );
    observation = evidence_band(&quality["evaluated_band_hz"]).or(observation);
    let mut rows = Vec::new();
    let outcome = acceptance["outcome"].as_str();
    let mut outcome_message = match outcome {
        Some("accepted") if acceptance["accepted"] != true || acceptance["decision"] != "accepted" => "The acceptance record is incomplete or inconsistent; final correction is not verified.".to_string(),
        Some("accepted") => "The final correction passed the recorded acceptance checks. This is not a listening-benefit claim.".into(),
        Some("unchanged") => "The final outcome is unchanged; earlier correction attempts are not proof of delivered correction.".into(),
        Some("rejected") => "The final correction was rejected. Earlier applied stages do not establish an accepted result.".into(),
        Some("insufficient_evidence") => "The available evidence was insufficient to accept the final correction.".into(),
        _ => "No final correction decision was recorded.".into(),
    };
    if matches!(outcome, Some("accepted" | "unchanged"))
        && populated(&data["correction_decisions"])
        && !binding_verified
    {
        outcome_message = format!(
            "Recorded outcome: {}; not verified for this payload. {binding_reason}",
            outcome.unwrap_or_default()
        );
    }
    if acceptance["decision"] == "identity_fallback" {
        outcome_message = "The final outcome is unchanged: it fell back to identity; no applied correction records can override that outcome. The fallback withheld correction; candidate records below are not delivered correction.".into();
    }
    rows.push(vec![
        "System".into(),
        "All assessed frequencies".into(),
        "Final outcome".into(),
        outcome_message,
    ]);
    let correction = recorded.or(requested);
    if let Some(correction) = correction {
        rows.push(vec![
                "System".into(),
                evidence_band_label(correction),
                "Correction scope".into(),
                if recorded.is_some() {
                    "Recorded active-correction band.".into()
                } else {
                    "The configuration permits response shaping in this band; it does not prove that every frequency was corrected.".into()
                },
            ]);
        if let (Some(requested), Some(recorded)) = (requested, recorded)
            && requested != recorded
        {
            rows.push(vec!["System".into(), evidence_band_label(requested), "Requested scope".into(), "The requested correction band differs from the recorded active-correction band shown above.".into()]);
        }
        if let Some(observation) = observation {
            for outside in [
                (observation.0, correction.0.min(observation.1)),
                (correction.1.max(observation.0), observation.1),
            ] {
                if outside.0 > 0.0 && outside.0 < outside.1 {
                    rows.push(vec!["System".into(), evidence_band_label(outside), "Outside correction scope".into(), "Excluded from the active correction band. Routing, crossovers, gain and filter tails can still affect playback here.".into()]);
                }
            }
        }
    } else {
        rows.push(vec!["System".into(), "Frequency range unavailable".into(), "Scope not recorded".into(), "This output does not provide an explicit correction band. It is not inferred from the response curves.".into()]);
    }
    if let Some(seats) = quality["final_seats"].as_array() {
        for seat in seats {
            let scope = format!(
                "{} / {} / seat {}",
                seat["logical_input"].as_str().unwrap_or("Unknown input"),
                seat["partition"].as_str().unwrap_or("unknown partition"),
                seat.get("seat_index")
                    .map(python_string)
                    .unwrap_or_else(|| "?".into())
            );
            if let Some(bands) = seat["unassessed_bands_hz"].as_array() {
                for band in bands.iter().filter_map(evidence_band) {
                    rows.push(vec![scope.clone(), evidence_band_label(band), "Not assessed".into(), "Final playback was not assessed in this band. This does not establish that correction was absent or unnecessary.".into()]);
                }
            }
        }
    }
    let (ledger, ledger_history) = correction_ledger(data, binding_verified, binding_reason);
    if ledger.is_empty() {
        rows.push(vec!["System".into(), "Individual decisions unavailable".into(), "Reason unavailable".into(), "No versioned decision ledger was recorded with this output. Detailed reasons for individual peaks and dips are unavailable.".into()]);
    }
    let verification = verification_items(metadata);
    let output_loss = output_loss_table(quality);
    let veto_reasons = [
        (
            "SubJnd",
            "Below the configured level-difference proxy threshold.",
        ),
        (
            "SubErbWidth",
            "Below the configured affected-width proxy threshold.",
        ),
        (
            "HighQAboveGuard",
            "Exceeds the configured high-frequency Q guard.",
        ),
        (
            "ModeProximityBan",
            "Flagged by the recorded mode-proximity rule.",
        ),
        (
            "Audible",
            "Passed the configured veto rules; this is not proof of audibility.",
        ),
    ];
    if let Some(channels) = metadata["audibility_veto"].as_object() {
        for (channel, verdicts) in channels {
            if let Some(verdicts) = verdicts.as_array() {
                for verdict in verdicts {
                    let Some(frequency) = finite(&verdict["center_hz"]).filter(|v| *v > 0.0) else {
                        continue;
                    };
                    let decision = if verdict["decision"] == "Remove" {
                        "Removal nominated"
                    } else {
                        "Recorded veto decision"
                    };
                    let action = format!(
                        "{decision} ({})",
                        if verdict["enforced"] == true {
                            "stage history"
                        } else {
                            "advisory"
                        }
                    );
                    let reason = veto_reasons
                        .iter()
                        .find(|(code, _)| verdict["reason"] == *code)
                        .map(|(_, text)| *text)
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            format!(
                                "Recorded reason: {}.",
                                verdict["reason"].as_str().unwrap_or("unknown")
                            )
                        });
                    let assessment = &verdict["acceptance"];
                    let reason = format!(
                        "{reason} Confidence: {}. This filter-center record does not establish a frequency interval or the final exported filter set.",
                        assessment["confidence"].as_str().unwrap_or("unknown")
                    );
                    rows.push(vec![
                        channel.clone(),
                        format!("{} Hz (filter center)", format_general(frequency, 6)),
                        action,
                        reason,
                    ]);
                }
            }
        }
    }
    let mut details = ledger_history;
    if let Some(violations) = acceptance["violations"].as_array() {
        details.extend(
            violations
                .iter()
                .map(|value| format!("Final acceptance constraint: {}", python_string(value))),
        );
    }
    if let Some(stages) = acceptance["reverted_stages"].as_array() {
        details.extend(
            stages
                .iter()
                .map(|value| format!("Reverted stage: {}", python_string(value))),
        );
    }
    if let Some(stages) = metadata["stage_outcomes"].as_array() {
        for stage in stages {
            let name = stage["stage"].as_str().unwrap_or("unknown");
            let status = stage["status"].as_str().unwrap_or("unknown");
            if name == "measurement_input_conditioning" {
                details.extend(measurement_conditioning_history(stage));
            }
            if let Some(advisories) = stage["advisories"].as_array() {
                details.extend(advisories.iter().map(|advisory| {
                    format!("Stage {name} ({status}): {}", python_string(advisory))
                }));
            }
            let mut has_failure = false;
            if let Some(checks) = stage["checks"].as_array() {
                for check in checks {
                    if check["passed"] == false {
                        has_failure = true;
                        details.push(format!(
                            "Stage {name} ({status}), {}: {}",
                            check["id"].as_str().unwrap_or("unknown check"),
                            check["diagnostic"]
                                .as_str()
                                .filter(|value| !value.is_empty())
                                .unwrap_or("Check failed; no explanation recorded.")
                        ));
                    }
                }
            }
            if matches!(status, "skipped" | "degraded" | "failed")
                && stage["advisories"].as_array().is_none_or(Vec::is_empty)
                && !has_failure
            {
                details.push(format!("Stage {name}: {status}; no reason recorded."));
            }
        }
    }
    let mut blocks = vec![DiagnosticBlock::DecisionTable(ReportTable {
        title: String::new(),
        note: String::new(),
        columns: ["Channel / scope", "Frequency", "Decision", "Why"]
            .map(str::to_owned)
            .to_vec(),
        rows,
    })];
    blocks.extend(ledger);
    if !verification.is_empty() {
        blocks.push(DiagnosticBlock::Heading(
            "Recorded playback and listening verification".into(),
        ));
        blocks.push(DiagnosticBlock::EvidenceList(verification));
    }
    if let Some(table) = output_loss {
        blocks.push(DiagnosticBlock::DecisionTable(table));
    }
    if !details.is_empty() {
        blocks.push(DiagnosticBlock::StageHistory {
            title: "Recorded constraints and stage history".into(),
            note: "Stage history may describe attempts that were later reverted. The final outcome above takes precedence.".into(),
            items: details,
        });
    }
    blocks.push(DiagnosticBlock::Paragraph("Detailed reasons for individual peaks and dips are unavailable unless recorded above. An unchanged response alone does not identify a cancellation, poor measurement, or an already acceptable result.".into()));
    let mut section = diagnostic_section(blocks);
    section["introduction"] = json!(DiagnosticIntroduction {
        title: labelled_heading("Why this correction?", label),
        text: "Recorded reasons for correction scope, limitations and the final outcome.".into(),
        style: DiagnosticSectionStyle::CorrectionExplanation,
    });
    section
}

/// Builds four global evidence renderers from saved data and externally
/// verified binding evidence. Missing verification remains unavailable.
pub fn global_evidence_sections(data: &Value, label: &str, binding: &Value) -> Map<String, Value> {
    let mut sections = global_document_evidence(data, label, binding);
    for key in [
        "correction_explanation_html",
        "acceptance_views_html",
        "waveform_status_html",
        "capture_clock_qa_html",
        "capture_reflections_html",
        "modes_summary_html",
    ] {
        if let Some(section) = sections.get_mut(key) {
            *section = json!(
                section_fragment(section.take())
                    .expect("global evidence table is serialized by AutoEQ")
            );
        }
    }
    sections
}

pub(crate) fn global_document_evidence(
    data: &Value,
    label: &str,
    binding: &Value,
) -> Map<String, Value> {
    let verified = binding["verified"].as_bool().unwrap_or(false);
    let reason = binding["reason"]
        .as_str()
        .unwrap_or("Payload binding evidence unavailable.");
    let graph_identity = binding["graph_identity"].as_str().unwrap_or("");
    let mut sections = Map::new();
    sections.insert(
        "correction_explanation_html".into(),
        correction_explanation_section(data, label, verified, reason),
    );
    sections.insert(
        "acceptance_views_html".into(),
        acceptance_views_section(&json!({
            "data":data,
            "label":label,
            "binding_verified":verified,
            "binding_reason":reason,
            "binding_graph_identity":graph_identity,
            "acceptance_evidence_verified":binding["acceptance_evidence_verified"],
            "acceptance_evidence_reason":binding["acceptance_evidence_reason"],
        })),
    );
    sections.insert(
        "waveform_status_html".into(),
        table_section(waveform_status(data, label), TableStyle::Waveform),
    );
    sections.insert(
        "capture_clock_qa_html".into(),
        table_section(capture_clock_qa(data), TableStyle::CaptureClock),
    );
    sections.insert(
        "capture_reflections_html".into(),
        capture_reflections_section(data),
    );
    sections.insert(
        "modes_summary_html".into(),
        resonance_summary(data)
            .map(|block| diagnostic_section(vec![block]))
            .unwrap_or(Value::Null),
    );
    sections
}

fn group_delay(metadata: &Value) -> String {
    let gd = &metadata["group_delay"];
    if !populated(gd) {
        return EMPTY_CELL.repeat(5);
    }
    let advisory = gd
        .get("advisory")
        .map(python_string)
        .unwrap_or_else(|| "-".into());
    let advisory_color = if advisory == "success" { OK } else { "#b9770e" };
    let applied = gd["applied"].as_bool();
    let applied_str = match applied {
        Some(true) => "yes",
        Some(false) => "no",
        None => "-",
    };
    let applied_color = if applied == Some(true) { OK } else { "#999" };
    let improvement = number(&gd["improvement_db"]);
    let improvement_str = improvement
        .map(|value| format!("{value:+.2} dB"))
        .unwrap_or_else(|| "-".into());
    let improvement_color = if improvement.is_some_and(|value| value >= 0.0) {
        OK
    } else {
        BAD
    };
    // Python counts integer entries, including booleans, but excludes float entries.
    let ap_total: i128 = gd["per_channel_ap_count"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| {
            value
                .as_i64()
                .map(i128::from)
                .or_else(|| value.as_u64().map(i128::from))
                .or_else(|| value.as_bool().map(i128::from))
        })
        .sum();
    let coherence = number(&gd["mean_coherence"])
        .map(|value| format!(", coh={value:.2}"))
        .unwrap_or_default();
    format!(
        "<td style=\"color:{advisory_color};font-weight:600\">{}</td><td style=\"color:{applied_color};font-weight:600\">{applied_str}</td><td>{} → {}</td><td style=\"color:{improvement_color};font-weight:600\">{improvement_str}</td><td>{ap_total}{coherence}</td>",
        escape_html(&advisory),
        format_ms(gd.get("sum_gd_pre_rms_ms")),
        format_ms(gd.get("sum_gd_post_rms_ms"))
    )
}

fn ringing_pair(masking: &Value, peak: &str, audible: &str) -> String {
    match (number(&masking[peak]), number(&masking[audible])) {
        (Some(peak), Some(audible)) => format!("{peak:+.2} / {audible:+.2} dB"),
        (Some(peak), None) => format!("{peak:+.2} / - dB"),
        (None, Some(audible)) => format!("- / {audible:+.2} dB"),
        _ => "-".into(),
    }
}

fn mixed_phase(metadata: &Value, channels: &Value) -> String {
    mixed_phase_block(metadata, channels)
        .map(|block| diagnostic_html(&[block]))
        .unwrap_or_default()
}

pub(crate) fn mixed_phase_block(metadata: &Value, channels: &Value) -> Option<DiagnosticBlock> {
    let phase = &metadata["mixed_phase_per_channel"];
    let mut masking = std::collections::BTreeMap::new();
    if let Some(channels) = channels.as_object() {
        for (name, channel) in channels {
            if populated(&channel["fir_temporal_masking"]) {
                masking.insert(name.clone(), &channel["fir_temporal_masking"]);
            }
        }
    } else if let Some(channels) = channels.as_array() {
        for channel in channels {
            if populated(&channel["channel"]) && populated(&channel["fir_temporal_masking"]) {
                masking.insert(
                    python_string(&channel["channel"]),
                    &channel["fir_temporal_masking"],
                );
            }
        }
    }
    let perceptual = &metadata["perceptual_metrics"];
    let pre = number(&perceptual["fir_pre_ringing_audible_db"]);
    let post = number(&perceptual["fir_post_ringing_audible_db"]);
    let penalty = number(&perceptual["fir_temporal_masking_penalty"]);
    if !populated(phase)
        && masking.is_empty()
        && pre.is_none()
        && post.is_none()
        && penalty.is_none()
    {
        return None;
    }
    let mut names: Vec<_> = phase
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, _)| name.clone())
        .chain(masking.keys().cloned())
        .collect();
    names.sort_by(|left, right| channel_sort_key(left).cmp(&channel_sort_key(right)));
    names.dedup();
    let mut rows = Vec::new();
    for name in names {
        let report = &phase[&name];
        let empty = Value::Null;
        let mask = masking.get(&name).copied().unwrap_or(&empty);
        let taps = if report["fir_taps"].is_i64()
            || report["fir_taps"].is_u64()
            || report["fir_taps"].is_boolean()
        {
            python_string(&report["fir_taps"])
        } else {
            "-".into()
        };
        let range = match (
            number(&report["residual_excess_phase_min_deg"]),
            number(&report["residual_excess_phase_max_deg"]),
        ) {
            (Some(low), Some(high)) => format!("{low:+.2}° to {high:+.2}°"),
            _ => "-".into(),
        };
        let rms = number(&report["residual_excess_phase_rms_deg"])
            .map(|value| format!("{value:.2}°"))
            .unwrap_or_else(|| "-".into());
        let penalty = number(&mask["penalty"])
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".into());
        let delay = format_ms(report.get("estimated_delay_ms"));
        rows.push(vec![
            name,
            delay,
            taps,
            range,
            rms,
            format_ms(mask.get("main_time_ms")),
            ringing_pair(mask, "pre_ringing_peak_db", "pre_ringing_audible_db"),
            ringing_pair(mask, "post_ringing_peak_db", "post_ringing_audible_db"),
            penalty,
        ]);
    }
    let mut parts = Vec::new();
    if pre.is_some() || post.is_some() {
        parts.push(format!(
            "Worst audible pre/post ringing: {} / {}",
            pre.map(|value| format!("{value:+.2} dB"))
                .unwrap_or_else(|| "-".into()),
            post.map(|value| format!("{value:+.2} dB"))
                .unwrap_or_else(|| "-".into())
        ));
    }
    if let Some(penalty) = penalty {
        parts.push(format!("worst temporal masking penalty: {penalty:.3}"));
    }
    let note = if parts.is_empty() {
        String::new()
    } else {
        format!("{}.", parts.join("; "))
    };
    Some(DiagnosticBlock::MixedPhaseTable(ReportTable {
        title: "Mixed-Phase and FIR Timing".into(),
        note,
        columns: [
            "Channel",
            "Estimated delay",
            "FIR taps",
            "Residual phase range",
            "Residual RMS",
            "Main impulse",
            "Pre peak / audible",
            "Post peak / audible",
            "Penalty",
        ]
        .map(str::to_owned)
        .to_vec(),
        rows,
    }))
}

// Labels and precision are the existing EPA display contract, not scoring rules.
pub(crate) const EPA_FIELDS: &[(&str, &str, usize)] = &[
    ("preference", "Preference", 2),
    ("evaluation", "Evaluation", 2),
    ("potency", "Potency", 2),
    ("activity", "Activity", 2),
    ("sharpness_acum", "Sharpness (acum)", 2),
    ("roughness", "Roughness", 3),
    ("total_loudness_sone", "Total loudness (sone)", 2),
    ("loudness_balance", "Loudness balance", 3),
];

fn delta(field: &str, pre: &Value, post: &Value) -> EpaDelta {
    let (Some(pre), Some(post)) = (number(pre), number(post)) else {
        return EpaDelta::Unavailable;
    };
    let delta = post - pre;
    if delta.abs() < 1e-9 {
        return EpaDelta::Unchanged;
    }
    let higher_is_better = matches!(
        field,
        "preference" | "evaluation" | "total_loudness_sone" | "loudness_balance"
    );
    let display = format!("{delta:+.3}");
    if (delta > 0.0) == higher_is_better {
        EpaDelta::Improved(display)
    } else {
        EpaDelta::Worsened(display)
    }
}

pub(crate) fn epa_channel_block(epa: &Value) -> Option<DiagnosticBlock> {
    let (pre, post) = (&epa["pre"], &epa["post"]);
    if !populated(pre) && !populated(post) {
        return None;
    }
    let rows = EPA_FIELDS
        .iter()
        .map(|(field, label, precision)| {
            let formatted = |value: &Value| {
                number(value)
                    .map(|value| format!("{value:+.precision$}"))
                    .unwrap_or_else(|| "-".into())
            };
            EpaScoreRow {
                metric: (*label).into(),
                before: formatted(&pre[*field]),
                after: formatted(&post[*field]),
                delta: delta(field, &pre[*field], &post[*field]),
            }
        })
        .collect();
    Some(DiagnosticBlock::EpaScores(rows))
}

fn epa_summary(metadata: &Value) -> Value {
    let mean = |side: &str| {
        let values: Vec<_> = metadata["epa_per_channel"]
            .as_object()
            .into_iter()
            .flatten()
            .filter_map(|(_, entry)| number(&entry[side]["preference"]))
            .collect();
        if values.is_empty() {
            Value::Null
        } else {
            json!(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    json!([mean("pre"), mean("post")])
}

/// Produces a display section from saved evidence and verified binding status.
///
/// # Errors
/// Returns an error for an unsupported builder or missing playback binding data.
pub fn evidence_section(request: &Value) -> Result<Value, String> {
    Ok(match request["builder"].as_str().unwrap_or_default() {
        "playback_status" => {
            if !request["binding_verified"].is_boolean() || !request["binding_reason"].is_string() {
                return Err("playback status requires a verified binding result and reason".into());
            }
            json!(playback_status(request))
        }
        "waveform_status" => json!(waveform_status_html(
            &request["data"],
            request["label"].as_str().unwrap_or_default()
        )),
        "capture_clock_qa" => json!(capture_clock_qa_html(&request["data"])),
        "capture_reflections" => json!(capture_reflections_html(&request["data"])),
        "modes_summary" => json!(resonance_summary_html(&request["data"])),
        "capture_views" => json!(capture_views_html(
            &request["report"],
            &request["verification"]
        )),
        "acceptance_views" => json!(acceptance_views_html(request)),
        "correction_explanation" => json!(correction_explanation_html(
            &request["data"],
            request["label"].as_str().unwrap_or_default(),
            request["binding_verified"].as_bool().unwrap_or(false),
            request["binding_reason"]
                .as_str()
                .unwrap_or("Payload binding evidence unavailable.")
        )),
        "group_delay" => json!(group_delay(&request["metadata"])),
        "mixed_phase" => json!(mixed_phase(&request["metadata"], &request["channels"])),
        "epa_channel" => json!(
            epa_channel_block(&request["epa"])
                .map(|block| diagnostic_html(&[block]))
                .unwrap_or_default()
        ),
        "epa_summary" => epa_summary(&request["metadata"]),
        "delta" => json!(
            delta(
                request["field"].as_str().unwrap_or_default(),
                &request["pre"],
                &request["post"]
            )
            .browser_html()
        ),
        other => return Err(format!("unsupported evidence section builder `{other}`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_evidence_sections_match_pre_port_python() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/evidence_sections_python.json"
        ))
        .unwrap();
        for (index, case) in cases.as_array().unwrap().iter().enumerate() {
            assert_eq!(
                evidence_section(&case["request"]).unwrap(),
                case["expected"],
                "case {index}"
            );
        }
    }

    #[test]
    fn playback_does_not_infer_binding_from_acceptance_flags() {
        assert!(evidence_section(&json!({"builder":"playback_status","metadata":{"correction_acceptance":{"outcome":"accepted","accepted":true,"decision":"accepted"}}})).is_err());
        let html = evidence_section(&json!({"builder":"playback_status","metadata":{"correction_acceptance":{"outcome":"accepted","accepted":true,"decision":"accepted"}},"binding_verified":false,"binding_reason":"member hash mismatch"})).unwrap();
        assert!(
            html.as_str()
                .unwrap()
                .contains("Not approved for playback.")
        );
    }

    #[test]
    fn global_evidence_renderers_match_pre_port_python_snapshots() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/global_evidence_python.json"
        ))
        .unwrap();
        for (index, case) in cases.as_array().unwrap().iter().enumerate() {
            let request = case.get("request").cloned().unwrap_or_else(|| {
                json!({
                    "builder": case["builder"],
                    "data": case["data"],
                    "label": case.get("label").unwrap_or(&Value::Null),
                })
            });
            let actual = evidence_section(&request).unwrap();
            assert_eq!(actual, case["expected"], "case {index}");
        }
    }

    #[test]
    fn waveform_status_distinguishes_absent_stage_outcomes_from_null() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/waveform_missing_null_python.json"
        ))
        .unwrap();
        for case in ["missing", "null"] {
            assert_eq!(
                waveform_status_html(&fixture[case]["data"], ""),
                fixture[case]["expected"]
            );
        }
    }

    #[test]
    fn global_capture_sections_match_pre_port_python_snapshots() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/global_capture_sections_python.json"
        ))
        .unwrap();
        let capture_actual = capture_views_html(
            &fixture["capture_views"]["report"],
            &fixture["capture_views"]["verification"],
        );
        assert_embedded_html_equal(
            &capture_actual,
            fixture["capture_views"]["expected"].as_str().unwrap(),
        );
        assert_eq!(
            resonance_summary_html(&fixture["resonance_summary"]["data"]),
            fixture["resonance_summary"]["expected"]
        );
        let reflection_actual = capture_reflections_html(&fixture["capture_reflections"]["data"]);
        assert_embedded_html_equal(
            &reflection_actual,
            fixture["capture_reflections"]["expected"].as_str().unwrap(),
        );
    }

    #[test]
    fn complete_nested_capture_views_match_python_snapshot() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/capture_views_nested_python.json"
        ))
        .unwrap();
        let (section, available) =
            capture_views_section(&fixture["report"], &fixture["verification"]);
        assert!(available);
        let blocks: Vec<DiagnosticBlock> =
            serde_json::from_value(section["blocks"].clone()).unwrap();
        let native_figures: Vec<_> = blocks
            .into_iter()
            .filter_map(|block| match block {
                DiagnosticBlock::CaptureChart(chart) => {
                    serde_json::from_value::<crate::schema::Section>(chart.clone())
                        .expect("capture chart is readable by the shared native renderer");
                    Some(chart)
                }
                _ => None,
            })
            .collect();
        let actual = section_fragment(section).unwrap();
        let (actual_text, actual_figures) = split_embedded_figures(&actual);
        let (expected_text, expected_figures) =
            split_embedded_figures(fixture["expected"].as_str().unwrap());
        assert_eq!(native_figures, expected_figures);
        assert_eq!(actual_text, expected_text);
        assert_eq!(actual_figures.len(), expected_figures.len());
        for (index, (actual, expected)) in actual_figures.iter().zip(&expected_figures).enumerate()
        {
            assert_eq!(actual, expected, "embedded figure {index}");
        }
    }

    #[test]
    fn capture_views_require_native_comparison_binding() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/global_capture_sections_python.json"
        ))
        .unwrap();
        let mut verification = fixture["capture_views"]["verification"].clone();
        verification["comparisons"][0]["verified"] = json!(false);
        let html = capture_views_html(&fixture["capture_views"]["report"], &verification);
        assert!(html.contains("capture view payload or graph/source/seat binding changed"));
        assert!(!html.contains("Matched IR evidence"));
    }

    #[test]
    fn coherent_clock_gate_resolves_saved_sources_and_fails_closed() {
        let data = json!({
            "system":{"speakers":{"L":"left_source","R":"right_source"}},
            "speakers":{"left_source":{},"right_source":{}}
        });
        assert_eq!(
            coherent_capture_clock_reason(&data, ["L", "R"]).as_deref(),
            Some("left_source: per-microphone clock bounds are unavailable")
        );
        assert_eq!(
            coherent_capture_clock_reason(&data, ["L", "L"]).as_deref(),
            Some("two distinct output channels are required")
        );
        let missing_source = json!({"speakers":{}});
        assert_eq!(
            coherent_capture_clock_reason(&missing_source, ["L", "R"]).as_deref(),
            Some("capture provenance is unavailable for L")
        );
    }

    fn assert_embedded_html_equal(actual: &str, expected: &str) {
        let (actual_text, actual_figures) = split_embedded_figures(actual);
        let (expected_text, expected_figures) = split_embedded_figures(expected);
        assert_eq!(actual_text, expected_text);
        assert_eq!(actual_figures, expected_figures);
    }

    fn split_embedded_figures(html: &str) -> (String, Vec<Value>) {
        let marker = "<div class=\"report-figure\" data-section=\"";
        let mut remaining = html;
        let mut plain = String::new();
        let mut figures = Vec::new();
        while let Some(start) = remaining.find(marker) {
            plain.push_str(&remaining[..start]);
            let rest = &remaining[start + marker.len()..];
            let Some(end) = rest.find("\"></div>") else {
                panic!("unterminated embedded figure");
            };
            let encoded = &rest[..end];
            let decoded = encoded
                .replace("&quot;", "\"")
                .replace("&#x27;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&");
            figures.push(serde_json::from_str(&decoded).expect("embedded figure JSON"));
            plain.push_str("<figure></figure>");
            remaining = &rest[end + 8..];
        }
        plain.push_str(remaining);
        (plain, figures)
    }
}
