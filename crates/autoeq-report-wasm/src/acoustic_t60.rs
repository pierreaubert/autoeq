//! Validates and renders measured octave-band reverberation-time report data.
//!
//! This module only consumes serialized measurement results. T60 estimation is
//! performed by the upstream RoomEQ analyzer; this module validates its emitted
//! measured-room basis and formats the resulting report fields.

// Rust guideline compliant 2026-02-21

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    report_document::{DiagnosticBlock, MetricCell, MetricTone, diagnostic_html},
    report_table::ReportTable,
};

const CENTERS_HZ: [u32; 9] = [63, 125, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000];
const DEFAULT_TOLERANCE_S: f64 = 0.05;

/// One validated octave-band fit emitted by the measured-room analyzer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct T60BandRow {
    /// Octave-band center in hertz.
    pub centre_hz: u32,
    /// Measured decay time, unavailable when the analyzer rejected this band.
    pub t60_s: Option<f64>,
    /// Fit range used by the analyzer, when the band is valid.
    pub fit_range: Option<String>,
    /// Coefficient of determination emitted by the analyzer.
    pub r2: f64,
    /// Analyzer-provided reason for an invalid band, or an empty string.
    pub reason: String,
}

/// Validate one channel's serialized `t60_octaves` measured-room result.
///
/// The basis, exact nine-band topology, fit type, minimum R², and invalid-band
/// reason are checked before any values are used in summaries.
pub fn t60_rows(channel_data: &Value) -> Option<Vec<T60BandRow>> {
    let report = channel_data.get("t60_octaves")?.as_object()?;
    if report.get("basis")?.as_str()? != "measured_room_ir" {
        return None;
    }
    let min_r2 = finite_number(report.get("min_r2")?)?;
    if !(0.0 < min_r2 && min_r2 <= 1.0) {
        return None;
    }
    let bands = report.get("bands")?.as_array()?;
    if bands.len() != CENTERS_HZ.len() {
        return None;
    }

    let mut validated = Vec::with_capacity(CENTERS_HZ.len());
    for (center, band) in CENTERS_HZ.iter().zip(bands) {
        let band = band.as_object()?;
        if finite_number(band.get("centre_hz")?)? != f64::from(*center) {
            return None;
        }
        let valid = band.get("valid")?.as_bool()?;
        let r2 = finite_number(band.get("r2")?)?;
        if !(0.0..=1.0).contains(&r2) {
            return None;
        }
        if valid {
            let fit_range = band.get("fit_range")?.as_str()?;
            if fit_range != "T20" && fit_range != "T30" {
                return None;
            }
            let t60_s = finite_number(band.get("t60_s")?)?;
            if t60_s <= 0.0 || r2 < min_r2 {
                return None;
            }
            validated.push(T60BandRow {
                centre_hz: *center,
                t60_s: Some(t60_s),
                fit_range: Some(fit_range.to_owned()),
                r2,
                reason: String::new(),
            });
        } else {
            let reason = band.get("reason")?.as_str()?;
            if reason.trim().is_empty() {
                return None;
            }
            validated.push(T60BandRow {
                centre_hz: *center,
                t60_s: None,
                fit_range: None,
                r2,
                reason: reason.to_owned(),
            });
        }
    }
    Some(validated)
}

/// Build all T60 report values from serialized acoustic-report input.
///
/// The returned object contains reference, room rows and HTML, plus per-channel
/// rows, flatness/ITU summaries, and their presentation cells.
pub fn t60_report(data: &Value) -> Value {
    let tolerance = tolerance(data);
    let channels = data["channels"].as_object();
    let flatness_mean = complete_channel_mean(channels);
    let flatness_window =
        flatness_mean.map(|mean| [0.0_f64.max(mean - tolerance.0), mean + tolerance.0]);
    let reference = itu_reference(data);
    let room_rows = room_t60_rows(data);
    let mut channel_reports = Map::new();
    if let Some(channels) = channels {
        for (name, channel) in channels {
            let rows = t60_rows(channel);
            let flatness = flatness_pct(rows.as_deref(), flatness_mean, tolerance.0);
            let itu = itu_pct(rows.as_deref(), reference.as_ref());
            channel_reports.insert(
                name.clone(),
                json!({
                    "rows": rows,
                    "flatness_pct": flatness,
                    "flatness_window_s": flatness_window,
                    "itu_pct": itu,
                    "t60_table_html": diagnostic_html(&[t60_band_table(rows.as_deref())]),
                    "flatness_cell_html": flatness_metric(flatness, tolerance).browser_html(),
                    "itu_cell_html": itu_metric(itu).browser_html(),
                }),
            );
        }
    }
    let room_table_html = room_t60_table(data, room_rows.as_deref())
        .map(|table| diagnostic_html(&[table]))
        .unwrap_or_default();
    json!({
        "tolerance": {"seconds": tolerance.0, "declared": tolerance.1},
        "reference": reference,
        "room_rows": room_rows,
        "room_t60_table_html": room_table_html,
        "itu_note_html": t60_itu_note(reference.as_ref().unwrap_or(&Value::Null)),
        "channels": channel_reports,
    })
}

fn complete_channel_mean(channels: Option<&Map<String, Value>>) -> Option<f64> {
    let complete = channels
        .map(|channels| {
            channels
                .iter()
                .filter_map(|(name, channel)| {
                    let rows = t60_rows(channel)?;
                    rows.iter()
                        .all(|row| row.t60_s.is_some())
                        .then_some((name, rows))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let all_complete_values: Vec<f64> = complete
        .iter()
        .flat_map(|(_, rows)| rows.iter().filter_map(|row| row.t60_s))
        .collect();
    stable_mean(&all_complete_values)
}

fn flatness_pct(rows: Option<&[T60BandRow]>, mean: Option<f64>, tolerance_s: f64) -> Option<f64> {
    let rows = rows?;
    let mean = mean?;
    if rows.iter().any(|row| row.t60_s.is_none()) {
        return None;
    }
    let inside = rows
        .iter()
        .filter(|row| {
            row.t60_s
                .is_some_and(|value| (value - mean).abs() <= tolerance_s)
        })
        .count();
    Some(100.0 * inside as f64 / rows.len() as f64)
}

pub(crate) fn summary_metrics(
    data: &Value,
    channels: &Map<String, Value>,
) -> std::collections::BTreeMap<String, [MetricCell; 2]> {
    let tolerance = tolerance(data);
    let mean = complete_channel_mean(Some(channels));
    let reference = itu_reference(data);
    channels
        .iter()
        .map(|(name, channel)| {
            let rows = t60_rows(channel);
            let flatness = flatness_pct(rows.as_deref(), mean, tolerance.0);
            let itu = itu_pct(rows.as_deref(), reference.as_ref());
            (
                name.clone(),
                [flatness_metric(flatness, tolerance), itu_metric(itu)],
            )
        })
        .collect()
}

pub(crate) fn t60_document_data(data: &Value) -> (Value, Option<DiagnosticBlock>) {
    let rows = room_t60_rows(data);
    let table = room_t60_table(data, rows.as_deref());
    (
        json!({"room_rows": rows, "reference": itu_reference(data)}),
        table,
    )
}

/// Render the report's explanatory BS.1116 note for a reference or JSON null.
pub fn t60_itu_note(reference: &Value) -> String {
    diagnostic_html(&[t60_reference(reference)])
}

fn finite_number(value: &Value) -> Option<f64> {
    let number = value.as_f64()?;
    number.is_finite().then_some(number)
}

fn tolerance(data: &Value) -> (f64, bool) {
    let value = data
        .get("metadata")
        .and_then(|metadata| metadata.get("t60_flatness_tolerance_s"))
        .and_then(finite_number);
    match value {
        Some(value) if value > 0.0 => (value, true),
        _ => (DEFAULT_TOLERANCE_S, false),
    }
}

fn stable_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    // Neumaier summation mirrors Python's math.fsum behavior for the scaled
    // terms while avoiding an overflowing unscaled intermediate sum.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let scaled = value / values.len() as f64;
        let next = sum + scaled;
        if sum.abs() >= scaled.abs() {
            correction += (sum - next) + scaled;
        } else {
            correction += (scaled - next) + sum;
        }
        sum = next;
    }
    let scaled_sum = sum + correction;
    scaled_sum.is_finite().then_some(scaled_sum)
}

pub(crate) fn itu_reference(data: &Value) -> Option<Value> {
    let dimensions = data
        .get("metadata")
        .and_then(|metadata| metadata.get("effective_config"))
        .and_then(|config| config.get("recording_config"))
        .and_then(|recording| recording.get("room_dimensions"))
        .and_then(Value::as_object);
    if let Some(dimensions) = dimensions {
        let values = ["length", "width", "height"]
            .iter()
            .map(|key| dimensions.get(*key).and_then(finite_number))
            .collect::<Option<Vec<_>>>();
        if let Some(values) = values
            && values.iter().all(|value| *value > 0.0)
        {
            let volume = values.iter().product::<f64>();
            if volume.is_finite() && volume > 0.0 {
                let tm = 0.25 * (volume / 100.0).powf(1.0 / 3.0);
                if tm.is_finite() {
                    return Some(json!({
                        "tm": tm,
                        "volume_m3": volume,
                        "source": format!("volume recommendation ({} m³)", general_number(volume)),
                    }));
                }
            }
        }
    }

    let mut channel_means = Vec::new();
    if let Some(channels) = data.get("channels").and_then(Value::as_object) {
        for channel in channels.values() {
            let Some(rows) = t60_rows(channel) else {
                continue;
            };
            let midband: Vec<f64> = rows
                .iter()
                .filter(|row| (200..=4_000).contains(&row.centre_hz))
                .filter_map(|row| row.t60_s)
                .collect();
            if midband.len() == 5
                && let Some(mean) = stable_mean(&midband)
            {
                channel_means.push(mean);
            }
        }
    }
    let tm = stable_mean(&channel_means)?;
    Some(json!({"tm": tm, "source": "measured midband mean; room volume unavailable"}))
}

pub(crate) fn t60_reference(reference: &Value) -> DiagnosticBlock {
    let source = if reference.is_null() {
        "limits unavailable: no room volume or complete midband fits".to_owned()
    } else {
        let tm = reference.get("tm").and_then(finite_number).unwrap_or(0.0);
        let description = reference
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or("");
        format!("Tm = {tm:.3} s — {description}")
    };
    DiagnosticBlock::Reference {
        text: format!(
            "ITU-R BS.1116-3 §8.2.3.1, Figure 1: {source}. Upper limit covers 63 Hz–8 kHz; lower limit starts at 100 Hz. No extrapolated limits outside that range. A measured-mean reference checks relative decay shape only, not the volume-based recommendation. The summary’s fixed-window flatness score is a separate metric."
        ),
        label: "ITU recommendation".into(),
        url: "https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1116-3-201502-I!!PDF-E.pdf"
            .into(),
    }
}

fn room_t60_rows(data: &Value) -> Option<Vec<RoomRow>> {
    let channels = data.get("channels")?.as_object()?;
    let sources: Vec<Vec<T60BandRow>> = channels.values().filter_map(t60_rows).collect();
    if sources.is_empty() {
        return None;
    }
    Some(
        CENTERS_HZ
            .iter()
            .enumerate()
            .map(|(index, center)| {
                let values: Vec<f64> = sources
                    .iter()
                    .filter_map(|rows| rows[index].t60_s)
                    .collect();
                RoomRow {
                    centre_hz: *center,
                    t60_s: if values.is_empty() {
                        None
                    } else {
                        Some(values.iter().sum::<f64>() / values.len() as f64)
                    },
                    speaker_count: values.len(),
                }
            })
            .collect(),
    )
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct RoomRow {
    centre_hz: u32,
    t60_s: Option<f64>,
    speaker_count: usize,
}

fn room_t60_table(data: &Value, rows: Option<&[RoomRow]>) -> Option<DiagnosticBlock> {
    let rows = rows.filter(|rows| !rows.is_empty())?;
    let channels = data
        .get("channels")
        .and_then(Value::as_object)
        .map(|channels| {
            channels
                .iter()
                .filter_map(|(name, channel)| t60_rows(channel).map(|rows| (name, rows)))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut columns = ["Centre (Hz)", "Room mean (s)", "Valid speakers"]
        .map(str::to_owned)
        .to_vec();
    columns.extend(channels.iter().map(|(name, _)| format!("{name} T60 (s)")));
    let rows = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let mut cells = vec![
                row.centre_hz.to_string(),
                format_t60(row.t60_s),
                row.speaker_count.to_string(),
            ];
            cells.extend(
                channels
                    .iter()
                    .map(|(_, bands)| format_t60(bands[index].t60_s)),
            );
            cells
        })
        .collect();
    Some(DiagnosticBlock::EvidenceTable(ReportTable {
        title: "Octave-band T60 — room and speakers".into(),
        note: String::new(),
        columns,
        rows,
    }))
}

pub(crate) fn t60_band_table(rows: Option<&[T60BandRow]>) -> DiagnosticBlock {
    let Some(rows) = rows else {
        return DiagnosticBlock::Caption(
            "Octave T60: pending roomeq field t60_octaves from a measured room impulse response."
                .into(),
        );
    };
    DiagnosticBlock::EvidenceTable(ReportTable {
        title: "Octave-band T60".into(),
        note: "Measured room IR; T30 preferred, T20 fallback. Invalid bands are excluded from the curve.".into(),
        columns: ["Centre (Hz)", "T60 (s)", "Fit", "R²", "Status"].map(str::to_owned).to_vec(),
        rows: rows.iter().map(|row| {
            vec![
                row.centre_hz.to_string(),
                format_t60(row.t60_s),
                row.fit_range.as_deref().unwrap_or("n/a").to_owned(),
                format!("{:.3}", row.r2),
                if row.reason.is_empty() { "valid".into() } else { row.reason.clone() },
            ]
        }).collect(),
    })
}

fn format_t60(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.3}"))
}

fn itu_pct(rows: Option<&[T60BandRow]>, reference: Option<&Value>) -> Option<f64> {
    let reference = reference?;
    reference.get("volume_m3")?;
    let rows = rows?;
    let bands: Vec<&T60BandRow> = rows
        .iter()
        .filter(|row| (63..=8_000).contains(&row.centre_hz))
        .collect();
    if bands.len() != 8 || bands.iter().any(|row| row.t60_s.is_none()) {
        return None;
    }
    let tm = finite_number(reference.get("tm")?)?;
    let mut inside = 0;
    for row in bands {
        let hz = f64::from(row.centre_hz);
        let value = row.t60_s?;
        let upper = tm
            + if hz < 200.0 {
                0.30 - 0.25 * (hz / 63.0).ln() / (200.0_f64 / 63.0).ln()
            } else if hz <= 4_000.0 {
                0.05
            } else {
                0.10
            };
        let lower = if hz < 100.0 {
            None
        } else if hz <= 4_000.0 {
            Some(tm - 0.05)
        } else {
            Some(tm - 0.10)
        };
        inside +=
            usize::from(value <= upper + 1e-12 && lower.is_none_or(|bound| value >= bound - 1e-12));
    }
    Some(100.0 * inside as f64 / 8.0)
}

fn t60_metric_tone(metric: f64) -> MetricTone {
    if metric > 80.0 {
        MetricTone::Good
    } else if metric >= 50.0 {
        MetricTone::Warning
    } else {
        MetricTone::Poor
    }
}

fn itu_metric(metric: Option<f64>) -> MetricCell {
    let Some(metric) = metric else {
        return MetricCell {
            value: "Not assessed".into(),
            tone: MetricTone::Unassessed,
            detail: Some("Requires valid recording_config.room_dimensions and all eight measured octave T60 fits from 63 Hz to 8 kHz".into()),
        };
    };
    let count = (metric * 8.0 / 100.0).round();
    MetricCell {
        value: format!("{metric:.1}"),
        tone: t60_metric_tone(metric),
        detail: Some(format!(
            "{count:.0}/8 octave centers within ITU-R BS.1116-3 Figure 1 limits around the volume recommendation; 63 Hz has only an upper bound; 4 kHz uses the midband limits; 16 kHz is excluded. Not overall room compliance."
        )),
    }
}

fn flatness_metric(metric: Option<f64>, tolerance: (f64, bool)) -> MetricCell {
    let Some(metric) = metric else {
        return MetricCell::pending("nine valid measured t60_octaves fits");
    };
    let source = if tolerance.1 {
        format!("declared ±{} s", general_number(tolerance.0))
    } else {
        format!(
            "default ±{} s (ITU-R BS.1116-2 §8.2.3.1 Fig. 1 midband; declare reporting.t60_flatness_tolerance_s to override)",
            general_number(tolerance.0)
        )
    };
    MetricCell {
        value: format!("{metric:.1}"),
        tone: t60_metric_tone(metric),
        detail: Some(format!(
            "nine measured octave fits within {source} of the complete-channel room mean"
        )),
    }
}

fn general_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        let mut result = format!("{value}");
        if result.contains('.') {
            while result.ends_with('0') {
                result.pop();
            }
            if result.ends_with('.') {
                result.pop();
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn golden() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../scripts/tests/fixtures/acoustic_t60_python_goldens.json"
        );
        serde_json::from_str(&fs::read_to_string(path).expect("golden fixture"))
            .expect("valid golden JSON")
    }

    #[test]
    fn measured_basis_validation_and_tables_match_python_goldens() {
        let golden = golden();
        let inputs = &golden["inputs"];
        let expected = &golden["expected"];
        let valid = t60_rows(&inputs["valid"]["channels"]["L"]).expect("valid measured result");
        assert_eq!(
            serde_json::to_value(valid).unwrap(),
            expected["valid_l_rows"]
        );
        let invalid =
            t60_rows(&inputs["invalid"]["channels"]["L"]).expect("invalid band remains visible");
        assert_eq!(
            serde_json::to_value(invalid).unwrap(),
            expected["invalid_l_rows"]
        );
        assert!(t60_rows(&inputs["unsupported"]["channels"]["L"]).is_none());

        let report = t60_report(&inputs["invalid"]);
        assert_eq!(
            report["channels"]["L"]["t60_table_html"],
            expected["t60_table_html_invalid_l"]
        );
        let unsupported = t60_report(&inputs["unsupported"]);
        assert_eq!(
            unsupported["channels"]["L"]["t60_table_html"],
            expected["t60_table_html_pending"]
        );
        assert_eq!(
            report["room_t60_table_html"],
            expected["room_t60_table_html"]
        );
        assert_eq!(report["room_rows"], expected["room_rows"]);
        let (document, room_table) = t60_document_data(&inputs["invalid"]);
        assert_eq!(document["room_rows"], expected["room_rows"]);
        let Some(DiagnosticBlock::EvidenceTable(room_table)) = room_table else {
            panic!("validated room data should produce a native evidence table");
        };
        assert_eq!(room_table.rows.len(), CENTERS_HZ.len());
        assert_eq!(
            room_table.columns[..3],
            ["Centre (Hz)", "Room mean (s)", "Valid speakers"]
        );
        assert_eq!(
            diagnostic_html(&[DiagnosticBlock::EvidenceTable(room_table)]),
            expected["room_t60_table_html"].as_str().unwrap()
        );
    }

    #[test]
    fn measured_references_and_scores_match_python_goldens() {
        let golden = golden();
        let inputs = &golden["inputs"];
        let expected = &golden["expected"];
        let report = t60_report(&inputs["valid"]);
        assert_eq!(
            report["tolerance"],
            json!({"seconds": 0.05, "declared": true})
        );
        assert_eq!(
            report["reference"]["tm"].as_f64(),
            expected["volume_reference"]["tm"].as_f64()
        );
        assert_eq!(
            report["reference"]["volume_m3"].as_f64(),
            expected["volume_reference"]["volume_m3"].as_f64()
        );
        assert_eq!(
            report["reference"]["source"],
            expected["volume_reference"]["source"]
        );
        assert_eq!(
            report["channels"]["L"]["flatness_window_s"],
            expected["flatness_window"]
        );
        assert_eq!(
            report["channels"]["L"]["flatness_pct"],
            expected["flatness_pct_l"]
        );
        assert_eq!(
            report["channels"]["L"]["itu_pct"],
            expected["itu_pct_volume_l"]
        );
        assert_eq!(
            report["channels"]["L"]["flatness_cell_html"],
            expected["flatness_cell_html"]
        );
        assert_eq!(
            report["channels"]["L"]["itu_cell_html"],
            expected["itu_cell_html"]
        );
        assert_eq!(report["itu_note_html"], expected["itu_note_html"]);
        let measured = t60_report(&inputs["measured_reference"]);
        assert_eq!(
            measured["reference"]["tm"].as_f64(),
            expected["measured_reference"]["tm"].as_f64()
        );
        assert_eq!(
            measured["reference"]["source"],
            expected["measured_reference"]["source"]
        );
        assert_eq!(
            measured["channels"]["L"]["itu_pct"],
            expected["itu_pct_missing_volume_l"]
        );
        let unsupported = t60_report(&inputs["unsupported"]);
        assert_eq!(unsupported["channels"]["L"]["flatness_pct"], Value::Null);
        assert_eq!(
            unsupported["channels"]["L"]["flatness_cell_html"],
            expected["flatness_cell_pending_html"]
        );
        assert_eq!(
            unsupported["channels"]["L"]["itu_cell_html"],
            expected["itu_cell_pending_html"]
        );
        assert_eq!(
            unsupported["itu_note_html"],
            expected["itu_note_unavailable_html"]
        );
        let invalid = t60_report(&inputs["invalid"]);
        assert_eq!(invalid["channels"]["L"]["itu_pct"], Value::Null);
        assert_eq!(invalid["channels"]["L"]["flatness_pct"], Value::Null);
        assert_eq!(
            t60_itu_note(&expected["volume_reference"]),
            expected["itu_note_html"].as_str().unwrap()
        );
    }

    #[test]
    fn missing_and_malformed_evidence_stays_unavailable() {
        let mut input = json!({"channels":{"L":{"t60_octaves":null}},"metadata":{}});
        assert_eq!(tolerance(&input), (DEFAULT_TOLERANCE_S, false));
        assert_eq!(t60_report(&input)["room_rows"], Value::Null);
        assert_eq!(
            t60_itu_note(&Value::Null),
            golden()["expected"]["itu_note_unavailable_html"]
        );
        input["metadata"]["t60_flatness_tolerance_s"] = json!(true);
        assert_eq!(tolerance(&input), (DEFAULT_TOLERANCE_S, false));
    }
}
