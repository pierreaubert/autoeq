//! Summarizes measured and predicted response curves without reading artifacts.
//!
//! These display calculations preserve the standalone report's frequency-grid
//! validation, triangular octave smoothing, landmark prominence, and downstream
//! trim convention. Predicted curves are never presented as verification captures.

use crate::{
    report_document::{DiagnosticBlock, MetricCell, MetricTone, diagnostic_html},
    report_table::ReportTable,
};
use serde_json::{Map, Value, json};

fn points(curve: &Value) -> Option<Vec<(f64, f64)>> {
    let frequencies = curve.get("freq")?.as_array()?;
    let levels = curve.get("spl")?.as_array()?;
    if frequencies.is_empty() || frequencies.len() != levels.len() {
        return None;
    }
    let mut previous = 0.0;
    let mut result = Vec::with_capacity(frequencies.len());
    for (frequency, level) in frequencies.iter().zip(levels) {
        let frequency = frequency.as_f64()?;
        let level = level.as_f64()?;
        if !frequency.is_finite() || frequency <= previous || !level.is_finite() {
            return None;
        }
        previous = frequency;
        result.push((frequency, level));
    }
    Some(result)
}

// Maintain nonoverlapping partials to retain cancellation-sensitive band means,
// matching the Python report's math.fsum convention rather than naive summation.
fn precise_sum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut partials: Vec<f64> = Vec::new();
    for mut value in values {
        let mut next = Vec::new();
        for mut partial in partials {
            if value.abs() < partial.abs() {
                std::mem::swap(&mut value, &mut partial);
            }
            let high = value + partial;
            if !high.is_finite() {
                return None;
            }
            let low = partial - (high - value);
            if low != 0.0 {
                next.push(low);
            }
            value = high;
        }
        next.push(value);
        partials = next;
    }
    let sum: f64 = partials.into_iter().rev().sum();
    sum.is_finite().then_some(sum)
}

/// Returns the mean level within an inclusive validated frequency band.
pub fn band_mean(curve: &Value, low_hz: f64, high_hz: f64) -> Option<f64> {
    let values: Vec<f64> = points(curve)?
        .into_iter()
        .filter(|(f, _)| low_hz <= *f && *f <= high_hz)
        .map(|(_, s)| s)
        .collect();
    if values.is_empty() {
        return None;
    }
    let mean = precise_sum(values.iter().copied())? / values.len() as f64;
    mean.is_finite().then_some(mean)
}

/// Returns the lowest predicted bass level relative to its monitor-band mean.
pub fn deepest_notch_db(channel: &Value) -> Option<f64> {
    let curve = channel.get("final_curve")?;
    let samples = points(curve)?;
    let reference = band_mean(curve, 500.0, 3000.0)?;
    let lowest = samples
        .into_iter()
        .filter(|(f, _)| *f < 300.0)
        .map(|(_, s)| s)
        .min_by(f64::total_cmp)?;
    let difference = lowest - reference;
    difference.is_finite().then_some(difference)
}

/// Smooths a validated response with the report's triangular octave window.
pub fn smooth_curve(curve: &Value, octaves: f64) -> Option<Value> {
    let samples = points(curve)?;
    if !octaves.is_finite() {
        return None;
    }
    let ratio = 2.0_f64.powf(octaves / 2.0);
    let half_width = ratio.log10();
    if ratio <= 1.0 {
        return Some(
            json!({"freq":samples.iter().map(|(f,_)|*f).collect::<Vec<_>>(),
            "spl":samples.iter().map(|(_,s)|*s).collect::<Vec<_>>() }),
        );
    }
    let logs: Vec<_> = samples.iter().map(|(f, _)| f.log10()).collect();
    // A triangular log-frequency weight is affine on either side of its
    // center. Four prefix sums evaluate both halves without an N² scan.
    let mut prefixes = vec![[0.0; 3]];
    let mut sums = [0.0; 3];
    let mut corrections = [0.0; 3];
    for (&log, &(_, level)) in logs.iter().zip(&samples) {
        for (component, term) in [log, level, log * level].into_iter().enumerate() {
            let adjusted = term - corrections[component];
            let next = sums[component] + adjusted;
            corrections[component] = (next - sums[component]) - adjusted;
            sums[component] = next;
        }
        prefixes.push(sums);
    }
    // Keep the exact legacy accumulation for short report/golden curves;
    // prefix subtraction is useful only once dense captures dominate work.
    let prefixes_finite = samples.len() > 512
        && prefixes.iter().flatten().all(|v| v.is_finite())
        && samples.iter().all(|(_, level)| level.abs() <= 1e6)
        && logs.iter().all(|log| log.abs() <= 12.0);
    let direct = |center: usize, first: usize, last: usize| {
        let mut weighted = 0.0;
        let mut weights = 0.0;
        for index in first..last {
            let weight = (1.0 - (logs[index] - logs[center]).abs() / half_width).max(0.0);
            weighted += samples[index].1 * weight;
            weights += weight;
        }
        if weights > 0.0 {
            weighted / weights
        } else {
            samples[center].1
        }
    };
    let mut levels = Vec::with_capacity(samples.len());
    for (index, &(center, original)) in samples.iter().enumerate() {
        let first = samples.partition_point(|(f, _)| *f < center / ratio);
        let last = samples.partition_point(|(f, _)| *f <= center * ratio);
        let level = if last - first <= 1 {
            original
        } else if !half_width.is_finite() {
            samples.iter().map(|(_, level)| level).sum::<f64>() / samples.len() as f64
        } else if prefixes_finite && half_width >= 1e-6 {
            let range = |a: usize, b: usize| {
                let count = (b - a) as f64;
                let log_sum = prefixes[b][0] - prefixes[a][0];
                let level_sum = prefixes[b][1] - prefixes[a][1];
                let moment = prefixes[b][2] - prefixes[a][2];
                (count, log_sum, level_sum, moment)
            };
            let (ln, lx, ly, lxy) = range(first, index);
            let (rn, rx, ry, rxy) = range(index, last);
            let logarithm = logs[index];
            let weight = ln + rn + (lx - logarithm * ln + logarithm * rn - rx) / half_width;
            let weighted = ly + ry + (lxy - logarithm * ly + logarithm * ry - rxy) / half_width;
            if weight > 0.0 && weighted.is_finite() {
                weighted / weight
            } else {
                direct(index, first, last)
            }
        } else {
            direct(index, first, last)
        };
        if !level.is_finite() {
            return None;
        }
        levels.push(level);
    }
    Some(json!({"freq":samples.iter().map(|(f,_)|*f).collect::<Vec<_>>(),"spl":levels}))
}

/// Finds prominent landmarks in the measured one-third-octave response.
pub fn response_landmarks(channel: &Value) -> Option<Value> {
    let curve = channel.get("initial_curve")?;
    let samples = points(curve)?;
    if samples.len() < 5 {
        return None;
    }
    let smoothed = smooth_curve(curve, 1.0 / 3.0)?;
    let levels: Vec<f64> = smoothed["spl"]
        .as_array()?
        .iter()
        .filter_map(Value::as_f64)
        .collect();
    let prominence = |index: usize, sign: f64| {
        let level = levels[index] * sign;
        let mut left = level;
        let mut right = level;
        for &candidate in levels[..index].iter().rev() {
            let candidate = candidate * sign;
            if candidate > level {
                break;
            }
            left = left.min(candidate);
        }
        for &candidate in &levels[index + 1..] {
            let candidate = candidate * sign;
            if candidate > level {
                break;
            }
            right = right.min(candidate);
        }
        level - left.max(right)
    };
    let mut peaks = Vec::new();
    let mut notches = Vec::new();
    for index in 1..samples.len() - 1 {
        let frequency = samples[index].0;
        if !(20.0..=20000.0).contains(&frequency) {
            continue;
        }
        let kind = if levels[index] > levels[index - 1] && levels[index] >= levels[index + 1] {
            1.0
        } else if levels[index] < levels[index - 1] && levels[index] <= levels[index + 1] {
            -1.0
        } else {
            continue;
        };
        let prominence = prominence(index, kind);
        if prominence >= 3.0 {
            if kind > 0.0 {
                peaks.push((frequency, levels[index], prominence));
            } else {
                notches.push((frequency, levels[index], prominence));
            }
        }
    }
    let merge = |mut extrema: Vec<(f64, f64, f64)>| {
        extrema.sort_by(|a, b| b.2.total_cmp(&a.2));
        let mut selected: Vec<(f64, f64)> = Vec::new();
        for (frequency, level, _) in extrema {
            // The report merges within 1/12 octave and retains three strongest
            // extrema of each kind; these are display rules, not EQ targets.
            if selected
                .iter()
                .all(|(kept, _)| (frequency / kept).log2().abs() >= 1.0 / 12.0)
            {
                selected.push((frequency, level));
                if selected.len() == 3 {
                    break;
                }
            }
        }
        selected
    };
    let band: Vec<(f64, f64)> = samples
        .iter()
        .zip(&levels)
        .filter(|((f, _), _)| (30.0..=200.0).contains(f))
        .map(|((f, _), s)| (*f, *s))
        .collect();
    let extension = band
        .iter()
        .map(|(_, s)| *s)
        .max_by(f64::total_cmp)
        .and_then(|peak| {
            let peak_frequency = band.iter().find(|(_, s)| *s == peak)?.0;
            samples
                .iter()
                .zip(&levels)
                .rev()
                .find(|((f, _), s)| *f <= peak_frequency && **s < peak - 6.0)
                .map(|((f, _), _)| *f)
        });
    Some(json!({"peaks":merge(peaks),"notches":merge(notches),"lf_extension_hz":extension}))
}

fn is_sub(name: &str) -> bool {
    let name = name.trim().to_lowercase();
    name == "lfe" || name.starts_with("sub")
}

/// Calculates downstream trims without applying in-chain level alignment twice.
pub fn level_compensation(channels: &Value) -> Vec<Value> {
    let Some(channels) = channels.as_object() else {
        return Vec::new();
    };
    let monitor_means: Vec<(&str, f64)> = channels
        .iter()
        .filter(|(name, _)| !is_sub(name))
        .filter_map(|(name, ch)| {
            band_mean(&ch["initial_curve"], 500.0, 3000.0).map(|mean| (name.as_str(), mean))
        })
        .collect();
    let Some(&(reference_name, reference)) =
        monitor_means.iter().min_by(|a, b| a.1.total_cmp(&b.1))
    else {
        return Vec::new();
    };
    let predicted_reference = band_mean(&channels[reference_name]["final_curve"], 500.0, 3000.0);
    channels.iter().map(|(name,ch)| {
        if ch["initial_curve"]["freq"].is_null() || ch["initial_curve"]["freq"].as_array().is_some_and(Vec::is_empty) { return json!({"speaker":name,"comp_db":null,"residual_db":null}); }
        let (low,high) = if is_sub(name) { (30.0,80.0) } else { (500.0,3000.0) };
        let mean = band_mean(&ch["initial_curve"],low,high);
        let predicted = band_mean(&ch["final_curve"],low,high);
        let compensation = match (predicted,predicted_reference) { (Some(value),Some(target)) => Some(target-value), _ => mean.map(|mean|reference-mean) };
        let landed = predicted.zip(compensation).map(|(p,c)|p+c);
        json!({"speaker":name,"comp_db":compensation,"residual_db":landed.map(|l|l-reference),"balance_db":landed.zip(predicted_reference).map(|(l,r)|l-r)})
    }).collect()
}

/// Formats the deepest-notch cell while retaining unavailable values.
pub fn notch_cell(notch: Option<f64>) -> String {
    notch_metric(notch).browser_html()
}

pub(crate) fn notch_metric(notch: Option<f64>) -> MetricCell {
    let Some(notch) = notch else {
        return MetricCell {
            value: "n/a".into(),
            tone: MetricTone::Pending,
            detail: None,
        };
    };
    let tone = if notch > -10.0 {
        MetricTone::Good
    } else if notch >= -20.0 {
        MetricTone::Warning
    } else {
        MetricTone::Poor
    };
    MetricCell {
        value: format!("{notch:+.1}"),
        tone,
        detail: None,
    }
}

pub(crate) fn level_table(rows: &[Value]) -> Option<DiagnosticBlock> {
    if rows.is_empty() {
        return None;
    }
    let rows = rows
        .iter()
        .map(|row| {
            let cell = |key: &str| {
                row[key]
                    .as_f64()
                    .map(|value| format!("{value:+.1}"))
                    .unwrap_or_else(|| "n/a".into())
            };
            vec![
                row["speaker"].as_str().unwrap_or_default().into(),
                cell("comp_db"),
                cell("residual_db"),
                cell("balance_db"),
            ]
        })
        .collect();
    Some(DiagnosticBlock::CurveTable {
        table: ReportTable {
            title: "Relative level compensation".into(),
            note: "Monitor band 0.5–3 kHz, sub band 30–80 Hz. Compensation is the attenuation still needed downstream, measured on the predicted post-DSP curves so any in-chain level alignment is not applied twice. The post-DSP columns include that trim: the offset is the predicted landing level versus the quietest monitor before correction (it includes EQ, time alignment and headroom attenuation; a negative value does not by itself indicate channel imbalance), and the balance column verifies the remaining spread against the quietest monitor after DSP. These are predictions, not verification captures.".into(),
            columns: [
                "Speaker",
                "Level compensation (dB)",
                "Post-DSP offset from original reference (dB)",
                "Post-DSP balance vs reference monitor (dB)",
            ].map(str::to_owned).to_vec(),
            rows,
        },
        dim_unavailable: false,
    })
}

pub(crate) fn landmarks_table(
    channels: impl IntoIterator<Item = (String, Option<Value>)>,
) -> Option<DiagnosticBlock> {
    let rows: Vec<_> = channels
        .into_iter()
        .map(|(name, marks)| {
            let Some(marks) = marks else {
                return vec![name, "n/a".into(), "n/a".into(), "n/a".into()];
            };
            let extension = marks["lf_extension_hz"]
                .as_f64()
                .map(|f| format!("{f:.0} Hz"))
                .unwrap_or_else(|| "< 20 Hz".into());
            let extrema = |key: &str| {
                let values: Vec<_> = marks[key]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|pair| {
                        Some(format!(
                            "{:.0} Hz / {:+.1} dB",
                            pair[0].as_f64()?,
                            pair[1].as_f64()?
                        ))
                    })
                    .collect();
                if values.is_empty() {
                    "—".into()
                } else {
                    values.join("\n")
                }
            };
            vec![name, extension, extrema("peaks"), extrema("notches")]
        })
        .collect();
    (!rows.is_empty()).then(|| DiagnosticBlock::CurveTable {
        table: ReportTable {
            title: "Frequency landmarks — peaks, notches, LF extension".into(),
            note: "1/3-octave-smoothed measured response. Extrema need 3 dB prominence, 1/12 octave apart (top 3 each). LF extension is the −6 dB point below the 30–200 Hz peak.".into(),
            columns: [
                "Speaker",
                "LF extension (−6 dB)",
                "Strongest peaks (freq / level)",
                "Strongest notches (freq / level)",
            ].map(str::to_owned).to_vec(),
            rows,
        },
        dim_unavailable: true,
    })
}

/// Builds curve-derived report summaries from one in-memory RoomEQ result.
pub fn curve_report(data: &Value) -> Value {
    let mut channels = Map::new();
    let mut landmarks = Vec::new();
    if let Some(input) = data["channels"].as_object() {
        for (name, channel) in input {
            let notch = deepest_notch_db(channel);
            let marks = response_landmarks(channel);
            channels.insert(name.clone(), json!({"landmarks":marks,"deepest_notch_db":notch,"notch_cell_html":notch_cell(notch)}));
            landmarks.push((name.clone(), marks));
        }
    }
    let rows = level_compensation(&data["channels"]);
    let html = |block: Option<DiagnosticBlock>| {
        block
            .map(|block| diagnostic_html(&[block]))
            .unwrap_or_default()
    };
    json!({"landmarks_table_html":html(landmarks_table(landmarks)),"level_compensation_html":html(level_table(&rows)),"level_compensation":rows,"channels":channels})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Number(a), Value::Number(b)) => assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs()
                    <= 1e-10 * b.as_f64().unwrap().abs().max(1.0),
                "{a} != {b}"
            ),
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    close(a, b);
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                for (key, b) in b {
                    close(a.get(key).unwrap(), b);
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn captured_python_curve_goldens_preserve_html_and_numeric_summaries() {
        let fixtures: Vec<Value> = serde_json::from_str(include_str!(
            "../../../scripts/tests/fixtures/acoustic_curves_python_goldens.json"
        ))
        .unwrap();
        for fixture in fixtures {
            close(&curve_report(&fixture["data"]), &fixture["expected"]);
        }
    }

    fn legacy_smoothed_sample(
        frequencies: &[f64],
        levels: &[f64],
        center: usize,
        octaves: f64,
    ) -> f64 {
        let ratio = 2.0_f64.powf(octaves / 2.0);
        let width = ratio.log10();
        let mut sum = 0.0;
        let mut weights = 0.0;
        for (&frequency, &level) in frequencies.iter().zip(levels) {
            if frequencies[center] / ratio <= frequency && frequency <= frequencies[center] * ratio
            {
                let weight = if width > 0.0 {
                    (1.0 - (frequency.log10() - frequencies[center].log10()).abs() / width).max(0.0)
                } else {
                    1.0
                };
                sum += level * weight;
                weights += weight;
            }
        }
        if weights > 0.0 {
            sum / weights
        } else {
            levels[center]
        }
    }

    #[test]
    fn prefix_smoothing_matches_legacy_on_irregular_and_dense_capture_grids() {
        for count in [321, 43_637] {
            let frequencies: Vec<_> = (0..count)
                .map(|i| 20.0 + 16_000.0 * (i as f64 / (count - 1) as f64).powf(1.7))
                .collect();
            let levels: Vec<_> = frequencies
                .iter()
                .map(|f| {
                    75.0 + 15.0 * (f.ln() * 5.0).sin()
                        - 30.0 * (-((f / 180.0).ln() / 0.03).powi(2)).exp()
                })
                .collect();
            let curve = json!({"freq":frequencies,"spl":levels});
            for octaves in [0.0, -0.5, 1e-8, 1.0 / 6.0, 1.0 / 3.0, 1.0, 100.0] {
                let smoothed = smooth_curve(&curve, octaves).unwrap();
                assert_eq!(smoothed["freq"], curve["freq"]);
                for index in (0..count)
                    .step_by((count / 64).max(1))
                    .chain(std::iter::once(count - 1))
                {
                    let expected = legacy_smoothed_sample(&frequencies, &levels, index, octaves);
                    let actual = smoothed["spl"][index].as_f64().unwrap();
                    assert!(
                        (actual - expected).abs() < 2e-8,
                        "count={count} width={octaves} index={index}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn band_mean_preserves_cancellation_and_rejects_bad_axes() {
        assert_eq!(
            band_mean(
                &json!({"freq":[500,1000,2000],"spl":[1e16,1,-1e16]}),
                500.0,
                3000.0
            ),
            Some(1.0 / 3.0)
        );
        assert_eq!(
            band_mean(&json!({"freq":[500,500],"spl":[1,2]}), 500.0, 3000.0),
            None
        );
        assert_eq!(
            band_mean(&json!({"freq":[500],"spl":[true]}), 500.0, 3000.0),
            None
        );
    }
}
