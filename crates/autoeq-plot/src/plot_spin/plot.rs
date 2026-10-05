use super::consts::CEA2034_CURVE_NAMES;
use super::create::{
    create_cea2034_combined_series, create_cea2034_series, create_cea2034_with_eq_combined_series,
    create_cea2034_with_eq_series,
};
use super::misc::shorten_curve_name;
use crate::filter_color::filter_color;
use crate::ref_lines::make_ref_series;
use crate::trend_lines::{
    calculate_tonal_balance, create_regression_trace, generate_regression_line,
};
use autoeq_report_wasm::{AxisSpec, Figure, XScale};
use ndarray::Array1;
use std::collections::HashMap;

fn log_freq_axis() -> AxisSpec {
    AxisSpec {
        label: "Frequency (Hz)".to_string(),
        scale: XScale::Log,
        min: Some(20.0),
        max: Some(20000.0),
    }
}

fn spl_axis(lo: f64, hi: f64) -> AxisSpec {
    AxisSpec {
        label: "SPL (dB)".to_string(),
        scale: XScale::Linear,
        min: Some(lo),
        max: Some(hi),
    }
}

const DETAIL_TITLES: [&str; 4] = [
    "On Axis",
    "Listening Window",
    "Early Reflections",
    "Sound Power",
];

const TONAL_TITLES: [&str; 4] = [
    "On Axis Tonal Balance",
    "Listening Window Tonal Balance",
    "Early Reflections Tonal Balance",
    "Sound Power Tonal Balance",
];

/// y ranges per detail/tonal panel (as before: ±10, ±10, -15..5, -15..5).
fn detail_y_range(panel: usize) -> (f64, f64) {
    match panel {
        0 | 1 => (-10.0, 10.0),
        _ => (-15.0, 5.0),
    }
}

/// Detailed CEA2034 spinorama figures (one per panel; previously a 2x2 grid).
///
/// Shows On Axis, Listening Window, Early Reflections, and Sound Power curves,
/// optionally with EQ-applied variants overlaid, plus ±1 dB references.
pub fn plot_spin_details(
    cea2034_curves: Option<&HashMap<String, crate::Curve>>,
    eq_response: Option<&Array1<f64>>,
) -> Vec<Figure> {
    let mut panels: [Vec<autoeq_report_wasm::Series>; 4] = Default::default();
    if let Some(curves) = cea2034_curves {
        for (panel, s) in create_cea2034_series(curves) {
            panels[panel].push(s);
        }
        if let Some(eq_resp) = eq_response {
            for (panel, s) in create_cea2034_with_eq_series(curves, eq_resp) {
                panels[panel].push(s);
            }
        }
    }
    // ±1 dB references on the first two panels (as before).
    let refs = make_ref_series();
    panels[0].extend(refs.clone());
    panels[1].extend(refs);

    DETAIL_TITLES
        .iter()
        .enumerate()
        .map(|(i, title)| {
            let (lo, hi) = detail_y_range(i);
            Figure {
                title: (*title).to_string(),
                x: log_freq_axis(),
                y: spl_axis(lo, hi),
                y2: None,
                series: std::mem::take(&mut panels[i]),
                hlines: vec![],
                vlines: vec![],
                xranges: vec![],
                annotations: vec![],
                legend: true,
            }
        })
        .collect()
}

/// Tonal-balance figures (regression slopes per panel, with and without EQ).
pub fn plot_spin_tonal(
    cea2034_curves: Option<&HashMap<String, crate::Curve>>,
    eq_response: Option<&Array1<f64>>,
) -> Vec<Figure> {
    let mut panels: [Vec<autoeq_report_wasm::Series>; 4] = Default::default();
    if let Some(curves) = cea2034_curves {
        for (i, curve_name) in CEA2034_CURVE_NAMES.iter().enumerate() {
            if let Some(curve) = curves.get(*curve_name)
                && let Some((slope, intercept)) =
                    calculate_tonal_balance(&curve.freq, &curve.spl, 100.0, 10000.0)
            {
                let regression_line = generate_regression_line(slope, intercept, &curve.freq);
                panels[i].push(create_regression_trace(
                    &curve.freq,
                    &regression_line,
                    &format!("{} {:.2} dB/oct", shorten_curve_name(curve_name), slope),
                    filter_color(i),
                ));
            }
        }
        if let Some(eq_resp) = eq_response {
            for (i, curve_name) in CEA2034_CURVE_NAMES.iter().enumerate() {
                if let Some(curve) = curves.get(*curve_name) {
                    let eq_applied = &curve.spl + eq_resp;
                    if let Some((slope, intercept)) =
                        calculate_tonal_balance(&curve.freq, &eq_applied, 100.0, 10000.0)
                    {
                        let regression_line =
                            generate_regression_line(slope, intercept, &curve.freq);
                        panels[i].push(create_regression_trace(
                            &curve.freq,
                            &regression_line,
                            &format!(
                                "{} w/EQ {:.2} dB/oct",
                                shorten_curve_name(curve_name),
                                slope
                            ),
                            filter_color(i + 4),
                        ));
                    }
                }
            }
        }
    }

    TONAL_TITLES
        .iter()
        .enumerate()
        .map(|(i, title)| {
            let (lo, hi) = detail_y_range(i);
            Figure {
                title: (*title).to_string(),
                x: log_freq_axis(),
                y: spl_axis(lo, hi),
                y2: None,
                series: std::mem::take(&mut panels[i]),
                hlines: vec![],
                vlines: vec![],
                xranges: vec![],
                annotations: vec![],
                legend: true,
            }
        })
        .collect()
}

/// CEA2034 spinorama overview figures (response + DI on the secondary axis).
///
/// Shows directivity indices (ERDI, SPDI) overlaid on the main CEA2034 curves,
/// with and without the EQ response.
pub fn plot_spin(
    cea2034_curves: Option<&HashMap<String, crate::Curve>>,
    eq_response: Option<&Array1<f64>>,
) -> Vec<Figure> {
    let di_axis = || AxisSpec {
        label: "DI (dB)".to_string(),
        scale: XScale::Linear,
        min: Some(-5.0),
        max: Some(45.0),
    };
    let mut figs = Vec::new();
    if let Some(curves) = cea2034_curves {
        figs.push(Figure {
            title: "CEA2034".to_string(),
            x: log_freq_axis(),
            y: spl_axis(-40.0, 10.0),
            y2: Some(di_axis()),
            series: create_cea2034_combined_series(curves),
            hlines: vec![],
            vlines: vec![],
            xranges: vec![],
            annotations: vec![],
            legend: true,
        });
        if let Some(eq_resp) = eq_response {
            figs.push(Figure {
                title: "CEA2034 + EQ".to_string(),
                x: log_freq_axis(),
                y: spl_axis(-40.0, 10.0),
                y2: Some(di_axis()),
                series: create_cea2034_with_eq_combined_series(curves, eq_resp),
                hlines: vec![],
                vlines: vec![],
                xranges: vec![],
                annotations: vec![],
                legend: true,
            });
        }
    }
    figs
}
