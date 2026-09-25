use super::consts::CEA2034_CURVE_NAMES;
use super::consts::CEA2034_CURVE_NAMES_DI;
use super::consts::CEA2034_CURVE_NAMES_FULL;
use super::misc::shorten_curve_name;
use crate::filter_color::filter_color;
use autoeq_report_wasm::{DashOption, Series};
use ndarray::Array1;
use std::collections::HashMap;

fn line_series(name: String, x: Vec<f64>, y: Vec<f64>, color: &str, width: f32) -> Series {
    Series {
        name,
        x,
        y: y.into_iter().map(Some).collect(),
        color: Some(color.to_string()),
        width,
        dash: DashOption::Solid,
        visible: true,
        y_axis: 0,
    }
}

/// CEA2034 per-panel series (panel index, series), without EQ.
pub(super) fn create_cea2034_series(
    curves: &HashMap<String, crate::Curve>,
) -> Vec<(usize, Series)> {
    let mut out = Vec::new();
    for (i, curve_name) in CEA2034_CURVE_NAMES.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            out.push((
                i,
                line_series(
                    shorten_curve_name(curve_name).to_string(),
                    curve.freq.to_vec(),
                    curve.spl.to_vec(),
                    filter_color(i),
                    2.0,
                ),
            ));
        }
    }
    out
}

/// CEA2034 per-panel series with the EQ response applied.
pub(super) fn create_cea2034_with_eq_series(
    curves: &HashMap<String, crate::Curve>,
    eq_response: &Array1<f64>,
) -> Vec<(usize, Series)> {
    let mut out = Vec::new();
    for (i, curve_name) in CEA2034_CURVE_NAMES.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            out.push((
                i,
                line_series(
                    format!("{} w/EQ", shorten_curve_name(curve_name)),
                    curve.freq.to_vec(),
                    (&curve.spl + eq_response).to_vec(),
                    filter_color(i + 4),
                    2.0,
                ),
            ));
        }
    }
    out
}

/// Combined CEA2034 + DI series for one overview panel (DI on `y_axis` 1).
pub(super) fn create_cea2034_combined_series(
    curves: &HashMap<String, crate::Curve>,
) -> Vec<Series> {
    let mut out = Vec::new();
    for (i, curve_name) in CEA2034_CURVE_NAMES_FULL.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            out.push(line_series(
                shorten_curve_name(curve_name).to_string(),
                curve.freq.to_vec(),
                curve.spl.to_vec(),
                filter_color(i),
                2.0,
            ));
        }
    }
    for (j, curve_name) in CEA2034_CURVE_NAMES_DI.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            let mut s = line_series(
                shorten_curve_name(curve_name).to_string(),
                curve.freq.to_vec(),
                curve.spl.to_vec(),
                filter_color(j + 2),
                2.0,
            );
            s.y_axis = 1;
            out.push(s);
        }
    }
    out
}

/// Combined CEA2034 + DI series with the EQ response applied to the main curves.
pub(super) fn create_cea2034_with_eq_combined_series(
    curves: &HashMap<String, crate::Curve>,
    eq_response: &Array1<f64>,
) -> Vec<Series> {
    let mut out = Vec::new();
    for (i, curve_name) in CEA2034_CURVE_NAMES_FULL.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            out.push(line_series(
                format!("{} w/EQ", shorten_curve_name(curve_name)),
                curve.freq.to_vec(),
                (&curve.spl + eq_response).to_vec(),
                filter_color(i),
                2.0,
            ));
        }
    }
    for (j, curve_name) in CEA2034_CURVE_NAMES_DI.iter().enumerate() {
        if let Some(curve) = curves.get(*curve_name) {
            let mut s = line_series(
                shorten_curve_name(curve_name).to_string(),
                curve.freq.to_vec(),
                curve.spl.to_vec(),
                filter_color(j + 2),
                2.0,
            );
            s.y_axis = 1;
            out.push(s);
        }
    }
    out
}
