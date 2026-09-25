use autoeq_report_wasm::{DashOption, Series};

// ±1 dB reference segments spanning x=100..10000, as named line series
// (schema `hlines` always span the full width, which would change the look).
pub fn make_ref_series() -> Vec<Series> {
    let xs = vec![100.0_f64, 10000.0_f64];
    vec![
        Series {
            name: "+1 dB ref".to_string(),
            x: xs.clone(),
            y: vec![Some(1.0_f64), Some(1.0_f64)],
            color: Some("#000000".to_string()),
            width: 1.0,
            dash: DashOption::Solid,
            visible: true,
            y_axis: 0,
        },
        Series {
            name: "-1 dB ref".to_string(),
            x: xs,
            y: vec![Some(-1.0_f64), Some(-1.0_f64)],
            color: Some("#000000".to_string()),
            width: 1.0,
            dash: DashOption::Solid,
            visible: true,
            y_axis: 0,
        },
    ]
}
