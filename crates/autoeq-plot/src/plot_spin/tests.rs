use super::create::{
    create_cea2034_combined_series, create_cea2034_series,
    create_cea2034_with_eq_combined_series, create_cea2034_with_eq_series,
};
use super::misc::shorten_curve_name;
use crate::ref_lines::make_ref_series;
use ndarray::Array1;
use std::collections::HashMap;

fn four_panel_curves() -> HashMap<String, crate::Curve> {
    let mut curves = HashMap::new();
    let freq = Array1::from(vec![20.0, 100.0, 1000.0, 10000.0, 20000.0]);
    let spl = Array1::from(vec![80.0, 85.0, 90.0, 85.0, 80.0]);
    for name in [
        "On Axis",
        "Listening Window",
        "Early Reflections",
        "Sound Power",
    ] {
        curves.insert(
            name.to_string(),
            crate::Curve {
                freq: freq.clone(),
                spl: spl.clone(),
                phase: None,
                ..Default::default()
            },
        );
    }
    curves
}

fn di_curves() -> HashMap<String, crate::Curve> {
    let mut curves = HashMap::new();
    let freq = Array1::from(vec![100.0, 1000.0, 10000.0]);
    let spl_primary = Array1::from(vec![80.0, 85.0, 82.0]);
    for name in [
        "On Axis",
        "Listening Window",
        "Early Reflections",
        "Sound Power",
    ] {
        curves.insert(
            name.to_string(),
            crate::Curve {
                freq: freq.clone(),
                spl: spl_primary.clone(),
                phase: None,
                ..Default::default()
            },
        );
    }
    let spl_di = Array1::from(vec![5.0, 6.0, 7.0]);
    for name in ["Early Reflections DI", "Sound Power DI"] {
        curves.insert(
            name.to_string(),
            crate::Curve {
                freq: freq.clone(),
                spl: spl_di.clone(),
                phase: None,
                ..Default::default()
            },
        );
    }
    curves
}

#[test]
fn test_create_cea2034_series() {
    let curves = four_panel_curves();
    let series = create_cea2034_series(&curves);
    assert_eq!(series.len(), 4);
    // Panels in order, short names.
    let panels: Vec<usize> = series.iter().map(|(p, _)| *p).collect();
    assert_eq!(panels, vec![0, 1, 2, 3]);
    let names: Vec<&str> = series.iter().map(|(_, s)| s.name.as_str()).collect();
    assert_eq!(names, vec!["ON", "LW", "ER", "SP"]);

    let eq_response = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    let eq_series = create_cea2034_with_eq_series(&curves, &eq_response);
    assert_eq!(eq_series.len(), 4);
    assert_eq!(eq_series[0].1.name, "ON w/EQ");
}

#[test]
fn test_make_ref_series_values() {
    let lines = make_ref_series();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].x, vec![100.0, 10000.0]);
    assert_eq!(lines[1].x, vec![100.0, 10000.0]);
    assert_eq!(lines[0].y, vec![Some(1.0), Some(1.0)]);
    assert_eq!(lines[1].y, vec![Some(-1.0), Some(-1.0)]);
    assert_eq!(lines[0].name, "+1 dB ref");
    assert_eq!(lines[1].name, "-1 dB ref");
}

#[test]
fn test_create_cea2034_combined_series_counts_and_axes() {
    let curves = di_curves();
    let series = create_cea2034_combined_series(&curves);
    assert_eq!(series.len(), 6);

    let names: Vec<&str> = series.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"ERDI"));
    assert!(names.contains(&"SPDI"));

    // DI series target the secondary axis; mains stay primary.
    for s in &series {
        if s.name == "ERDI" || s.name == "SPDI" {
            assert_eq!(s.y_axis, 1, "{}", s.name);
        } else {
            assert_eq!(s.y_axis, 0, "{}", s.name);
        }
    }
}

#[test]
fn test_create_cea2034_with_eq_combined_series_counts_and_names() {
    let curves = di_curves();
    let eq = Array1::from(vec![1.0, -1.0, 0.5]);
    let series = create_cea2034_with_eq_combined_series(&curves, &eq);
    assert_eq!(series.len(), 6);
    let names: Vec<&str> = series.iter().map(|s| s.name.as_str()).collect();
    assert!(names.iter().any(|n| *n == "ON w/EQ"));
    assert!(names.iter().any(|n| *n == "LW w/EQ"));
    assert!(names.iter().any(|n| *n == "ER w/EQ"));
    assert!(names.iter().any(|n| *n == "SP w/EQ"));
    assert!(names.iter().any(|n| *n == "ERDI"));
    assert!(names.iter().any(|n| *n == "SPDI"));
    // EQ shifts the primary values but leaves DI untouched.
    let on = series.iter().find(|s| s.name == "ON w/EQ").unwrap();
    assert_eq!(on.y, vec![Some(81.0), Some(84.0), Some(82.5)]);
    let erdi = series.iter().find(|s| s.name == "ERDI").unwrap();
    assert_eq!(erdi.y, vec![Some(5.0), Some(6.0), Some(7.0)]);
    assert_eq!(erdi.y_axis, 1);
}

#[test]
fn test_shorten_curve_name() {
    // Test basic curve name abbreviations
    assert_eq!(shorten_curve_name("On Axis"), "ON");
    assert_eq!(shorten_curve_name("Listening Window"), "LW");
    assert_eq!(shorten_curve_name("Early Reflections"), "ER");
    assert_eq!(shorten_curve_name("Sound Power"), "SP");
    assert_eq!(shorten_curve_name("Estimated In-Room Response"), "PIR");

    // Test DI curve abbreviations
    assert_eq!(shorten_curve_name("Early Reflections DI"), "ERDI");
    assert_eq!(shorten_curve_name("Sound Power DI"), "SPDI");

    // Test unknown curve name (should return original)
    assert_eq!(shorten_curve_name("Unknown Curve"), "Unknown Curve");
    assert_eq!(shorten_curve_name(""), "");

    // Test case sensitivity (should return original since no exact match)
    assert_eq!(shorten_curve_name("on axis"), "on axis");
    assert_eq!(shorten_curve_name("ON AXIS"), "ON AXIS");
}

#[test]
fn test_shorten_curve_name_edge_cases() {
    // Long unknown name should be returned unchanged
    let long_name = "A".repeat(200);
    assert_eq!(shorten_curve_name(&long_name), long_name.as_str());

    // Whitespace-only unknown name
    assert_eq!(shorten_curve_name("   "), "   ");

    // Names containing special characters
    assert_eq!(shorten_curve_name("On Axis @ 0°"), "On Axis @ 0°");

    // Partial match should not abbreviate
    assert_eq!(shorten_curve_name("On Axis DI"), "On Axis DI");
    assert_eq!(shorten_curve_name("Sound Power X"), "Sound Power X");
}
