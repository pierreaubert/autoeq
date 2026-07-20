use super::extract::extract_cea2034_curves_original;
use super::fetch::{
    fetch_measurement_plot_data, fetch_measurement_plot_data_at_cache_root, read_spinorama,
    read_spinorama_at_cache_root,
};
use super::types::Cea2034Data;
use crate::Curve;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

/// Load spinorama measurement with full CEA2034 spin data
///
/// This fetches the requested curve and also extracts all CEA2034 curves
/// when the measurement type is CEA2034.
///
/// # Arguments
/// * `speaker` - Speaker name (e.g., "KEF R3")
/// * `version` - Version (e.g., "asr")
/// * `measurement` - Measurement type (e.g., "CEA2034")
/// * `curve_name` - Specific curve to use as primary (e.g., "Listening Window")
///
/// # Returns
/// Tuple of (primary curve, optional spin data)
pub async fn load_spinorama_with_spin(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
) -> Result<(Curve, Option<Cea2034Data>), Box<dyn Error>> {
    load_spinorama_with_spin_from_cache_root(speaker, version, measurement, curve_name, None).await
}

/// Load spinorama measurement data using an explicit cache root.
pub async fn load_spinorama_with_spin_at_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
    cache_root: &Path,
) -> Result<(Curve, Option<Cea2034Data>), Box<dyn Error>> {
    load_spinorama_with_spin_from_cache_root(
        speaker,
        version,
        measurement,
        curve_name,
        Some(cache_root),
    )
    .await
}

async fn load_spinorama_with_spin_from_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
    cache_root: Option<&Path>,
) -> Result<(Curve, Option<Cea2034Data>), Box<dyn Error>> {
    // Handle Estimated In-Room Response specially
    if measurement == "Estimated In-Room Response"
        || (measurement == "CEA2034" && curve_name == "Estimated In-Room Response")
    {
        let plot_data = fetch_plot_data(speaker, version, "CEA2034", cache_root).await?;
        let curves = extract_cea2034_curves_original(&plot_data, "CEA2034")?;

        let pir_curve = curves
            .get("Estimated In-Room Response")
            .ok_or_else(|| {
                Box::<dyn Error>::from("Estimated In-Room Response curve not found in CEA2034 data")
            })?
            .clone();

        let spin_data = build_cea2034_data(curves)?;
        return Ok((pir_curve, Some(spin_data)));
    }

    // Standard curve fetch
    let curve = match cache_root {
        Some(root) => {
            read_spinorama_at_cache_root(speaker, version, measurement, curve_name, root).await?
        }
        None => read_spinorama(speaker, version, measurement, curve_name).await?,
    };

    // Extract spin data if CEA2034
    let spin_data = if measurement == "CEA2034" {
        let plot_data = fetch_plot_data(speaker, version, measurement, cache_root).await?;
        let curves = extract_cea2034_curves_original(&plot_data, "CEA2034")?;
        Some(build_cea2034_data(curves)?)
    } else {
        None
    };

    Ok((curve, spin_data))
}

async fn fetch_plot_data(
    speaker: &str,
    version: &str,
    measurement: &str,
    cache_root: Option<&Path>,
) -> Result<serde_json::Value, Box<dyn Error>> {
    match cache_root {
        Some(root) => {
            fetch_measurement_plot_data_at_cache_root(speaker, version, measurement, root).await
        }
        None => fetch_measurement_plot_data(speaker, version, measurement).await,
    }
}

/// Build Cea2034Data from curves HashMap
pub fn build_cea2034_data(curves: HashMap<String, Curve>) -> Result<Cea2034Data, Box<dyn Error>> {
    Ok(crate::cea2034::SpinoramaBundleBuilder::new()
        .curves(curves)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Curve;
    use ndarray::Array1;

    fn make_curve(spl: &[f64]) -> Curve {
        Curve {
            freq: Array1::from_vec(vec![100.0, 500.0, 1000.0]),
            spl: Array1::from_vec(spl.to_vec()),
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn build_cea2034_data_computes_indices() {
        let mut curves = HashMap::new();
        for name in [
            "On Axis",
            "Listening Window",
            "Early Reflections",
            "Sound Power",
            "Estimated In-Room Response",
        ] {
            curves.insert(name.to_string(), make_curve(&[80.0, 82.0, 81.0]));
        }
        let data = build_cea2034_data(curves).unwrap();
        assert_eq!(data.on_axis.freq.len(), 3);
        assert_eq!(data.er_di.spl.len(), 3);
        assert_eq!(data.sp_di.spl.len(), 3);
    }

    #[test]
    fn build_cea2034_data_missing_curve_errors() {
        let mut curves = HashMap::new();
        curves.insert("On Axis".to_string(), make_curve(&[80.0, 82.0, 81.0]));
        let result = build_cea2034_data(curves);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Missing CEA2034 curve")
        );
    }
}
