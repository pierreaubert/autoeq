use super::backend::{
    FsMeasurementCache, MeasurementBackend, MeasurementCache, ReqwestMeasurementBackend,
};
#[cfg(test)]
use super::backend::{InMemoryMeasurementCache, MockMeasurementBackend};
use super::extract::extract_contour_data;
use super::extract::extract_curve_by_name;
use super::extract::extract_directivity_curves;
use super::parse::parse_headphone_csv;
use super::types::ContourPlotData;
use crate::read::directory::{
    cache_root, data_dir_for_cache_root, headphone_cache_dir_for_cache_root, measurement_filename,
};
use crate::read::plot::{normalize_plotly_json_from_str, normalize_plotly_value_with_suggestions};
use crate::{DirectivityData, MeasurementRecord};
use serde_json::Value;
use std::error::Error;
use std::path::Path;
use urlencoding;

/// Fetch a frequency response curve from the spinorama API
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Measurement version
/// * `measurement` - Measurement type (e.g., "CEA2034")
/// * `curve_name` - Name of the specific curve to extract
///
/// # Returns
/// * Result containing a Curve struct or an error
pub async fn read_spinorama(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
) -> Result<crate::Curve, Box<dyn Error>> {
    fetch_curve_from_api(speaker, version, measurement, curve_name).await
}

/// Fetch a frequency response curve using an explicit cache root.
pub async fn read_spinorama_at_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
    cache_root: &Path,
) -> Result<crate::Curve, Box<dyn Error>> {
    fetch_curve_from_api_at_cache_root(speaker, version, measurement, curve_name, cache_root).await
}

/// Fetch a frequency response curve from the spinorama API
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Measurement version
/// * `measurement` - Measurement type (e.g., "CEA2034")
/// * `curve_name` - Name of the specific curve to extract
///
/// # Returns
/// * Result containing a Curve struct or an error
pub async fn fetch_curve_from_api(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
) -> Result<crate::Curve, Box<dyn Error>> {
    // Fetch the full measurement once, then extract the requested curve
    let plot_data = fetch_measurement_plot_data(speaker, version, measurement).await?;
    extract_curve_by_name(&plot_data, measurement, curve_name)
}

/// Fetch a frequency response curve using an explicit cache root.
pub async fn fetch_curve_from_api_at_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
    cache_root: &Path,
) -> Result<crate::Curve, Box<dyn Error>> {
    let plot_data =
        fetch_measurement_plot_data_at_cache_root(speaker, version, measurement, cache_root)
            .await?;
    extract_curve_by_name(&plot_data, measurement, curve_name)
}

/// Fetch a curve together with an API-origin provenance record.
pub async fn fetch_record_from_api(
    speaker: &str,
    version: &str,
    measurement: &str,
    curve_name: &str,
) -> Result<MeasurementRecord, Box<dyn Error>> {
    let curve = fetch_curve_from_api(speaker, version, measurement, curve_name).await?;
    let uri = format!(
        "https://api.spinorama.org/v1/speaker/{}/version/{}/measurements/{}?measurement_format=json#{}",
        urlencoding::encode(speaker),
        urlencoding::encode(version),
        urlencoding::encode(measurement),
        urlencoding::encode(curve_name),
    );
    MeasurementRecord::from_api(curve, uri).map_err(|error| error.into())
}

/// Fetch and parse the full Plotly JSON object for a given measurement (single HTTP GET)
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Measurement version
/// * `measurement` - Measurement type (e.g., "CEA2034")
///
/// # Returns
/// * Result containing the Plotly JSON data or an error
pub async fn fetch_measurement_plot_data(
    speaker: &str,
    version: &str,
    measurement: &str,
) -> Result<Value, Box<dyn Error>> {
    let root = cache_root();
    fetch_measurement_plot_data_at_cache_root(speaker, version, measurement, &root).await
}

/// Fetch and parse Plotly JSON using an explicit cache root.
pub async fn fetch_measurement_plot_data_at_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    cache_root: &Path,
) -> Result<Value, Box<dyn Error>> {
    fetch_measurement_plot_data_with_backend_at_cache_root(
        speaker,
        version,
        measurement,
        cache_root,
        &ReqwestMeasurementBackend::new(),
        &FsMeasurementCache::new(),
    )
    .await
}

/// Fetch and parse Plotly JSON using the supplied backend and cache.
///
/// This is the seam that makes network-dependent code testable: tests can pass
/// an in-memory cache and a [`MockMeasurementBackend`] to run offline.
pub async fn fetch_measurement_plot_data_with_backend(
    speaker: &str,
    version: &str,
    measurement: &str,
    backend: &dyn MeasurementBackend,
    cache: &dyn MeasurementCache,
) -> Result<Value, Box<dyn Error>> {
    let root = cache_root();
    fetch_measurement_plot_data_with_backend_at_cache_root(
        speaker,
        version,
        measurement,
        &root,
        backend,
        cache,
    )
    .await
}

/// Fetch and parse Plotly JSON using supplied adapters and an explicit root.
pub async fn fetch_measurement_plot_data_with_backend_at_cache_root(
    speaker: &str,
    version: &str,
    measurement: &str,
    cache_root: &Path,
    backend: &dyn MeasurementBackend,
    cache: &dyn MeasurementCache,
) -> Result<Value, Box<dyn Error>> {
    // 1) Try local cache first: data_cached/speakers/org.spinorama/{speaker}/{measurement}.json
    // We keep filename identical to measurement name when possible (with path separators replaced).
    let cache_dir = data_dir_for_cache_root(cache_root, speaker);
    let cache_file = cache_dir.join(measurement_filename(measurement));

    if let Ok(Some(content)) = cache.read_to_string(&cache_file).await {
        if let Ok(plot_data) = normalize_plotly_json_from_str(&content) {
            return Ok(plot_data);
        } else {
            log::debug!(
                "⚠️  Cache file exists but could not be parsed as Plotly JSON: {:?}",
                cache_file
            );
        }
    }

    // URL-encode the parameters
    let encoded_speaker = urlencoding::encode(speaker);
    let encoded_version = urlencoding::encode(version);
    let encoded_measurement = urlencoding::encode(measurement);

    let url = format!(
        "https://api.spinorama.org/v1/speaker/{}/version/{}/measurements/{}?measurement_format=json",
        encoded_speaker, encoded_version, encoded_measurement
    );

    let body = backend.get_text(&url).await?;
    let api_response: Value = serde_json::from_str(&body)?;

    // Normalize from API response (array-of-string JSON) to Plotly JSON object
    let plot_data = normalize_plotly_value_with_suggestions(&api_response).await?;

    // 2) Save normalized Plotly JSON to cache for future use
    if let Err(e) = cache.create_dir_all(&cache_dir).await {
        log::debug!("⚠️  Failed to create cache dir {:?}: {}", cache_dir, e);
    } else {
        match serde_json::to_string(&plot_data) {
            Ok(serialized) => {
                if let Err(e) = cache.write(&cache_file, &serialized).await {
                    log::debug!("⚠️  Failed to write cache file {:?}: {}", cache_file, e);
                }
            }
            Err(e) => log::debug!("⚠️  Failed to serialize plot data for cache: {}", e),
        }
    }

    Ok(plot_data)
}

/// Fetch directivity data (SPL Horizontal and SPL Vertical) from the spinorama API
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Measurement version
///
/// # Returns
/// * Result containing DirectivityData or an error
///
/// # Details
/// Fetches both "SPL Horizontal" and "SPL Vertical" measurements, extracting
/// all angle traces (typically -60° to +60° in 10° increments).
pub async fn fetch_directivity_data(
    speaker: &str,
    version: &str,
) -> Result<DirectivityData, Box<dyn Error>> {
    let root = cache_root();
    fetch_directivity_data_at_cache_root(speaker, version, &root).await
}

/// Fetch directivity data using an explicit cache root.
pub async fn fetch_directivity_data_at_cache_root(
    speaker: &str,
    version: &str,
    cache_root: &Path,
) -> Result<DirectivityData, Box<dyn Error>> {
    // Fetch horizontal data
    let horizontal_data =
        fetch_measurement_plot_data_at_cache_root(speaker, version, "SPL Horizontal", cache_root)
            .await?;
    let horizontal = extract_directivity_curves(&horizontal_data)?;

    if horizontal.is_empty() {
        return Err("No horizontal directivity curves found".into());
    }

    // Fetch vertical data
    let vertical_data =
        fetch_measurement_plot_data_at_cache_root(speaker, version, "SPL Vertical", cache_root)
            .await?;
    let vertical = extract_directivity_curves(&vertical_data)?;

    if vertical.is_empty() {
        return Err("No vertical directivity curves found".into());
    }

    Ok(DirectivityData {
        horizontal,
        vertical,
    })
}

/// Fetch contour plot data (SPL Horizontal Contour or SPL Vertical Contour) from the spinorama API
///
/// # Arguments
/// * `speaker` - Speaker name
/// * `version` - Measurement version
/// * `plane` - Either "horizontal" or "vertical"
///
/// # Returns
/// * Result containing ContourPlotData or an error
///
/// # Details
/// Fetches "SPL Horizontal Contour" or "SPL Vertical Contour" measurements, which are
/// pre-computed heatmaps with full angle range (typically -180° to +180°).
pub async fn fetch_contour_data(
    speaker: &str,
    version: &str,
    plane: &str,
) -> Result<ContourPlotData, Box<dyn Error>> {
    let root = cache_root();
    fetch_contour_data_at_cache_root(speaker, version, plane, &root).await
}

/// Fetch contour data using an explicit cache root.
pub async fn fetch_contour_data_at_cache_root(
    speaker: &str,
    version: &str,
    plane: &str,
    cache_root: &Path,
) -> Result<ContourPlotData, Box<dyn Error>> {
    let measurement = match plane.to_lowercase().as_str() {
        "horizontal" | "h" => "SPL Horizontal Contour",
        "vertical" | "v" => "SPL Vertical Contour",
        _ => {
            return Err(format!("Invalid plane: {}. Use 'horizontal' or 'vertical'", plane).into());
        }
    };

    let plot_data =
        fetch_measurement_plot_data_at_cache_root(speaker, version, measurement, cache_root)
            .await?;
    extract_contour_data(&plot_data)
}

/// Fetch a headphone's frequency response from the spinorama API.
///
/// Uses `GET /v1/headphone/{name}/frequency_response`.
/// The API returns CSV data (4-column: L_freq, L_spl, R_freq, R_spl with header rows).
/// Caches the raw CSV and also writes a simplified 2-column CSV for the optimization pipeline.
pub async fn fetch_headphone_frequency_response(
    headphone: &str,
) -> Result<(String, crate::Curve), Box<dyn Error>> {
    let root = cache_root();
    fetch_headphone_frequency_response_at_cache_root(headphone, &root).await
}

/// Fetch a headphone response using an explicit cache root.
pub async fn fetch_headphone_frequency_response_at_cache_root(
    headphone: &str,
    cache_root: &Path,
) -> Result<(String, crate::Curve), Box<dyn Error>> {
    fetch_headphone_frequency_response_with_backend_at_cache_root(
        headphone,
        cache_root,
        &ReqwestMeasurementBackend::new(),
        &FsMeasurementCache::new(),
    )
    .await
}

/// Fetch a headphone response using the supplied backend and cache.
pub async fn fetch_headphone_frequency_response_with_backend(
    headphone: &str,
    backend: &dyn MeasurementBackend,
    cache: &dyn MeasurementCache,
) -> Result<(String, crate::Curve), Box<dyn Error>> {
    let root = cache_root();
    fetch_headphone_frequency_response_with_backend_at_cache_root(headphone, &root, backend, cache)
        .await
}

/// Fetch a headphone response using supplied adapters and an explicit root.
pub async fn fetch_headphone_frequency_response_with_backend_at_cache_root(
    headphone: &str,
    cache_root: &Path,
    backend: &dyn MeasurementBackend,
    cache: &dyn MeasurementCache,
) -> Result<(String, crate::Curve), Box<dyn Error>> {
    let cache_dir = headphone_cache_dir_for_cache_root(cache_root, headphone);
    let raw_cache = cache_dir.join("frequency_response_raw.csv");
    let csv_path = cache_dir.join("measurement.csv");

    // Try local cache first
    let csv_text = if let Ok(Some(content)) = cache.read_to_string(&raw_cache).await {
        content
    } else {
        fetch_headphone_csv_from_backend(headphone, &cache_dir, &raw_cache, backend, cache).await?
    };

    // Parse the CSV (handles 2-col and 4-col with multiple header rows)
    let curve = parse_headphone_csv(&csv_text)?;

    // Write simplified 2-column CSV for the optimization pipeline
    cache.create_dir_all(&cache_dir).await?;
    let mut out = String::from("frequency,spl\n");
    for (f, s) in curve.freq.iter().zip(curve.spl.iter()) {
        out.push_str(&format!("{},{}\n", f, s));
    }
    cache.write(&csv_path, &out).await?;

    Ok((csv_path.to_string_lossy().to_string(), curve))
}

async fn fetch_headphone_csv_from_backend(
    headphone: &str,
    cache_dir: &std::path::Path,
    raw_cache: &std::path::Path,
    backend: &dyn MeasurementBackend,
    cache: &dyn MeasurementCache,
) -> Result<String, Box<dyn Error>> {
    let encoded = urlencoding::encode(headphone);
    let url = format!(
        "https://api.spinorama.org/v1/headphone/{}/frequency_response",
        encoded
    );

    let body = backend.get_text(&url).await?;

    // Check for JSON error response (the API returns JSON errors even though
    // successful responses are CSV)
    if body.trim_start().starts_with('{')
        && let Ok(json) = serde_json::from_str::<Value>(&body)
        && let Some(error_msg) = json.get("error").and_then(|e| e.as_str())
    {
        return Err(error_msg.to_string().into());
    }

    // Cache the raw CSV
    if let Err(e) = cache.create_dir_all(cache_dir).await {
        log::debug!("Failed to create headphone cache dir: {}", e);
    } else if let Err(e) = cache.write(raw_cache, &body).await {
        log::debug!("Failed to write headphone cache: {}", e);
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::{
        data_dir_for_cache_root, headphone_cache_dir_for_cache_root, measurement_filename,
    };
    use serde_json::json;
    use tempfile::TempDir;

    fn plotly_object() -> Value {
        json!({
            "data": [
                {"name": "On Axis", "x": [100.0, 1000.0], "y": [80.0, 81.0]},
                {"name": "Listening Window", "x": [100.0, 1000.0], "y": [82.0, 83.0]},
                {"name": "Early Reflections", "x": [100.0, 1000.0], "y": [81.0, 82.0]},
                {"name": "Sound Power", "x": [100.0, 1000.0], "y": [80.5, 81.5]},
                {"name": "Early Reflections DI", "x": [100.0, 1000.0], "y": [1.0, 1.0]},
                {"name": "Sound Power DI", "x": [100.0, 1000.0], "y": [0.5, 0.5]},
            ]
        })
    }

    fn setup_cache() -> TempDir {
        TempDir::new().unwrap()
    }

    #[test]
    fn explicit_cache_roots_are_isolated() {
        let first = setup_cache();
        let second = setup_cache();
        let speaker = "isolated-speaker";
        assert_ne!(
            data_dir_for_cache_root(first.path(), speaker),
            data_dir_for_cache_root(second.path(), speaker)
        );
    }

    #[tokio::test]
    async fn fetch_measurement_plot_data_uses_local_cache() {
        let tmp = setup_cache();
        let speaker = "cache-speaker";
        let measurement = "CEA2034";
        let cache_file =
            data_dir_for_cache_root(tmp.path(), speaker).join(measurement_filename(measurement));
        tokio::fs::create_dir_all(cache_file.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(
            &cache_file,
            serde_json::to_string(&plotly_object()).unwrap(),
        )
        .await
        .unwrap();

        let plot =
            fetch_measurement_plot_data_at_cache_root(speaker, "asr", measurement, tmp.path())
                .await
                .unwrap();
        assert!(plot.get("data").is_some());
    }

    #[tokio::test]
    async fn fetch_curve_from_api_uses_cache() {
        let tmp = setup_cache();
        let speaker = "curve-speaker";
        let cache_file =
            data_dir_for_cache_root(tmp.path(), speaker).join(measurement_filename("CEA2034"));
        tokio::fs::create_dir_all(cache_file.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(
            &cache_file,
            serde_json::to_string(&plotly_object()).unwrap(),
        )
        .await
        .unwrap();

        let curve =
            fetch_curve_from_api_at_cache_root(speaker, "asr", "CEA2034", "On Axis", tmp.path())
                .await
                .unwrap();
        assert_eq!(curve.freq.len(), 2);
    }

    #[tokio::test]
    async fn fetch_directivity_data_uses_cache() {
        let tmp = setup_cache();
        let speaker = "dir-speaker";
        for measurement in ["SPL Horizontal", "SPL Vertical"] {
            let cache_file = data_dir_for_cache_root(tmp.path(), speaker)
                .join(measurement_filename(measurement));
            tokio::fs::create_dir_all(cache_file.parent().unwrap())
                .await
                .unwrap();
            let plot = json!({
                "data": [
                    {"name": "-10°", "x": {"dtype": "f8", "bdata": ""}, "y": {"dtype": "f8", "bdata": ""}},
                ]
            });
            tokio::fs::write(&cache_file, serde_json::to_string(&plot).unwrap())
                .await
                .unwrap();
        }

        let result = fetch_directivity_data_at_cache_root(speaker, "asr", tmp.path()).await;
        // Empty base64 decodes to no curves, so this returns an error about missing curves.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn fetch_contour_data_uses_cache() {
        let tmp = setup_cache();
        let speaker = "contour-speaker";
        let cache_file = data_dir_for_cache_root(tmp.path(), speaker)
            .join(measurement_filename("SPL Horizontal Contour"));
        tokio::fs::create_dir_all(cache_file.parent().unwrap())
            .await
            .unwrap();
        let plot = json!({
            "data": [{
                "x": [100.0, 1000.0],
                "y": [-10.0, 0.0, 10.0],
                "z": [[80.0, 81.0], [82.0, 83.0], [84.0, 85.0]],
            }]
        });
        tokio::fs::write(&cache_file, serde_json::to_string(&plot).unwrap())
            .await
            .unwrap();

        let contour = fetch_contour_data_at_cache_root(speaker, "asr", "horizontal", tmp.path())
            .await
            .unwrap();
        assert_eq!(contour.freq_count, 2);
        assert_eq!(contour.angle_count, 3);
    }

    #[tokio::test]
    async fn fetch_headphone_frequency_response_uses_cache() {
        let tmp = setup_cache();
        let headphone = "test-headphone";
        let cache_dir = headphone_cache_dir_for_cache_root(tmp.path(), headphone);
        tokio::fs::create_dir_all(&cache_dir).await.unwrap();
        let csv = "20,100,20,104\n100,90,100,92\n";
        tokio::fs::write(cache_dir.join("frequency_response_raw.csv"), csv)
            .await
            .unwrap();

        let (path, curve) = fetch_headphone_frequency_response_at_cache_root(headphone, tmp.path())
            .await
            .unwrap();
        assert!(path.contains("measurement.csv"));
        assert_eq!(curve.freq.len(), 2);
        assert!((curve.spl[0] - 102.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn fetch_measurement_plot_data_with_backend_uses_cache_and_backend() {
        let cache_root = std::path::Path::new("isolated-memory-cache");
        let cache = InMemoryMeasurementCache::new();
        let speaker = "mock-speaker";
        let measurement = "CEA2034";
        let cache_file =
            data_dir_for_cache_root(cache_root, speaker).join(measurement_filename(measurement));

        // First call: cache miss -> backend is queried.
        // The backend returns the API array-of-string format.
        let backend = MockMeasurementBackend::new(
            serde_json::to_string(&json!([serde_json::to_string(&plotly_object()).unwrap()]))
                .unwrap(),
        );
        let plot = fetch_measurement_plot_data_with_backend_at_cache_root(
            speaker,
            "asr",
            measurement,
            cache_root,
            &backend,
            &cache,
        )
        .await
        .unwrap();
        assert!(plot.get("data").is_some());
        assert!(
            cache.get(&cache_file).is_some(),
            "plot data should be cached"
        );

        // Second call: cache hit -> backend is not queried (empty response would fail).
        let silent_backend = MockMeasurementBackend::new("not-json");
        let plot2 = fetch_measurement_plot_data_with_backend_at_cache_root(
            speaker,
            "asr",
            measurement,
            cache_root,
            &silent_backend,
            &cache,
        )
        .await
        .unwrap();
        assert!(plot2.get("data").is_some());
    }
}
