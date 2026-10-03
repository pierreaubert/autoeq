use autoeq::Curve;
use autoeq::OptimParams;
use autoeq::read;
use autoeq_workflow::workflow::{
    PreparedProduct, ProductCurves, ProductRequest, ProductSourceAdapters,
};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;

/// Load input data and prepare frequency grid and curves
pub(super) async fn load_and_prepare(
    args: &autoeq::cli::Args,
) -> Result<
    (
        ndarray::Array1<f64>,
        Curve,
        Curve,
        Curve,
        Option<HashMap<String, Curve>>,
    ),
    Box<dyn std::error::Error>,
> {
    // Load input data
    let (input_curve_raw, spin_data_raw) =
        autoeq::workflow::load_input_curve(&autoeq::workflow::InputConfig::from(args)).await?;

    // Determine if this is headphone or speaker optimization
    let is_headphone = matches!(
        args.loss,
        autoeq::LossType::HeadphoneFlat | autoeq::LossType::HeadphoneScore
    );

    // Determine if we can use the original frequency grid from CEA2034 data
    // to avoid unnecessary resampling while maintaining accuracy
    let use_original_freq = if let Some(ref spin_data) = spin_data_raw {
        // Check if all curves have the same frequency grid as input_curve_raw
        let input_freq = &input_curve_raw.freq;
        spin_data.iter().all(|(_, curve)| {
            curve.freq.len() == input_freq.len()
                && curve
                    .freq
                    .iter()
                    .zip(input_freq.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-9) // Allow tiny numerical differences
        })
    } else {
        false
    };

    // Use original frequency grid from API data if available and consistent,
    // otherwise create a standard log-spaced grid
    let standard_freq = if use_original_freq {
        input_curve_raw.freq.clone()
    } else {
        let num_points = if is_headphone { 120 } else { 200 };
        read::create_log_frequency_grid(num_points, 20.0, 20000.0)
    };

    // Interpolate input curve first (needed for deviation calculation)
    let input_curve = read::normalize_and_interpolate_response(&standard_freq, &input_curve_raw);

    // Build target curve in parallel with spinorama interpolation
    let (target_res, spin_data_res) = tokio::join!(
        async {
            let target_raw = autoeq::workflow::build_target_curve(
                &autoeq::workflow::TargetConfig::from(args),
                &standard_freq,
                &input_curve_raw,
            )?;
            Ok::<_, Box<dyn std::error::Error>>(read::interpolate_log_space(
                &standard_freq,
                &target_raw,
            ))
        },
        async {
            // Interpolate spinorama data if available
            match spin_data_raw {
                Some(spin_data) => {
                    let interpolated: HashMap<String, Curve> = spin_data
                        .into_iter()
                        .map(|(name, curve)| {
                            let interp = read::interpolate_log_space(&standard_freq, &curve);
                            (name, interp)
                        })
                        .collect();
                    Ok::<_, Box<dyn std::error::Error>>(Some(interpolated))
                }
                None => Ok::<_, Box<dyn std::error::Error>>(None),
            }
        }
    );

    // Unpack results and compute deviation
    let target_curve = target_res?;
    let deviation_curve = Curve {
        freq: standard_freq.clone(),
        spl: &target_curve.spl - &input_curve.spl,
        phase: None,
        ..Default::default()
    };
    let spin_data = spin_data_res?;

    Ok((
        standard_freq,
        input_curve,
        target_curve,
        deviation_curve,
        spin_data,
    ))
}

pub(super) struct LoadedProductInput {
    pub(super) request: ProductRequest,
    pub(super) prepared: PreparedProduct,
    pub(super) curves: ProductCurves,
}

/// Load the product manifest through injectable measurement/cache adapters,
/// then prepare source and target curves on one optimizer grid.
pub(super) async fn load_product_config(
    path: &Path,
    params: &OptimParams,
) -> Result<LoadedProductInput, Box<dyn std::error::Error>> {
    const MAX_PRODUCT_CONFIG_BYTES: u64 = 1024 * 1024;
    let request_bytes = read_product_config_bytes(fs::File::open(path)?, MAX_PRODUCT_CONFIG_BYTES)?;
    let request: ProductRequest = serde_json::from_slice(&request_bytes)?;
    let cache_root = read::cache_root();
    let backend = read::ReqwestMeasurementBackend::new();
    let cache = read::FsMeasurementCache::new();
    let adapters = ProductSourceAdapters {
        cache_root: &cache_root,
        backend: &backend,
        cache: &cache,
    };
    let prepared = PreparedProduct::load(&request, params, &adapters).await?;
    let curves = prepared.curves_for_optimizer(params)?;
    Ok(LoadedProductInput {
        request,
        prepared,
        curves,
    })
}

fn read_product_config_bytes<R: Read>(
    reader: R,
    maximum_bytes: u64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    reader
        .take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(
            format!("product configuration exceeds the {maximum_bytes}-byte safety limit").into(),
        );
    }
    Ok(bytes)
}

#[cfg(test)]
mod product_config_read_tests {
    use super::read_product_config_bytes;
    use std::io::Cursor;

    #[test]
    fn bounded_product_config_reader_accepts_limit_and_rejects_one_more_byte() {
        assert_eq!(
            read_product_config_bytes(Cursor::new(b"12345678"), 8).unwrap(),
            b"12345678"
        );
        assert!(
            read_product_config_bytes(Cursor::new(b"123456789"), 8)
                .unwrap_err()
                .to_string()
                .contains("8-byte safety limit")
        );
    }
}
