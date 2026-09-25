use autoeq_report_wasm::{
    Annotation, AxisSpec, DashOption, Figure, LineMark, Series, XScale,
};

use crate::loss::{DriversLossData, compute_drivers_combined_response};

fn driver_color(i: usize) -> &'static str {
    match i {
        0 => "rgb(31, 119, 180)",  // Blue (woofer)
        1 => "rgb(255, 127, 14)",  // Orange (tweeter)
        2 => "rgb(44, 160, 44)",   // Green (midrange)
        3 => "rgb(214, 39, 40)",   // Red (super tweeter)
        _ => "rgb(128, 128, 128)", // Gray (fallback)
    }
}

/// Create the multi-driver crossover figure (same traces as before).
///
/// # Arguments
/// * `drivers_data` - Multi-driver measurement data
/// * `gains` - Optimized gain values for each driver (in dB)
/// * `crossover_freqs` - Optimized crossover frequencies (in Hz)
/// * `delays` - Optional per-driver delays
/// * `sample_rate` - Sample rate for filter design
pub fn plot_driver_figures(
    drivers_data: &DriversLossData,
    gains: &[f64],
    crossover_freqs: &[f64],
    delays: Option<&[f64]>,
    sample_rate: f64,
) -> Vec<Figure> {
    let mut all_series: Vec<Series> = Vec::new();
    let freq_grid = &drivers_data.freq_grid;

    // First, compute the combined response to get a reference normalization
    let combined_response = compute_drivers_combined_response(
        drivers_data,
        gains,
        crossover_freqs,
        delays,
        sample_rate,
    );
    let combined_mean = combined_response.mean().unwrap_or(0.0);

    // Individual drivers (raw responses, dashed)
    for (i, driver) in drivers_data.drivers.iter().enumerate() {
        let interpolated = crate::read::normalize_and_interpolate_response(
            freq_grid,
            &crate::Curve {
                freq: driver.freq.clone(),
                spl: driver.spl.clone(),
                phase: driver.phase.clone(),
                ..Default::default()
            },
        );
        all_series.push(Series {
            name: format!("Driver {} (raw)", i + 1),
            x: freq_grid.to_vec(),
            y: interpolated.spl.iter().map(|&v| Some(v)).collect(),
            color: Some(driver_color(i).to_string()),
            width: 1.5,
            dash: DashOption::Dash,
            visible: true,
            y_axis: 0,
        });
    }

    // Individual drivers with gains and crossovers applied
    for (i, driver) in drivers_data.drivers.iter().enumerate() {
        let driver_freq_grid = &driver.freq;
        let mut response = &driver.spl + gains[i];

        if let crate::loss::CrossoverType::None = drivers_data.crossover_type {
            // No filters
        } else {
            if i > 0 {
                let xover_freq = crossover_freqs[i - 1];
                let hp_filter = match drivers_data.crossover_type {
                    crate::loss::CrossoverType::Butterworth2 => {
                        crate::iir::peq_butterworth_highpass(2, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::Butterworth4 => {
                        crate::iir::peq_butterworth_highpass(4, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley2 => {
                        crate::iir::peq_linkwitzriley_highpass(2, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley4 => {
                        crate::iir::peq_linkwitzriley_highpass(4, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley8 => {
                        crate::iir::peq_linkwitzriley_highpass(8, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinearPhase => vec![],
                    crate::loss::CrossoverType::None => vec![],
                };
                let hp_response = if matches!(
                    drivers_data.crossover_type,
                    crate::loss::CrossoverType::LinearPhase
                ) {
                    let crossover = math_audio_iir_fir::FirCrossover::new(
                        xover_freq,
                        sample_rate,
                        1,
                        math_audio_iir_fir::DEFAULT_FIR_CROSSOVER_TAPS,
                    );
                    ndarray::Array1::from_iter(
                        crate::response::compute_fir_complex_response(
                            &crossover.highpass_coefficients(),
                            driver_freq_grid,
                            sample_rate,
                        )
                        .into_iter()
                        .map(|z| 20.0 * z.norm().max(1e-12).log10()),
                    )
                } else {
                    crate::iir::compute_peq_response(driver_freq_grid, &hp_filter, sample_rate)
                };
                response = response + hp_response;
            }

            if i < drivers_data.drivers.len() - 1 {
                let xover_freq = crossover_freqs[i];
                let lp_filter = match drivers_data.crossover_type {
                    crate::loss::CrossoverType::Butterworth2 => {
                        crate::iir::peq_butterworth_lowpass(2, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::Butterworth4 => {
                        crate::iir::peq_butterworth_lowpass(4, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley2 => {
                        crate::iir::peq_linkwitzriley_lowpass(2, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley4 => {
                        crate::iir::peq_linkwitzriley_lowpass(4, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinkwitzRiley8 => {
                        crate::iir::peq_linkwitzriley_lowpass(8, xover_freq, sample_rate)
                    }
                    crate::loss::CrossoverType::LinearPhase => vec![],
                    crate::loss::CrossoverType::None => vec![],
                };
                let lp_response = if matches!(
                    drivers_data.crossover_type,
                    crate::loss::CrossoverType::LinearPhase
                ) {
                    let crossover = math_audio_iir_fir::FirCrossover::new(
                        xover_freq,
                        sample_rate,
                        1,
                        math_audio_iir_fir::DEFAULT_FIR_CROSSOVER_TAPS,
                    );
                    ndarray::Array1::from_iter(
                        crate::response::compute_fir_complex_response(
                            crossover.lowpass_coefficients(),
                            driver_freq_grid,
                            sample_rate,
                        )
                        .into_iter()
                        .map(|z| 20.0 * z.norm().max(1e-12).log10()),
                    )
                } else {
                    crate::iir::compute_peq_response(driver_freq_grid, &lp_filter, sample_rate)
                };
                response = response + lp_response;
            }
        }

        all_series.push(Series {
            name: format!("Driver {} ({:+.1} dB)", i + 1, gains[i]),
            x: driver_freq_grid.to_vec(),
            y: response.iter().map(|&v| Some(v)).collect(),
            color: Some(driver_color(i).to_string()),
            width: 2.0,
            dash: DashOption::Solid,
            visible: true,
            y_axis: 0,
        });
    }

    // Combined response, mean-normalized, black.
    let combined_response_normalized = &combined_response - combined_mean;
    all_series.push(Series {
        name: "Combined Response".to_string(),
        x: freq_grid.to_vec(),
        y: combined_response_normalized
            .iter()
            .map(|&v| Some(v))
            .collect(),
        color: Some("rgb(0, 0, 0)".to_string()),
        width: 3.0,
        dash: DashOption::Solid,
        visible: true,
        y_axis: 0,
    });

    // Crossover markers: dotted vertical lines plus labels near the top.
    let mut vlines = Vec::new();
    let mut annotations = Vec::new();
    for (i, &xover_freq) in crossover_freqs.iter().enumerate() {
        vlines.push(LineMark {
            at: xover_freq,
            color: "rgba(150, 150, 150, 0.6)".to_string(),
            dash: DashOption::Dot,
            width: 2.0,
            label: None,
        });
        annotations.push(Annotation {
            x: xover_freq,
            y: 28.0,
            text: format!("Crossover {}: {:.0} Hz", i + 1, xover_freq),
        });
    }

    let crossover_type_str = drivers_data.crossover_type.display_name();
    vec![Figure {
        title: format!("Multi-Driver Crossover Optimization ({crossover_type_str})"),
        x: AxisSpec {
            label: "Frequency (Hz)".to_string(),
            scale: XScale::Log,
            min: Some(20.0),
            max: Some(20000.0),
        },
        y: AxisSpec {
            label: "SPL (dB)".to_string(),
            scale: XScale::Linear,
            min: Some(-30.0),
            max: Some(30.0),
        },
        y2: None,
        series: all_series,
        hlines: vec![],
        vlines,
        xranges: vec![],
        annotations,
        legend: true,
    }]
}

/// Generate and save an HTML+WASM report for multi-driver results.
///
/// # Arguments
/// * `drivers_data` - Multi-driver measurement data
/// * `gains` - Optimized gain values for each driver (in dB)
/// * `crossover_freqs` - Optimized crossover frequencies (in Hz)
/// * `delays` - Optional per-driver delays
/// * `sample_rate` - Sample rate for filter design
/// * `output_path` - Base path for the HTML report (extension forced to html)
pub fn plot_drivers_results(
    drivers_data: &DriversLossData,
    gains: &[f64],
    crossover_freqs: &[f64],
    delays: Option<&[f64]>,
    sample_rate: f64,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let figures = plot_driver_figures(drivers_data, gains, crossover_freqs, delays, sample_rate);
    let title_text = format!(
        "{}-Way Speaker Crossover Optimization",
        drivers_data.drivers.len()
    );
    let mut payload = autoeq_report_wasm::ReportPayload::new(&title_text);
    for fig in figures {
        payload.sections.push(autoeq_report_wasm::Section::Figure {
            figure: fig,
            tab: None,
        });
    }
    write_report(&title_text, &payload, output_path)
}

/// Assemble a payload into a self-contained HTML file.
pub(crate) fn write_report(
    title: &str,
    payload: &autoeq_report_wasm::ReportPayload,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;
    let payload_json = serde_json::to_string(payload)?;
    let assets = autoeq_report_wasm::assemble::checked_in_assets().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("report shell assets missing (rebuild with just report-wasm): {e}"),
        )
    })?;
    let html = autoeq_report_wasm::assemble::assemble_html(title, &payload_json, &assets);
    let html_output_path = output_path.with_extension("html");
    if let Some(parent) = html_output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(&html_output_path)?;
    file.write_all(html.as_bytes())?;
    file.flush()?;
    Ok(())
}
