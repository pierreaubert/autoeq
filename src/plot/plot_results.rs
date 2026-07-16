use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use build_html::*;
use plotly::Plot;

use crate::plot::plot_filters::plot_filters;
use crate::plot::plot_spin::{plot_spin, plot_spin_details, plot_spin_tonal};
use crate::x2peq::compute_peq_response_from_x;

pub async fn plot_compute(
    config: &crate::plot::PlotConfig,
    optimized_params: &[f64],
    input_curve: &crate::Curve,
    target_curve: &crate::Curve,
    deviation_curve: &crate::Curve,
    cea2034_curves: &Option<HashMap<String, crate::Curve>>,
) -> (Plot, Option<Plot>, Option<Plot>, Option<Plot>) {
    let freqs = input_curve.freq.clone();

    // gather all subplots
    let plot_filters = plot_filters(
        config,
        input_curve,
        target_curve,
        deviation_curve,
        optimized_params,
    );

    let eq_response = compute_peq_response_from_x(
        &freqs,
        optimized_params,
        config.sample_rate,
        config.peq_model,
    );
    let plot_spin_details = if cea2034_curves.is_some() {
        Some(plot_spin_details(
            cea2034_curves.as_ref(),
            Some(&eq_response),
        ))
    } else {
        None
    };

    let plot_spin_tonal = if cea2034_curves.is_some() {
        Some(plot_spin_tonal(cea2034_curves.as_ref(), Some(&eq_response)))
    } else {
        None
    };

    let plot_spin_opt = if cea2034_curves.is_some() {
        Some(plot_spin(cea2034_curves.as_ref(), Some(&eq_response)))
    } else {
        None
    };

    (
        plot_filters,
        plot_spin_details,
        plot_spin_tonal,
        plot_spin_opt,
    )
}

/// Generate and save an HTML plot comparing the input curve with the optimized EQ response.
///
/// # Arguments
/// * `args` - The list of args from the command line
/// * `input_curve` - The original frequency response curve
/// * `smoothed_curve` - Optional smoothed inverted target curve
/// * `target_curve` - The target curve
/// * `optimized_params` - The optimized filter parameters
/// * `output_path` - The path to save the HTML output file
/// * `cea2034_curves` - Optional CEA2034 curves to include in the plot
/// * `eq_response` - Optional EQ response to include in the plot
///
/// # Returns
/// * Result indicating success or failure
pub async fn plot_results(
    config: &crate::plot::PlotConfig,
    optimized_params: &[f64],
    input_curve: &crate::Curve,
    target_curve: &crate::Curve,
    deviation_curve: &crate::Curve,
    cea2034_curves: &Option<HashMap<String, crate::Curve>>,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let speaker = config.speaker_name.as_deref();
    let (plot_filters, plot_spin_details, plot_spin_tonal, plot_spin_opt) = plot_compute(
        config,
        optimized_params,
        input_curve,
        target_curve,
        deviation_curve,
        cea2034_curves,
    )
    .await;

    // Title with optional speaker name
    let title_text = match speaker {
        Some(s) if !s.is_empty() => format!("{} -- #{} peq(s)", s, config.num_filters),
        _ => "IIR Filter Optimization Results".to_string(),
    };

    let html: String = {
        let base = HtmlPage::new()
            .with_title(title_text)
            .with_script_link("https://cdn.plot.ly/plotly-3.2.0.min.js")
            .with_raw(plot_filters.to_inline_html(Some("filters")));
        let page = if let Some(ref plot_spin) = plot_spin_details {
            base.with_raw(plot_spin.to_inline_html(Some("details")))
        } else {
            base
        };
        let page2 = if let Some(ref plot_spin) = plot_spin_tonal {
            page.with_raw(plot_spin.to_inline_html(Some("tonal")))
        } else {
            page
        };
        let page3 = if let Some(ref plot_spin) = plot_spin_opt {
            page2.with_raw(plot_spin.to_inline_html(Some("spinorama")))
        } else {
            page2
        };
        page3.to_html_string()
    };

    // Ensure parent directory exists before writing files
    let html_output_path = output_path.with_extension("html");
    if let Some(parent) = html_output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(&html_output_path)?;
    file.write_all(html.as_bytes())?;
    file.flush()?;

    // plot_spin.write_html(output_path.with_extension("html"));

    #[cfg(feature = "plotly_static")]
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    #[cfg(feature = "plotly_static")]
    let mut plots: Vec<(Plot, &str, usize, usize)> =
        vec![(plot_filters, "filters", 1280usize, 800usize)];
    #[cfg(feature = "plotly_static")]
    if let Some(plot_spin) = plot_spin_details {
        plots.push((plot_spin, "details", 1280, 650));
    }

    #[cfg(feature = "plotly_static")]
    if let Some(plot_spin) = plot_spin_tonal {
        plots.push((plot_spin, "tonal", 1280, 650));
    }

    #[cfg(feature = "plotly_static")]
    if let Some(plot_spin) = plot_spin_opt {
        plots.push((plot_spin, "spinorama", 1280, 450));
    }

    // WebDriver-free PNG export via deterministic SVG + resvg.
    #[cfg(feature = "plotly_static")]
    {
        for (plot, name, width, height) in plots {
            let img_path = output_path.with_file_name(format!("{}-{}.png", stem, name));
            if let Some(parent) = img_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            crate::plot::static_export::write_plot_png(&plot, &img_path, width, height)?;
        }
    }

    #[cfg(not(feature = "plotly_static"))]
    {
        eprintln!(
            "ℹ️ Info: PNG export disabled. Enable with --features plotly_static. HTML report was generated at {}",
            html_output_path.display()
        );
    }

    Ok(())
}
