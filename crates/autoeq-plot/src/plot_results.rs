use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use autoeq_report_wasm::{ReportPayload, Section};

use crate::plot_drivers::write_report;
use crate::plot_filters::plot_filter_figures;
use crate::plot_spin::{plot_spin, plot_spin_details, plot_spin_tonal};
use crate::x2peq::compute_peq_response_from_x;

/// Build the report payload (filter figures plus optional spinorama figures).
pub fn plot_compute(
    config: &crate::PlotConfig,
    optimized_params: &[f64],
    input_curve: &crate::Curve,
    target_curve: &crate::Curve,
    deviation_curve: &crate::Curve,
    cea2034_curves: &Option<HashMap<String, crate::Curve>>,
) -> ReportPayload {
    let freqs = input_curve.freq.clone();
    let speaker = config.speaker_name.as_deref();
    let title_text = match speaker {
        Some(s) if !s.is_empty() => format!("{} -- #{} peq(s)", s, config.num_filters),
        _ => "IIR Filter Optimization Results".to_string(),
    };
    let mut payload = ReportPayload::new(&title_text);

    for fig in plot_filter_figures(
        config,
        input_curve,
        target_curve,
        deviation_curve,
        optimized_params,
    ) {
        payload.sections.push(Section::Figure {
            figure: fig,
            tab: None,
        });
    }

    if cea2034_curves.is_some() {
        let eq_response = compute_peq_response_from_x(
            &freqs,
            optimized_params,
            config.sample_rate,
            config.peq_model,
        );
        for fig in plot_spin_details(cea2034_curves.as_ref(), Some(&eq_response)) {
            payload.sections.push(Section::Figure {
                figure: fig,
                tab: None,
            });
        }
        for fig in plot_spin_tonal(cea2034_curves.as_ref(), Some(&eq_response)) {
            payload.sections.push(Section::Figure {
                figure: fig,
                tab: None,
            });
        }
        for fig in plot_spin(cea2034_curves.as_ref(), Some(&eq_response)) {
            payload.sections.push(Section::Figure {
                figure: fig,
                tab: None,
            });
        }
    }
    payload
}

/// Generate and save an HTML+WASM report comparing input, target, and EQ.
///
/// # Arguments
/// * `config` - Plot configuration
/// * `optimized_params` - Optimized filter parameters
/// * `input_curve` - Original frequency response
/// * `target_curve` - Target curve
/// * `deviation_curve` - Deviation from target
/// * `cea2034_curves` - Optional CEA2034 curves
/// * `output_path` - Base path for the HTML report (extension forced to html)
pub fn plot_results(
    config: &crate::PlotConfig,
    optimized_params: &[f64],
    input_curve: &crate::Curve,
    target_curve: &crate::Curve,
    deviation_curve: &crate::Curve,
    cea2034_curves: &Option<HashMap<String, crate::Curve>>,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let payload = plot_compute(
        config,
        optimized_params,
        input_curve,
        target_curve,
        deviation_curve,
        cea2034_curves,
    );
    let title = payload.title.clone();
    write_report(&title, &payload, output_path)
}
