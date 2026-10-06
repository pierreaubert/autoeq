use autoeq_report_wasm::{AxisSpec, DashOption, Figure, RangeMark, Series, XScale};
use ndarray::Array1;

use crate::filter_color::filter_color;
use crate::iir::{Biquad, BiquadFilterType};
use crate::param_utils::{determine_filter_type, get_filter_params, num_filters};
use crate::ref_lines::make_ref_series;
use crate::response::MIN_FILTER_RESPONSE_DB;

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

/// Shaded green ranges outside the optimization bounds (all four panels).
fn freq_range_marks(min_freq: f64, max_freq: f64) -> Vec<RangeMark> {
    let mut marks = Vec::new();
    if min_freq > 20.0 {
        marks.push(RangeMark {
            x0: 20.0,
            x1: min_freq,
            color: "rgba(144, 238, 144, 0.3)".to_string(),
        });
    }
    if max_freq < 20000.0 {
        marks.push(RangeMark {
            x0: max_freq,
            x1: 20000.0,
            color: "rgba(144, 238, 144, 0.3)".to_string(),
        });
    }
    marks
}

fn series(name: String, x: Vec<f64>, y: Vec<f64>, color: &str, width: f32) -> Series {
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

/// Create the four filter-analysis figures (previously one 2x2 Plotly grid).
///
/// Panel order and content are unchanged: individual filters + sum,
/// EQ vs deviation, zoomed error, response with EQ. Each panel becomes one
/// schema figure with the same series, colors, widths, and y ranges.
pub fn plot_filter_figures(
    config: &crate::PlotConfig,
    input_curve: &crate::Curve,
    target_curve: &crate::Curve,
    deviation_curve: &crate::Curve,
    optimized_params: &[f64],
) -> Vec<Figure> {
    let freqs = input_curve.freq.clone();
    let peq_model = config.peq_model;
    let filter_count = config
        .num_filters
        .min(num_filters(optimized_params, peq_model));
    let mut filters: Vec<(usize, BiquadFilterType, f64, f64, f64)> = (0..filter_count)
        .map(|i| {
            let params = get_filter_params(optimized_params, i, peq_model);
            (
                i,
                determine_filter_type(i, filter_count, peq_model, params.filter_type),
                10f64.powf(params.freq),
                params.q,
                params.gain,
            )
        })
        .collect();
    filters.sort_by(|a, b| a.2.total_cmp(&b.2));

    let mut individual: Vec<Series> = Vec::new();
    let mut combined_response: Array1<f64> = Array1::zeros(freqs.len());
    for (display_idx, (orig_i, ftype, f0, q, gain)) in filters.iter().enumerate() {
        let filter = Biquad::new(*ftype, *f0, config.sample_rate, *q, *gain);
        let filter_response = filter
            .np_log_result(&freqs)
            .mapv(|db| db.max(MIN_FILTER_RESPONSE_DB));
        combined_response += &filter_response;

        let label = match *ftype {
            BiquadFilterType::Highpass | BiquadFilterType::HighpassVariableQ => "HPQ",
            BiquadFilterType::Lowpass => "LP",
            BiquadFilterType::Lowshelf | BiquadFilterType::LowshelfOrf => "LS",
            BiquadFilterType::Highshelf | BiquadFilterType::HighshelfOrf => "HS",
            BiquadFilterType::Bandpass => "BP",
            BiquadFilterType::Notch => "NO",
            _ => "PK",
        };
        individual.push(series(
            format!("{} {} at {:5.0}Hz", label, orig_i + 1, f0),
            freqs.to_vec(),
            filter_response.to_vec(),
            filter_color(display_idx),
            1.0,
        ));
    }

    let combined = combined_response.to_vec();
    let auto_eq = series(
        "autoEQ".to_string(),
        freqs.to_vec(),
        combined.clone(),
        "#000000",
        2.0,
    );
    let deviation = series(
        "Deviation".to_string(),
        freqs.to_vec(),
        deviation_curve.spl.to_vec(),
        filter_color(0),
        2.0,
    );
    let error_vals = (&deviation_curve.spl - &combined_response).to_vec();
    let error = series(
        "Error".to_string(),
        freqs.to_vec(),
        error_vals,
        filter_color(1),
        2.0,
    );
    let target = series(
        "Target".to_string(),
        freqs.to_vec(),
        target_curve.spl.to_vec(),
        filter_color(0),
        2.0,
    );
    let input = series(
        "Input".to_string(),
        input_curve.freq.to_vec(),
        input_curve.spl.to_vec(),
        filter_color(1),
        2.0,
    );
    let input_plus_eq = series(
        "Input + EQ".to_string(),
        input_curve.freq.to_vec(),
        (&input_curve.spl + &combined_response).to_vec(),
        filter_color(2),
        3.0,
    );

    let marks = freq_range_marks(config.min_freq, config.max_freq);
    let ref_series = make_ref_series();

    let mut panel_filters = individual;
    panel_filters.push(auto_eq.clone());
    let mut panel_error = vec![error];
    panel_error.extend(ref_series);

    vec![
        Figure {
            title: "IIR filters and Sum of filters".to_string(),
            x: log_freq_axis(),
            y: spl_axis(-10.0, 10.0),
            y2: None,
            series: panel_filters,
            hlines: vec![],
            vlines: vec![],
            xranges: marks.clone(),
            annotations: vec![],
            legend: true,
        },
        Figure {
            title: "Autoeq v.s. Deviation from target".to_string(),
            x: log_freq_axis(),
            y: spl_axis(-10.0, 10.0),
            y2: None,
            series: vec![auto_eq, deviation],
            hlines: vec![],
            vlines: vec![],
            xranges: marks.clone(),
            annotations: vec![],
            legend: true,
        },
        Figure {
            title: "Error = Autoeq-Deviation (zoomed)".to_string(),
            x: log_freq_axis(),
            y: spl_axis(-5.0, 5.0),
            y2: None,
            series: panel_error,
            hlines: vec![],
            vlines: vec![],
            xranges: marks.clone(),
            annotations: vec![],
            legend: true,
        },
        Figure {
            title: "Response w/ autoEQ".to_string(),
            x: log_freq_axis(),
            y: spl_axis(-10.0, 10.0),
            y2: None,
            series: vec![target, input, input_plus_eq],
            hlines: vec![],
            vlines: vec![],
            xranges: marks,
            annotations: vec![],
            legend: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlotConfig;
    use crate::param_utils::encode_filter_type;
    use crate::{Curve, PeqModel};

    #[test]
    fn audit_plot_filters_decodes_free_model_layout() {
        let curve = Curve {
            freq: Array1::from_vec(vec![500.0, 1000.0, 2000.0]),
            spl: Array1::zeros(3),
            phase: None,
            ..Default::default()
        };
        let config = PlotConfig {
            speaker_name: None,
            num_filters: 1,
            sample_rate: 48_000.0,
            peq_model: PeqModel::Free,
            min_freq: 20.0,
            max_freq: 20_000.0,
        };
        let params = vec![
            encode_filter_type(BiquadFilterType::LowshelfOrf),
            1000.0_f64.log10(),
            0.7,
            3.0,
        ];
        let figs = plot_filter_figures(&config, &curve, &curve, &curve, &params);
        assert_eq!(figs.len(), 4);
        let first = &figs[0].series[0];
        assert!(
            first.name.starts_with("LS 1 at"),
            "unexpected series name: {}",
            first.name
        );
        assert!(
            first.name.ends_with("1000Hz"),
            "unexpected series name: {}",
            first.name
        );
        assert_eq!(figs[0].title, "IIR filters and Sum of filters");
        assert!(figs[1].series.iter().any(|s| s.name == "autoEQ"));
        assert!(figs[1].series.iter().any(|s| s.name == "Deviation"));
        assert!(figs[2].series.iter().any(|s| s.name == "Error"));
        assert!(figs[3].series.iter().any(|s| s.name == "Input + EQ"));
    }
}
