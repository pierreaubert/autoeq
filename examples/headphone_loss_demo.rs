//! CLI tool for computing headphone loss from frequency response files
//!
//! This tool computes the headphone preference loss score based on the model from
//! 'A Statistical Model that Predicts Listeners' Preference Ratings of In-Ear Headphones'
//! by Sean Olive et al. Lower scores indicate better predicted preference.
//!
//! Output plots are automatically saved to data_generated/headphone_loss_plots.html
//!
//! Usage:
//!   cargo run --example headphone_loss_demo -- --spl <file> --target <file> [--smooth] [--smooth-n <n>]

use autoeq::Curve;
use autoeq::loss::headphone_loss;
use autoeq::read::{
    create_log_frequency_grid, normalize_and_interpolate_response, read_curve_from_csv,
    smooth_one_over_n_octave,
};
use autoeq_report_wasm::{AxisSpec, DashOption, Figure, ReportPayload, Section, Series, XScale};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "headphone_loss_demo",
    about = "Compute headphone preference score from frequency response measurements",
    long_about = "Computes the headphone preference loss score based on the model from \n'A Statistical Model that Predicts Listeners' Preference Ratings of In-Ear Headphones' \nby Sean Olive et al. Lower scores indicate better predicted preference."
)]
struct Args {
    /// Path to SPL (frequency response) file (CSV or text with freq,spl columns)
    #[arg(long)]
    spl: PathBuf,

    /// Path to target frequency response file (CSV or text with freq,spl columns)
    #[arg(long)]
    target: PathBuf,

    /// Enable smoothing (regularization) of the inverted target curve
    #[arg(long, default_value_t = true)]
    pub smooth: bool,

    /// Smoothing level as 1/N octave (N in [1..24]). Example: N=6 => 1/6 octave smoothing
    #[arg(long, default_value_t = 2)]
    pub smooth_n: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // freqs on which we normalize every curve: 12 points per octave between 20 and 20kHz
    let freqs = create_log_frequency_grid(10 * 12, 20.0, 20000.0);

    // Load SPL data
    println!("Loading SPL data from: {:?}", args.spl);
    let input_curve_raw = read_curve_from_csv(&args.spl)?;
    println!(
        "  Loaded headphone response #{} data points from {:.1} Hz to {:.1} Hz SPL from {:.1} to {:.1} dB",
        input_curve_raw.freq.len(),
        input_curve_raw.freq[0],
        input_curve_raw.freq[input_curve_raw.freq.len() - 1],
        input_curve_raw.spl.fold(f64::INFINITY, |a, &b| a.min(b)),
        input_curve_raw
            .spl
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    );

    let input_curve = normalize_and_interpolate_response(&freqs, &input_curve_raw);
    println!(
        " Normalized headphone response #{} data points from {:.1} Hz to {:.1} Hz SPL from {:.1} to {:.1} dB",
        input_curve.freq.len(),
        input_curve.freq[0],
        input_curve.freq[input_curve.freq.len() - 1],
        input_curve.spl.fold(f64::INFINITY, |a, &b| a.min(b)),
        input_curve.spl.fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    );

    let target_curve_raw = read_curve_from_csv(&args.target)?;
    println!(
        "  Loaded {} targets from {:.1} Hz to {:.1} Hz SPL from {:.1} to {:.1} dB",
        target_curve_raw.freq.len(),
        target_curve_raw.freq[0],
        target_curve_raw.freq[target_curve_raw.freq.len() - 1],
        target_curve_raw.spl.fold(f64::INFINITY, |a, &b| a.min(b)),
        target_curve_raw
            .spl
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
    );

    let target_curve = normalize_and_interpolate_response(&freqs, &target_curve_raw);

    // compute deviation and potentially smooth it
    let deviation_spl = &target_curve.spl - &input_curve.spl;
    let deviation = Curve {
        freq: freqs.clone(),
        spl: deviation_spl,
        phase: None,
        ..Default::default()
    };
    let smooth_deviation = if args.smooth {
        smooth_one_over_n_octave(&deviation, args.smooth_n)
    } else {
        deviation.clone()
    };

    // Compute headphone loss and create plots
    let score = headphone_loss(&smooth_deviation);

    // Print results
    println!("\n{}", "=".repeat(50));
    println!("Headphone Loss Score: {:.3}", -score);
    println!("{}", "=".repeat(50));

    // Create output path in data_generated directory
    let data_generated_dir = std::path::PathBuf::from("data_generated");
    std::fs::create_dir_all(&data_generated_dir)?;
    let output_path = data_generated_dir.join("headphone_loss_plots.html");

    // Generate plots
    generate_plots(
        &input_curve,
        &target_curve,
        &deviation,
        &smooth_deviation,
        &output_path,
    )?;

    Ok(())
}

/// Generate plots for the input curve and target curve (if provided)
/// and their normalized versions
fn generate_plots(
    input_curve: &Curve,
    target_curve: &Curve,
    deviation: &Curve,
    smooth_deviation: &Curve,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    fn line(name: &str, x: Vec<f64>, y: Vec<f64>, color: &str) -> Series {
        Series {
            name: name.to_string(),
            x,
            y: y.into_iter().map(Some).collect(),
            color: Some(color.to_string()),
            width: 2.0,
            dash: DashOption::Solid,
            visible: true,
            y_axis: 0,
        }
    }
    fn freq_axis() -> AxisSpec {
        AxisSpec {
            label: "Frequency (Hz)".to_string(),
            scale: XScale::Log,
            min: Some(20.0),
            max: Some(20000.0),
        }
    }

    let fig1 = Figure {
        title: "Input Curve vs Target Curve".to_string(),
        x: freq_axis(),
        y: AxisSpec {
            label: "SPL (dB)".to_string(),
            scale: XScale::Linear,
            min: Some(-10.0),
            max: Some(10.0),
        },
        y2: None,
        series: vec![
            line(
                "Input Curve",
                input_curve.freq.to_vec(),
                input_curve.spl.to_vec(),
                "#1f77b4",
            ),
            line(
                "Harmann Target Curve",
                target_curve.freq.to_vec(),
                target_curve.spl.to_vec(),
                "#ff7f0e",
            ),
        ],
        hlines: vec![],
        vlines: vec![],
        xranges: vec![],
        annotations: vec![],
        legend: true,
    };
    let fig2 = Figure {
        title: "Normalized Curves".to_string(),
        x: freq_axis(),
        y: AxisSpec {
            label: "Normalized SPL (dB)".to_string(),
            scale: XScale::Linear,
            min: Some(-10.0),
            max: Some(10.0),
        },
        y2: None,
        series: vec![
            line(
                "Normalized Deviation",
                deviation.freq.to_vec(),
                deviation.spl.to_vec(),
                "#1f77b4",
            ),
            line(
                "Smooth Normalized Deviation",
                smooth_deviation.freq.to_vec(),
                smooth_deviation.spl.to_vec(),
                "#ff7f0e",
            ),
        ],
        hlines: vec![],
        vlines: vec![],
        xranges: vec![],
        annotations: vec![],
        legend: true,
    };

    let mut payload = ReportPayload::new("Headphone Loss Analysis Plots");
    payload.sections.push(Section::Figure {
        figure: fig1,
        tab: None,
    });
    payload.sections.push(Section::Figure {
        figure: fig2,
        tab: None,
    });
    let payload_json = serde_json::to_string(&payload)?;
    let assets = autoeq_report_wasm::assemble::checked_in_assets().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("report shell assets missing (rebuild with just report-wasm): {e}"),
        )
    })?;
    let html_content =
        autoeq_report_wasm::assemble::assemble_html(&payload.title, &payload_json, &assets);

    // Write HTML file
    std::fs::write(output_path, html_content)?;
    println!("\nPlots saved to: {:?}", output_path);

    Ok(())
}
