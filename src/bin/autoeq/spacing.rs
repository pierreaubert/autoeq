use autoeq::optim;

/// Check if spacing constraints are met
pub(super) fn check_spacing_constraints(x: &[f64], params: &autoeq::OptimParams) -> bool {
    let peq_model = params.peq_model;
    let (_, adj_spacings) = optim::compute_sorted_freqs_and_adjacent_octave_spacings(x, peq_model);
    let min_adj = adj_spacings.iter().cloned().fold(f64::INFINITY, f64::min);
    min_adj >= params.min_spacing_oct && min_adj.is_finite()
}

/// Print frequency spacing diagnostics and PEQ listing
pub(super) fn print_freq_spacing(x: &[f64], params: &autoeq::OptimParams, label: &str) {
    let peq_model = params.peq_model;
    let (sorted_freqs, adj_spacings) =
        optim::compute_sorted_freqs_and_adjacent_octave_spacings(x, peq_model);
    let min_adj = adj_spacings.iter().cloned().fold(f64::INFINITY, f64::min);
    let freqs_fmt: Vec<String> = sorted_freqs.iter().map(|f| format!("{:.0}", f)).collect();
    let spacings_fmt: Vec<String> = adj_spacings.iter().map(|s| format!("{:.2}", s)).collect();
    if min_adj >= params.min_spacing_oct {
        println!("✅ Spacing diagnostics ({}):", label);
    } else {
        println!("⚠️ Spacing diagnostics ({}):", label);
    }
    println!("  - Sorted center freqs (Hz): [{}]", freqs_fmt.join(", "));
    println!(
        "  - Adjacent spacings (oct):   [{}]",
        spacings_fmt.join(", ")
    );
    if min_adj.is_finite() {
        println!(
            "  - Min adjacent spacing: {:.4} oct (constraint {:.4} oct)",
            min_adj, params.min_spacing_oct
        );
    } else {
        println!("  - Not enough filters to compute spacing.");
    }
    autoeq::x2peq::peq_print_from_x(x, params.sample_rate, params.peq_model);
}
