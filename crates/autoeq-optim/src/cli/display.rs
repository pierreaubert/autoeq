use super::super::optim::{AlgorithmType, get_all_algorithms};
use super::peq_model::PeqModel;
use std::fmt::Write as _;
use std::process;

fn constraint_label(algo: &crate::optim::AlgorithmInfo) -> &'static str {
    if algo.supports_nonlinear_constraints {
        "✅ Nonlinear"
    } else if algo.supports_linear_constraints {
        "🔶 Linear only"
    } else {
        "❌ None"
    }
}

/// Render the algorithm list to a string (used by tests and by the CLI printer).
pub fn render_algorithm_list() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Available Optimization Algorithms");
    let _ = writeln!(out, "=================================\n");

    let algorithms = get_all_algorithms();

    // Group algorithms by library
    let mut nlopt_algos = Vec::new();
    let mut metaheuristics_algos = Vec::new();
    let mut autoeq_algos = Vec::new();

    for algo in &algorithms {
        match algo.library {
            "NLOPT" => nlopt_algos.push(algo),
            "Metaheuristics" => metaheuristics_algos.push(algo),
            "AutoEQ" => autoeq_algos.push(algo),
            _ => {} // Skip unknown libraries
        }
    }

    // Display NLOPT algorithms
    if !nlopt_algos.is_empty() {
        let _ = writeln!(out, "📊 NLOPT Library Algorithms:");

        // Separate global and local algorithms
        let mut global = Vec::new();
        let mut local = Vec::new();

        for algo in nlopt_algos {
            match algo.algorithm_type {
                AlgorithmType::Global => global.push(algo),
                AlgorithmType::Local => local.push(algo),
            }
        }

        if !global.is_empty() {
            let _ = writeln!(
                out,
                "   🌍 Global Optimizers (best for exploring solution space):"
            );
            for algo in global {
                let _ = write!(out, "   - {:<20}", algo.name);
                let _ = write!(out, " | Constraints: ");
                let _ = write!(out, "{}", constraint_label(algo));

                // Add specific descriptions
                let description = match algo.name {
                    "nlopt:isres" => {
                        " | Improved Stochastic Ranking Evolution Strategy (recommended)"
                    }
                    "nlopt:ags" => " | Adaptive Geometric Search",
                    "nlopt:origdirect" => " | DIRECT global optimization (original version)",
                    "nlopt:crs2lm" => " | Controlled Random Search with local mutation",
                    "nlopt:direct" => " | DIRECT global optimization",
                    "nlopt:directl" => " | DIRECT-L (locally biased version)",
                    "nlopt:gmlsl" => " | Global Multi-Level Single-Linkage",
                    "nlopt:gmlsllds" => " | GMLSL with low-discrepancy sequence",
                    "nlopt:stogo" => " | Stochastic Global Optimization",
                    "nlopt:stogorand" => " | StoGO with randomized search",
                    _ => "",
                };
                let _ = writeln!(out, "{}", description);
            }
            let _ = writeln!(out);
        }

        if !local.is_empty() {
            let _ = writeln!(
                out,
                "   🎯 Local Optimizers (fast refinement from good starting points):"
            );
            for algo in local {
                let _ = write!(out, "   - {:<20}", algo.name);
                let _ = write!(out, " | Constraints: ");
                let _ = write!(out, "{}", constraint_label(algo));

                let description = match algo.name {
                    "nlopt:cobyla" => {
                        " | Constrained Optimization BY Linear Approximations (recommended for local)"
                    }
                    "nlopt:bobyqa" => " | Bound Optimization BY Quadratic Approximation",
                    "nlopt:neldermead" => " | Nelder-Mead simplex algorithm",
                    "nlopt:sbplx" => " | Subplex (variant of Nelder-Mead)",
                    "nlopt:slsqp" => " | Sequential Least SQuares Programming",
                    _ => "",
                };
                let _ = writeln!(out, "{}", description);
            }
            let _ = writeln!(out);
        }
    }

    // Display Metaheuristics algorithms
    if !metaheuristics_algos.is_empty() {
        let _ = writeln!(out, "🧬 Metaheuristics Library Algorithms:");
        let _ = writeln!(
            out,
            "   Nature-inspired global optimization (penalty-based constraints)\n"
        );

        for algo in metaheuristics_algos {
            let _ = write!(out, "   - {:<20}", algo.name);
            let description = match algo.name {
                "mh:de" => " | Differential Evolution (robust, good convergence)",
                "mh:pso" => " | Particle Swarm Optimization (fast exploration)",
                "mh:rga" => " | Real-coded Genetic Algorithm (diverse search)",
                "mh:tlbo" => " | Teaching-Learning-Based Optimization (parameter-free)",
                "mh:firefly" => " | Firefly Algorithm (multi-modal problems)",
                _ => "",
            };
            let _ = writeln!(out, "{}", description);
        }
        let _ = writeln!(out);
    }

    // Display AutoEQ algorithms
    if !autoeq_algos.is_empty() {
        let _ = writeln!(out, "🎵 AutoEQ Custom Algorithms:");
        let _ = writeln!(
            out,
            "   Specialized algorithms developed for audio filter optimization\n"
        );

        for algo in autoeq_algos {
            let _ = write!(out, "   - {:<20}", algo.name);
            let _ = write!(out, " | Constraints: ");
            if algo.supports_nonlinear_constraints {
                let _ = write!(out, "✅ Nonlinear");
            } else {
                let _ = write!(out, "❌ Penalty-based");
            }

            let description = match algo.name {
                "autoeq:de" => " | Adaptive DE with constraint handling (experimental)",
                _ => "",
            };
            let _ = writeln!(out, "{}", description);
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "Usage Examples:");
    let _ = writeln!(out, "==============\n");
    let _ = writeln!(out, "  # Use ISRES (recommended global optimizer):");
    let _ = writeln!(out, "  autoeq --algo nlopt:isres --curve input.csv\n");
    let _ = writeln!(out, "  # Use COBYLA (fast local optimizer):");
    let _ = writeln!(out, "  autoeq --algo nlopt:cobyla --curve input.csv\n");
    let _ = writeln!(out, "  # Use Differential Evolution from metaheuristics:");
    let _ = writeln!(out, "  autoeq --algo mh:de --curve input.csv\n");
    let _ = writeln!(out, "  # Backward compatibility (maps to nlopt:cobyla):");
    let _ = writeln!(out, "  autoeq --algo cobyla --curve input.csv\n");

    let _ = writeln!(out, "Recommendations:");
    let _ = writeln!(out, "===============\n");
    let _ = writeln!(
        out,
        "  🎯 For best results: nlopt:isres (global) + --refine with nlopt:cobyla (local)"
    );
    let _ = writeln!(
        out,
        "  ⚡ For speed: nlopt:cobyla (if you have a good initial guess)"
    );
    let _ = writeln!(
        out,
        "  🧪 For experimentation: mh:de or mh:pso from metaheuristics library"
    );
    let _ = writeln!(
        out,
        "  ⚖️  For constrained problems: Prefer algorithms with ✅ Nonlinear constraint support"
    );

    out
}

/// Display available optimization algorithms with descriptions and exit
pub fn display_algorithm_list() -> ! {
    print!("{}", render_algorithm_list());
    process::exit(0);
}

/// Render the DE strategy list to a string.
pub fn render_strategy_list() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Available Differential Evolution (DE) Strategies");
    let _ = writeln!(out, "===============================================\n");

    let strategies = [
        (
            "best1bin",
            "Best1Bin",
            "Use best individual + 1 random difference (binomial crossover)",
            "Global exploration with fast convergence",
        ),
        (
            "best1exp",
            "Best1Exp",
            "Use best individual + 1 random difference (exponential crossover)",
            "Similar to best1bin with different crossover",
        ),
        (
            "rand1bin",
            "Rand1Bin",
            "Use random individual + 1 random difference (binomial crossover)",
            "Good diversity, slower convergence",
        ),
        (
            "rand1exp",
            "Rand1Exp",
            "Use random individual + 1 random difference (exponential crossover)",
            "Similar to rand1bin with different crossover",
        ),
        (
            "rand2bin",
            "Rand2Bin",
            "Use random individual + 2 random differences (binomial crossover)",
            "High exploration, may be slower",
        ),
        (
            "rand2exp",
            "Rand2Exp",
            "Use random individual + 2 random differences (exponential crossover)",
            "Similar to rand2bin with different crossover",
        ),
        (
            "currenttobest1bin",
            "CurrentToBest1Bin",
            "Blend current with best + random difference (binomial)",
            "Balanced exploration/exploitation (recommended)",
        ),
        (
            "currenttobest1exp",
            "CurrentToBest1Exp",
            "Blend current with best + random difference (exponential)",
            "Similar to currenttobest1bin",
        ),
        (
            "best2bin",
            "Best2Bin",
            "Use best individual + 2 random differences (binomial crossover)",
            "Fast convergence, may get trapped locally",
        ),
        (
            "best2exp",
            "Best2Exp",
            "Use best individual + 2 random differences (exponential crossover)",
            "Similar to best2bin",
        ),
        (
            "randtobest1bin",
            "RandToBest1Bin",
            "Blend random with best + random difference (binomial)",
            "Good balance of diversity and convergence",
        ),
        (
            "randtobest1exp",
            "RandToBest1Exp",
            "Blend random with best + random difference (exponential)",
            "Similar to randtobest1bin",
        ),
        (
            "adaptivebin",
            "AdaptiveBin",
            "Self-adaptive mutation with top-w% selection (binomial)",
            "Advanced adaptive strategy (experimental)",
        ),
        (
            "adaptiveexp",
            "AdaptiveExp",
            "Self-adaptive mutation with top-w% selection (exponential)",
            "Advanced adaptive strategy (experimental)",
        ),
    ];

    let _ = writeln!(out, "🎯 Classic DE Strategies (well-tested, reliable):");
    for &(name, _enum_name, description, recommendation) in strategies.iter().take(12) {
        if name.starts_with("adaptive") {
            continue;
        }
        let _ = writeln!(out, "   - {:<20} | {}", name, description);
        let _ = writeln!(out, "     {:<20} | 💡 {}", "", recommendation);
        if name == "currenttobest1bin" {
            let _ = writeln!(out, "     {:<20} | ⭐ Recommended default strategy", "");
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(
        out,
        "🧬 Adaptive DE Strategies (experimental, research-based):"
    );
    for &(name, _enum_name, description, recommendation) in strategies.iter() {
        if !name.starts_with("adaptive") {
            continue;
        }
        let _ = writeln!(out, "   - {:<20} | {}", name, description);
        let _ = writeln!(out, "     {:<20} | 💡 {}", "", recommendation);
        let _ = writeln!(
            out,
            "     {:<20} | 🔧 Requires --adaptive-weight-f and --adaptive-weight-cr",
            ""
        );
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "Strategy Naming Conventions:");
    let _ = writeln!(out, "==========================\n");
    let _ = writeln!(
        out,
        "  • 'bin' = Binomial (uniform) crossover - each gene has equal probability"
    );
    let _ = writeln!(
        out,
        "  • 'exp' = Exponential crossover - contiguous segments are more likely"
    );
    let _ = writeln!(
        out,
        "  • Numbers (1, 2) indicate how many difference vectors are used\n"
    );

    let _ = writeln!(out, "Usage Examples:");
    let _ = writeln!(out, "==============\n");
    let _ = writeln!(out, "  # Use recommended default strategy:");
    let _ = writeln!(
        out,
        "  autoeq --algo autoeq:de --strategy currenttobest1bin --curve input.csv\n"
    );
    let _ = writeln!(out, "  # Use adaptive strategy with custom weights:");
    let _ = writeln!(
        out,
        "  autoeq --algo autoeq:de --strategy adaptivebin --adaptive-weight-f 0.8 --adaptive-weight-cr 0.7\n"
    );
    let _ = writeln!(out, "  # Use classic exploration strategy:");
    let _ = writeln!(
        out,
        "  autoeq --algo autoeq:de --strategy rand1bin --curve input.csv\n"
    );

    let _ = writeln!(out, "Recommendations:");
    let _ = writeln!(out, "===============\n");
    let _ = writeln!(
        out,
        "  ⭐ For general use: currenttobest1bin (good balance of exploration and exploitation)"
    );
    let _ = writeln!(
        out,
        "  🚀 For fast convergence: best1bin or best2bin (may get trapped in local optima)"
    );
    let _ = writeln!(
        out,
        "  🌍 For thorough exploration: rand1bin or rand2bin (slower but more robust)"
    );
    let _ = writeln!(
        out,
        "  🧪 For research/experimentation: adaptivebin or adaptiveexp (requires parameter tuning)"
    );

    out
}

/// Display available DE strategies with descriptions and exit
pub fn display_strategy_list() -> ! {
    print!("{}", render_strategy_list());
    process::exit(0);
}

/// Render the PEQ model list to a string.
pub fn render_peq_model_list() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Available PEQ Models");
    let _ = writeln!(out, "===================");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The PEQ model defines the structure and constraints of the equalizer filters."
    );
    let _ = writeln!(out);

    for model in PeqModel::all() {
        let _ = writeln!(out, "  --peq-model {}", model);
        let _ = writeln!(out, "    {}", model.description());
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "Examples:");
    let _ = writeln!(
        out,
        "  autoeq --peq-model pk           # All peak filters (default)"
    );
    let _ = writeln!(out, "  autoeq --peq-model hp-pk        # Highpass + peaks");
    let _ = writeln!(
        out,
        "  autoeq --peq-model hp-pk-lp     # Highpass + peaks + lowpass"
    );

    out
}

/// Display available PEQ models with descriptions and exit
pub fn display_peq_model_list() -> ! {
    print!("{}", render_peq_model_list());
    process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_algorithm_list_contains_headers_and_known_algorithms() {
        let s = render_algorithm_list();
        assert!(s.contains("Available Optimization Algorithms"));
        assert!(s.contains("autoeq:de") || s.contains("mh:de"));
        assert!(s.contains("Usage Examples"));
        assert!(s.contains("Recommendations"));
    }

    #[test]
    fn render_strategy_list_contains_recommended_strategy() {
        let s = render_strategy_list();
        assert!(s.contains("Available Differential Evolution"));
        assert!(s.contains("currenttobest1bin"));
        assert!(s.contains("Recommended default strategy"));
        assert!(s.contains("adaptivebin"));
    }

    #[test]
    fn render_peq_model_list_contains_all_models() {
        let s = render_peq_model_list();
        assert!(s.contains("Available PEQ Models"));
        for model in PeqModel::all() {
            assert!(s.contains(&format!("--peq-model {}", model)));
            assert!(s.contains(model.description()));
        }
    }
}
