use super::args::Args;
use crate::LossType;
use crate::de::Strategy;
use std::process;

/// Validate CLI arguments and exit with error message if validation fails
pub fn validate_args(args: &Args) -> Result<(), String> {
    // Check if strategy is valid when using DE algorithm
    if args.algo == "autoeq:de" || args.algo.contains("de") {
        use std::str::FromStr;
        if let Err(err) = Strategy::from_str(&args.strategy) {
            return Err(format!(
                "Invalid DE strategy '{}': {}. Use --strategy-list to see available strategies.",
                args.strategy, err
            ));
        }
    }
    // Check if algorithm is valid
    if crate::optim::find_algorithm_info(&args.algo).is_some() {
        // Algorithm is valid
    } else {
        return Err(format!(
            "Unknown algorithm: '{}'. Use --algo-list to see available algorithms.",
            args.algo
        ));
    }

    // Check if local algorithm is valid (when refine is enabled)
    if args.refine {
        if crate::optim::find_algorithm_info(&args.local_algo).is_some() {
            // Local algorithm is valid
        } else {
            return Err(format!(
                "Unknown local algorithm: '{}'. Use --algo-list to see available algorithms.",
                args.local_algo
            ));
        }
    }

    // Check min/max Q factor constraints
    if args.min_q > args.max_q {
        return Err(format!(
            "Invalid Q factor range: min_q ({}) must be <= max_q ({})",
            args.min_q, args.max_q
        ));
    }

    // Check min/max frequency constraints
    if args.min_freq > args.max_freq {
        return Err(format!(
            "Invalid frequency range: min_freq ({}) must be <= max_freq ({})",
            args.min_freq, args.max_freq
        ));
    }

    // Check min/max dB constraints
    if args.min_db > args.max_db {
        return Err(format!(
            "Invalid dB range: min_db ({}) must be <= max_db ({})",
            args.min_db, args.max_db
        ));
    }

    // Check frequency bounds (reasonable audio range)
    if args.min_freq < 20.0 {
        return Err(format!(
            "Invalid min_freq: {} Hz. Must be >= 20 Hz (reasonable audio range)",
            args.min_freq
        ));
    }

    if args.max_freq > 20000.0 {
        return Err(format!(
            "Invalid max_freq: {} Hz. Must be <= 20,000 Hz (reasonable audio range)",
            args.max_freq
        ));
    }

    // Check that max_freq does not exceed Nyquist frequency
    let nyquist = args.sample_rate / 2.0;
    if args.max_freq > nyquist {
        return Err(format!(
            "max_freq ({:.0} Hz) exceeds Nyquist frequency ({:.0} Hz) at sample rate {:.0} Hz. \
             Biquad filters above Nyquist have undefined behavior.",
            args.max_freq, nyquist, args.sample_rate
        ));
    }

    // Check smoothing parameters
    if args.smooth_n < 1 || args.smooth_n > 24 {
        return Err(format!(
            "Invalid smooth_n: {}. Must be in range [1..24]",
            args.smooth_n
        ));
    }

    // Check that population size is reasonable
    if args.population == 0 {
        return Err("Population size must be > 0".to_string());
    }

    // Check that maxeval is reasonable
    if args.maxeval == 0 {
        return Err("Maximum evaluations must be > 0".to_string());
    }

    // Check that num_filters is reasonable
    if args.num_filters == 0 {
        return Err("Number of filters must be > 0".to_string());
    }

    if args.num_filters > 50 {
        return Err(format!(
            "Number of filters ({}) is very high. Consider using <= 50 filters for reasonable performance",
            args.num_filters
        ));
    }

    // Check tolerance parameters
    if args.tolerance <= 0.0 {
        return Err("Tolerance must be > 0".to_string());
    }

    if args.atolerance < 0.0 {
        return Err("Absolute tolerance must be >= 0".to_string());
    }

    // Check adaptive weight parameters (should be in [0, 1])
    if args.adaptive_weight_f < 0.0 || args.adaptive_weight_f > 1.0 {
        return Err("Adaptive weight for F must be between 0.0 and 1.0".to_string());
    }

    if args.adaptive_weight_cr < 0.0 || args.adaptive_weight_cr > 1.0 {
        return Err("Adaptive weight for CR must be between 0.0 and 1.0".to_string());
    }

    // Validate multi-driver arguments
    if args.loss == LossType::DriversFlat {
        // Check that at least driver1 and driver2 are provided
        if args.driver1.is_none() || args.driver2.is_none() {
            return Err("Multi-driver optimization requires at least --driver1 and --driver2 when using --loss drivers-flat".to_string());
        }

        // Check crossover type
        let valid_crossover_types = ["butterworth2", "linkwitzriley2", "linkwitzriley4"];
        if !valid_crossover_types.contains(&args.crossover_type.as_str()) {
            return Err(format!(
                "Invalid crossover type '{}'. Valid types: {}",
                args.crossover_type,
                valid_crossover_types.join(", ")
            ));
        }

        // Count number of drivers
        let n_drivers = [&args.driver1, &args.driver2, &args.driver3, &args.driver4]
            .iter()
            .filter(|d| d.is_some())
            .count();
        if !(2..=4).contains(&n_drivers) {
            return Err(format!(
                "Multi-driver optimization requires 2-4 drivers, got {}",
                n_drivers
            ));
        }
    } else {
        // If not using drivers-flat loss, driver arguments should not be provided
        if args.driver1.is_some()
            || args.driver2.is_some()
            || args.driver3.is_some()
            || args.driver4.is_some()
        {
            return Err("Driver arguments (--driver1, --driver2, etc.) can only be used with --loss drivers-flat".to_string());
        }
    }

    Ok(())
}

/// Validate arguments and exit with error if validation fails
pub fn validate_args_or_exit(args: &Args) {
    if let Err(error) = validate_args(args) {
        eprintln!("❌ Validation Error: {}", error);
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::super::args::Args;
    use super::validate_args;
    use crate::LossType;
    use clap::Parser;
    use std::path::PathBuf;

    fn valid_args() -> Args {
        Args::try_parse_from::<&[&str], _>(&["prog"]).unwrap()
    }

    #[test]
    fn valid_default_args_pass() {
        let args = valid_args();
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn min_q_greater_than_max_q_fails() {
        let mut args = valid_args();
        args.min_q = 5.0;
        args.max_q = 1.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("min_q"));
        assert!(err.contains("max_q"));
    }

    #[test]
    fn min_freq_greater_than_max_freq_fails() {
        let mut args = valid_args();
        args.min_freq = 200.0;
        args.max_freq = 100.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("min_freq"));
        assert!(err.contains("max_freq"));
    }

    #[test]
    fn max_freq_above_nyquist_fails() {
        let mut args = valid_args();
        args.sample_rate = 1000.0;
        args.max_freq = 600.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Nyquist"));
    }

    #[test]
    fn driver_args_without_drivers_flat_loss_fails() {
        let mut args = valid_args();
        args.loss = LossType::SpeakerFlat;
        args.driver1 = Some(PathBuf::from("d1.csv"));
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Driver arguments"));
    }

    #[test]
    fn drivers_flat_without_drivers_fails() {
        let mut args = valid_args();
        args.loss = LossType::DriversFlat;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("requires at least --driver1 and --driver2"));
    }

    #[test]
    fn drivers_flat_with_invalid_crossover_type_fails() {
        let mut args = valid_args();
        args.loss = LossType::DriversFlat;
        args.driver1 = Some(PathBuf::from("d1.csv"));
        args.driver2 = Some(PathBuf::from("d2.csv"));
        args.crossover_type = "invalid".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Invalid crossover type"));
    }

    #[test]
    fn drivers_flat_with_valid_crossover_passes() {
        let mut args = valid_args();
        args.loss = LossType::DriversFlat;
        args.driver1 = Some(PathBuf::from("d1.csv"));
        args.driver2 = Some(PathBuf::from("d2.csv"));
        args.crossover_type = "linkwitzriley4".to_string();
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn refine_with_unknown_local_algo_fails() {
        let mut args = valid_args();
        args.refine = true;
        args.local_algo = "unknown".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Unknown local algorithm"));
    }

    #[test]
    fn min_freq_below_reasonable_range_fails() {
        let mut args = valid_args();
        args.min_freq = 10.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Must be >= 20 Hz"));
    }

    #[test]
    fn max_freq_above_reasonable_range_fails() {
        let mut args = valid_args();
        args.max_freq = 25000.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Must be <= 20,000 Hz"));
    }

    #[test]
    fn smooth_n_out_of_range_fails() {
        let mut args = valid_args();
        args.smooth_n = 0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("smooth_n"));

        let mut args = valid_args();
        args.smooth_n = 25;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("smooth_n"));
    }

    #[test]
    fn population_zero_fails() {
        let mut args = valid_args();
        args.population = 0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Population size"));
    }

    #[test]
    fn maxeval_zero_fails() {
        let mut args = valid_args();
        args.maxeval = 0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Maximum evaluations"));
    }

    #[test]
    fn num_filters_zero_or_too_high_fails() {
        let mut args = valid_args();
        args.num_filters = 0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Number of filters"));

        let mut args = valid_args();
        args.num_filters = 51;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("<= 50 filters"));
    }

    #[test]
    fn tolerance_non_positive_fails() {
        let mut args = valid_args();
        args.tolerance = 0.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Tolerance"));
    }

    #[test]
    fn atolerance_negative_fails() {
        let mut args = valid_args();
        args.atolerance = -1.0;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Absolute tolerance"));
    }

    #[test]
    fn adaptive_weights_out_of_range_fails() {
        let mut args = valid_args();
        args.adaptive_weight_f = 1.5;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Adaptive weight for F"));

        let mut args = valid_args();
        args.adaptive_weight_cr = -0.1;
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Adaptive weight for CR"));
    }

    #[test]
    fn invalid_de_strategy_fails() {
        let mut args = valid_args();
        args.algo = "autoeq:de".to_string();
        args.strategy = "not-a-strategy".to_string();
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("Invalid DE strategy"));
    }

    #[test]
    fn drivers_flat_with_too_few_drivers_fails() {
        let mut args = valid_args();
        args.loss = LossType::DriversFlat;
        args.driver1 = Some(PathBuf::from("d1.csv"));
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("requires at least --driver1 and --driver2"));
    }

    #[test]
    fn drivers_flat_accepts_all_valid_crossover_types() {
        for ct in ["butterworth2", "linkwitzriley2", "linkwitzriley4"] {
            let mut args = valid_args();
            args.loss = LossType::DriversFlat;
            args.driver1 = Some(PathBuf::from("d1.csv"));
            args.driver2 = Some(PathBuf::from("d2.csv"));
            args.crossover_type = ct.to_string();
            assert!(
                validate_args(&args).is_ok(),
                "crossover {} should be valid",
                ct
            );
        }
    }
}
