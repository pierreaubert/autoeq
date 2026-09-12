use super::misc::initial_guess;
use super::{
    q_max_for_frequency_range, setup_bounds, setup_drivers_bounds,
    setup_drivers_bounds_fixed_freqs, setup_drivers_objective_data, setup_multisub_bounds,
    setup_multisub_objective_data,
};

use crate::FrequencyQPolicy;
use crate::OptimParams;
use crate::roomeq::OptimizerConfig;

#[test]
fn frequency_q_policy_keeps_modal_freedom_but_caps_guarded_ranges() {
    let config = OptimizerConfig {
        num_filters: 3,
        min_freq: 20.0,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: 12.0,
        ..OptimizerConfig::default()
    };
    let mut params = OptimParams::from(&config);
    params.frequency_q_policy = Some(FrequencyQPolicy {
        schroeder_hz: Some(500.0),
        low_max_q: Some(12.0),
        high_start_hz: Some(1_600.0),
        high_max_q: Some(0.8),
    });

    assert_eq!(q_max_for_frequency_range(&params, 80.0, 300.0), 12.0);
    let crossing = q_max_for_frequency_range(&params, 300.0, 800.0);
    assert!(crossing > 0.8 && crossing < 12.0);
    assert_eq!(q_max_for_frequency_range(&params, 2_000.0, 8_000.0), 0.8);

    let (_, upper) = setup_bounds(&params);
    for filter in 0..params.num_filters {
        let offset = filter * 3;
        let f_high = 10.0_f64.powf(upper[offset]);
        if f_high >= 1_600.0 {
            assert!(upper[offset + 1] <= 0.8);
        }
    }
}

#[test]
fn frequency_q_policy_is_continuous_at_schroeder_boundary() {
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20_000.0,
        min_q: 0.5,
        max_q: 12.0,
        ..OptimizerConfig::default()
    };
    let mut params = OptimParams::from(&config);
    params.frequency_q_policy = Some(FrequencyQPolicy {
        schroeder_hz: Some(500.0),
        low_max_q: Some(12.0),
        high_start_hz: Some(1_600.0),
        high_max_q: Some(0.8),
    });
    let below = q_max_for_frequency_range(&params, 499.0, 499.0);
    let above = q_max_for_frequency_range(&params, 501.0, 501.0);
    assert_eq!(below, 12.0);
    assert!(above < below);
    assert!(above > 0.8);
    assert!(
        (below - above) < 1.0,
        "Schroeder boundary Q jump is too large"
    );
}

#[test]
fn setup_bounds_keeps_special_filters_inside_narrow_measurement_ranges() {
    for peq_model in ["ls-pk-hs", "hp-pk-lp"] {
        let config = OptimizerConfig {
            peq_model: peq_model.to_string(),
            num_filters: 2,
            min_freq: 20.0,
            max_freq: 400.0,
            min_q: 0.5,
            max_q: 0.8,
            ..OptimizerConfig::default()
        };
        let params = OptimParams::from(&config);
        let (lower_bounds, upper_bounds) = setup_bounds(&params);

        for filter in 0..params.num_filters {
            let offset = filter * 3;
            assert!(lower_bounds[offset] >= params.min_freq.log10());
            assert!(upper_bounds[offset] <= params.max_freq.log10());
            assert!(lower_bounds[offset] <= upper_bounds[offset]);
            assert!(lower_bounds[offset + 1] >= params.min_q);
            assert!(upper_bounds[offset + 1] <= params.max_q);
        }

        let guess = initial_guess(&params, &lower_bounds, &upper_bounds);
        assert_eq!(guess.len(), params.num_filters * 3);
    }
}

#[test]
fn setup_bounds_uses_negative_min_db_as_lower_gain_bound() {
    let config = OptimizerConfig {
        peq_model: "pk".to_string(),
        num_filters: 2,
        min_db: -10.0,
        max_db: 3.0,
        ..OptimizerConfig::default()
    };
    let params = OptimParams::from(&config);
    let (lower_bounds, upper_bounds) = setup_bounds(&params);

    for i in 0..params.num_filters {
        let gain_idx = i * 3 + 2;
        assert_eq!(lower_bounds[gain_idx], -10.0);
        assert_eq!(upper_bounds[gain_idx], 3.0);
    }
}

// ---------------------------------------------------------------------------
// Drivers / multisub setup tests
// ---------------------------------------------------------------------------

use crate::loss::CrossoverType;
use crate::loss::DriverMeasurement;
use crate::loss::DriversLossData;

#[test]
fn setup_bounds_pins_standard_shelf_q_dimensions() {
    for peq_model in ["ls-pk", "ls-pk-hs"] {
        let config = OptimizerConfig {
            peq_model: peq_model.to_string(),
            num_filters: 4,
            min_q: 0.5,
            max_q: 3.0,
            ..OptimizerConfig::default()
        };
        let params = OptimParams::from(&config);
        let (lower_bounds, upper_bounds) = setup_bounds(&params);

        assert_eq!(lower_bounds[1], 1.0);
        assert_eq!(upper_bounds[1], 1.0);
        if peq_model == "ls-pk-hs" {
            let last_q = (params.num_filters - 1) * 3 + 1;
            assert_eq!(lower_bounds[last_q], 1.0);
            assert_eq!(upper_bounds[last_q], 1.0);
        }
    }
}

fn make_test_driver(freqs: Vec<f64>, spls: Vec<f64>) -> DriverMeasurement {
    DriverMeasurement::new(
        ndarray::Array1::from_vec(freqs),
        ndarray::Array1::from_vec(spls),
        None,
    )
}

#[test]
fn setup_drivers_objective_data_sets_correct_loss_type() {
    let driver = make_test_driver(vec![20.0, 100.0, 1000.0], vec![80.0, 82.0, 78.0]);
    let drivers_data = DriversLossData::new(
        vec![driver.clone(), driver.clone()],
        CrossoverType::LinkwitzRiley4,
    );
    let config = OptimizerConfig::default();
    let params = OptimParams::from(&config);

    let obj = setup_drivers_objective_data(&params, drivers_data);
    assert!(matches!(obj.loss_type, crate::loss::LossType::DriversFlat));
    assert!(obj.drivers_data.is_some());
}

#[test]
fn setup_drivers_bounds_has_gains_delays_and_crossovers() {
    let low_driver = make_test_driver(vec![20.0, 100.0, 500.0], vec![80.0, 82.0, 78.0]);
    let high_driver = make_test_driver(vec![500.0, 2000.0, 10000.0], vec![78.0, 80.0, 76.0]);
    let drivers_data =
        DriversLossData::new(vec![low_driver, high_driver], CrossoverType::LinkwitzRiley4);
    let config = OptimizerConfig::default();
    let params = OptimParams::from(&config);

    let (lower, upper) = setup_drivers_bounds(&params, &drivers_data);
    // 2 drivers => 2 gains + 2 delays + 1 crossover = 5 params
    assert_eq!(lower.len(), 5);
    assert_eq!(upper.len(), 5);
    // Gains bounded by [-max_db, max_db]
    assert_eq!(lower[0], -params.max_db);
    assert_eq!(upper[0], params.max_db);
    // Delays bounded by [-20, 20]
    assert_eq!(lower[2], -20.0);
    assert_eq!(upper[2], 20.0);
    // Crossover bounds should be valid (lower <= upper)
    assert!(lower[4] <= upper[4]);
}

#[test]
fn setup_drivers_bounds_fixed_freqs_has_no_crossovers() {
    let low_driver = make_test_driver(vec![20.0, 100.0, 500.0], vec![80.0, 82.0, 78.0]);
    let high_driver = make_test_driver(vec![500.0, 2000.0, 10000.0], vec![78.0, 80.0, 76.0]);
    let drivers_data =
        DriversLossData::new(vec![low_driver, high_driver], CrossoverType::LinkwitzRiley4);
    let config = OptimizerConfig::default();
    let params = OptimParams::from(&config);

    let (lower, upper) = setup_drivers_bounds_fixed_freqs(&params, &drivers_data);
    // 2 drivers => 2 gains + 2 delays = 4 params
    assert_eq!(lower.len(), 4);
    assert_eq!(upper.len(), 4);
}

#[test]
fn setup_multisub_objective_data_sets_correct_loss_type() {
    let driver = make_test_driver(vec![20.0, 50.0, 200.0], vec![85.0, 88.0, 86.0]);
    let drivers_data =
        DriversLossData::new(vec![driver.clone(), driver.clone()], CrossoverType::None);
    let config = OptimizerConfig::default();
    let params = OptimParams::from(&config);

    let obj = setup_multisub_objective_data(&params, drivers_data);
    assert!(matches!(obj.loss_type, crate::loss::LossType::MultiSubFlat));
    assert!(obj.drivers_data.is_some());
}

#[test]
fn setup_multisub_bounds_has_gains_and_delays() {
    let config = OptimizerConfig::default();
    let params = OptimParams::from(&config);
    let n_drivers = 3;

    let (lower, upper) = setup_multisub_bounds(&params, n_drivers);
    // 3 drivers => 3 gains + 3 delays = 6 params
    assert_eq!(lower.len(), 6);
    assert_eq!(upper.len(), 6);
    // Gains
    for i in 0..n_drivers {
        assert_eq!(lower[i], -params.max_db);
        assert_eq!(upper[i], params.max_db);
    }
    // Delays (0 to 20 ms)
    for i in n_drivers..(2 * n_drivers) {
        assert_eq!(lower[i], 0.0);
        assert_eq!(upper[i], 20.0);
    }
}
