//! Generated regression tests for PEQ layouts and optimizer setup invariants.

use crate::cli::Args;
use crate::param_utils::{self, PeqLayout};
use crate::workflow::{initial_guess, setup_bounds};
use crate::x2peq::{peq2x, try_x2spl, x2peq};
use crate::{OptimParams, PeqModel};
use ndarray::Array1;
use proptest::prelude::*;

fn peq_model_strategy() -> impl Strategy<Value = PeqModel> {
    prop_oneof![
        Just(PeqModel::Pk),
        Just(PeqModel::HpPk),
        Just(PeqModel::HpPkLp),
        Just(PeqModel::LsPk),
        Just(PeqModel::LsPkHs),
        Just(PeqModel::FreePkFree),
        Just(PeqModel::Free),
    ]
}

fn filter_parameter_strategy() -> impl Strategy<Value = (u8, f64, f64, f64)> {
    (
        0_u8..12,
        1.4_f64..4.2, // 25 Hz to 15.8 kHz, safely below Nyquist at 48 kHz.
        0.3_f64..10.0,
        -12.0_f64..12.0,
    )
}

fn canonical_parameter_vector(model: PeqModel, filters: &[(u8, f64, f64, f64)]) -> Vec<f64> {
    let mut parameters = Vec::with_capacity(filters.len() * param_utils::params_per_filter(model));
    let last = filters.len() - 1;

    for (index, &(filter_type, frequency, q, gain)) in filters.iter().enumerate() {
        if matches!(model, PeqModel::Free | PeqModel::FreePkFree) {
            let filter_type = match model {
                PeqModel::Free => filter_type,
                PeqModel::FreePkFree if index == 0 || index == last => filter_type,
                PeqModel::FreePkFree => 0, // Interior filters are fixed peaks.
                _ => unreachable!(),
            };
            parameters.push(f64::from(filter_type));
        }
        parameters.extend_from_slice(&[frequency, q, gain]);
    }

    parameters
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn filter_type_bins_stay_decodable_and_in_bounds(
        type_index in 0_u8..12,
        fractional_part in 0.0_f64..0.999,
    ) {
        let encoded = f64::from(type_index) + fractional_part;
        let expected = param_utils::decode_filter_type(f64::from(type_index));
        let (lower, upper) = param_utils::filter_type_bounds();

        prop_assert!(encoded >= lower && encoded <= upper);
        prop_assert_eq!(param_utils::decode_filter_type(encoded), expected);
    }

    #[test]
    fn peq_layout_conversion_and_response_preserve_invariants(
        model in peq_model_strategy(),
        filters in prop::collection::vec(filter_parameter_strategy(), 1..9),
    ) {
        let original = canonical_parameter_vector(model, &filters);
        let mut rebuilt = vec![0.0; original.len()];

        for index in 0..filters.len() {
            let parameters = param_utils::get_filter_params(&original, index, model);
            param_utils::set_filter_params(&mut rebuilt, index, &parameters, model);
        }
        prop_assert_eq!(rebuilt.as_slice(), original.as_slice());

        let peq = x2peq(&original, 48_000.0, model);
        let recovered = peq2x(&peq, model);
        prop_assert_eq!(recovered.len(), original.len());
        for (index, (&expected, &actual)) in original.iter().zip(&recovered).enumerate() {
            prop_assert!(
                (expected - actual).abs() <= 1e-10,
                "parameter {index} changed during {model} round-trip: {expected} -> {actual}"
            );
        }

        let frequencies = Array1::from_vec(vec![
            20.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0,
            4_000.0, 8_000.0, 16_000.0, 20_000.0,
        ]);
        let response = try_x2spl(&frequencies, &recovered, 48_000.0, model)
            .map_err(|error| TestCaseError::fail(format!("{model} response failed: {error}")))?;
        prop_assert_eq!(response.len(), frequencies.len());
        prop_assert!(response.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn optimizer_bounds_and_initial_guess_are_coherent(
        model in peq_model_strategy(),
        num_filters in 1_usize..33,
        min_frequency_log10 in 1.0_f64..3.0,
        frequency_span_log10 in 0.01_f64..1.3,
        min_q in 0.1_f64..5.0,
        q_span in 0.0_f64..15.0,
        min_gain in 0.05_f64..3.0,
        max_gain in 3.0_f64..24.0,
    ) {
        let mut args = Args::speaker_defaults();
        args.peq_model = model;
        args.num_filters = num_filters;
        args.min_freq = 10_f64.powf(min_frequency_log10);
        args.max_freq = 10_f64.powf(min_frequency_log10 + frequency_span_log10);
        args.min_q = min_q;
        args.max_q = min_q + q_span;
        args.min_db = min_gain;
        args.max_db = max_gain;
        args.qa = Some(0.0);

        let params = OptimParams::from(&args);
        let (lower, upper) = setup_bounds(&params);
        let candidate = initial_guess(&params, &lower, &upper);
        let expected_len = num_filters * param_utils::params_per_filter(model);
        let layout = model.layout();
        let parameters_per_filter = model.params_per_filter();
        let min_log_frequency = args.min_freq.log10();
        let max_log_frequency = args.max_freq.log10();
        let effective_gain_min = if args.min_db < 0.0 {
            args.min_db
        } else {
            -3.0 * args.max_db
        };

        prop_assert_eq!(lower.len(), expected_len);
        prop_assert_eq!(upper.len(), expected_len);
        prop_assert_eq!(candidate.len(), expected_len);

        for (index, ((&lower, &upper), &value)) in lower
            .iter()
            .zip(&upper)
            .zip(&candidate)
            .enumerate()
        {
            prop_assert!(lower.is_finite(), "lower bound {index} is not finite");
            prop_assert!(upper.is_finite(), "upper bound {index} is not finite");
            prop_assert!(lower <= upper, "bounds {index} are reversed: {lower} > {upper}");
            prop_assert!(value.is_finite(), "initial value {index} is not finite");
            prop_assert!(
                value >= lower && value <= upper,
                "initial value {index}={value} is outside [{lower}, {upper}]"
            );
        }

        for filter in 0..num_filters {
            let offset = filter * parameters_per_filter;
            let frequency_index = offset + layout.freq_idx;
            let q_index = offset + layout.q_idx;
            let gain_index = offset + layout.gain_idx;

            prop_assert!(lower[frequency_index] >= min_log_frequency);
            prop_assert!(upper[frequency_index] <= max_log_frequency);
            prop_assert!(lower[q_index] >= args.min_q.max(0.1));
            prop_assert!(upper[q_index] <= args.max_q);
            prop_assert!(lower[gain_index] >= effective_gain_min);
            prop_assert!(upper[gain_index] <= args.max_db);
        }
    }
}
