// Rust guideline compliant 2026-02-21
use autoeq_measurements::{
    InlineMeasurement, MeasurementMultiple, MeasurementRef, MeasurementSource,
};
use roomeq_analysis::spatial_robustness::{
    SpatialRobustnessConfig, analyze_spatial_robustness_weighted,
};
use roomeq_workflow::channel_measurements::prepare_channel_measurements_with_frequency_samples;

fn source(bands: &[[f64; 2]]) -> MeasurementSource {
    let frequencies: Vec<f64> = (1..=1000).map(|index| f64::from(index) * 10.0).collect();
    let measurements = [200_usize, 201]
        .into_iter()
        .map(|peak_index| {
            let mut levels = vec![80.0; frequencies.len()];
            levels[peak_index] = 90.0;
            MeasurementRef::Inline(InlineMeasurement {
                frequencies: frequencies.clone(),
                magnitude_db: levels,
                // A one-millisecond delay supplies phase without random fixture data.
                phase_deg: Some(
                    frequencies
                        .iter()
                        .map(|frequency| -0.36 * frequency)
                        .collect(),
                ),
                name: None,
                wav_path: None,
                csv_path: None,
            })
        })
        .collect();
    let mut multiple = MeasurementMultiple {
        measurements,
        speaker_name: None,
        provenance: Default::default(),
        estimated_position: None,
    };
    multiple.provenance.valid_bands_hz = bands.to_vec();
    MeasurementSource::Multiple(multiple)
}

fn assert_spatial_inputs(frequency_samples: usize, bands: &[[f64; 2]]) {
    let prepared =
        prepare_channel_measurements_with_frequency_samples(&source(bands), frequency_samples)
            .expect("prepare deterministic position measurements");
    let positions = prepared.individual();
    assert_eq!(positions.len(), 2);
    assert!(
        positions[0].freq == positions[1].freq,
        "conditioning must retain a common position grid despite different extrema"
    );
    assert!(
        prepared.representative().freq == positions[0].freq,
        "representative and position grids must agree"
    );
    for (position, peak_frequency) in positions.iter().zip([2010.0, 2020.0]) {
        let peak_index = position
            .freq
            .iter()
            .position(|&frequency| frequency == peak_frequency)
            .expect("retain the measured peak frequency");
        assert!((position.spl[peak_index] - 90.0).abs() < 1e-10);
        let phase = position.phase.as_ref().expect("retain measured phase");
        assert_eq!(phase.len(), position.freq.len());
        assert!(phase.iter().all(|value| value.is_finite()));
    }
    let usable: Vec<_> = positions
        .iter()
        .map(|curve| {
            if bands.is_empty() {
                curve.clone()
            } else {
                curve
                    .select_frequency_bands(bands)
                    .expect("select usable support")
            }
        })
        .collect();
    if !bands.is_empty() {
        assert!(
            usable
                .iter()
                .all(|curve| curve.freq.iter().all(|frequency| {
                    bands
                        .iter()
                        .any(|[low, high]| frequency >= low && frequency <= high)
                }))
        );
    }
    let analysis =
        analyze_spatial_robustness_weighted(&usable, &SpatialRobustnessConfig::default(), None)
            .expect("prepared positions must support spatial analysis");
    assert_eq!(analysis.correction_depth.len(), usable[0].freq.len());
}

#[test]
fn native_positions_support_spatial_analysis() {
    assert_spatial_inputs(0, &[]);
}

#[test]
fn reduced_positions_preserve_distinct_peaks_on_a_common_grid() {
    assert_spatial_inputs(128, &[]);
}

#[test]
fn reduced_positions_keep_declared_gaps_out_of_spatial_analysis() {
    assert_spatial_inputs(128, &[[10.0, 3000.0], [5000.0, 10000.0]]);
}
