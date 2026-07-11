#[cfg(test)]
mod tests {
    use crate::prescore::compute_pre_optimization_metrics;
    use autoeq::Curve;
    use autoeq::PeqModel;
    use autoeq::cli::Args;
    use autoeq::loss::LossType;
    use autoeq::optim::{ObjectiveData, ObjectiveDataBuilder};
    use clap::Parser;
    use ndarray::Array1;

    fn create_test_objective_data(loss_type: LossType) -> ObjectiveData {
        let freqs = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let deviation = Array1::from_vec(vec![2.0, 1.5, 1.0, 1.2, 0.8]);
        let target = Array1::zeros(freqs.len());

        ObjectiveDataBuilder::new(freqs, target, deviation, 48000.0, PeqModel::Pk, loss_type)
            .min_spacing_oct(0.1)
            .max_db(10.0)
            .min_db(-10.0)
            .freq_range(20.0, 20000.0)
            .smoothing(false, 3)
            .build()
            .expect("valid test objective data")
    }

    #[tokio::test]
    async fn test_compute_pre_optimization_headphone() {
        let args = Args::try_parse_from(["autoeq-test", "--loss", "headphone-flat"]).unwrap();
        let objective_data = create_test_objective_data(LossType::HeadphoneFlat);

        let deviation = Curve {
            freq: objective_data.freqs.clone(),
            spl: objective_data.deviation.clone(),
            phase: None,
            ..Default::default()
        };

        let metrics =
            compute_pre_optimization_metrics(&args, &objective_data, false, &deviation, &None)
                .await
                .expect("headphone pre-optimization metrics should compute");

        // Headphone loss should be computed
        assert!(metrics.headphone_loss.unwrap().is_finite());
        // CEA metrics should be None
        assert!(metrics.cea2034_metrics.is_none());
    }

    #[tokio::test]
    async fn test_compute_pre_optimization_speaker_flat() {
        let args = Args::try_parse_from(["autoeq-test", "--loss", "speaker-flat"]).unwrap();
        let objective_data = create_test_objective_data(LossType::SpeakerFlat);

        let deviation = Curve {
            freq: objective_data.freqs.clone(),
            spl: objective_data.deviation.clone(),
            phase: None,
            ..Default::default()
        };

        let metrics = compute_pre_optimization_metrics(
            &args,
            &objective_data,
            false, // no CEA data
            &deviation,
            &None,
        )
        .await
        .expect("speaker-flat pre-optimization metrics should compute");

        // Both should be None without CEA data
        assert!(metrics.headphone_loss.is_none());
        assert!(metrics.cea2034_metrics.is_none());
    }
}
