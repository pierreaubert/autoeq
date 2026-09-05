#[cfg(test)]
#[path = "../../tests/common/apo.rs"]
mod apo;

#[cfg(test)]
mod tests {
    use super::apo::parse_apo_filters;
    use crate::autoeq_command::save::save_peq_to_file;
    use autoeq::cli::Args;
    use autoeq::loss::LossType;
    use clap::Parser;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_peq_to_file_apo_format() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);

        // Example optimized parameters (3 filters)
        // Note: Frequency parameters must be in log10 scale
        let x = vec![
            500.0f64.log10(),
            2.0,
            -3.0, // Filter 1
            1000.0f64.log10(),
            5.0,
            2.0, // Filter 2
            3000.0f64.log10(),
            3.0,
            -1.0, // Filter 3
        ];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("speaker-flat filters should save");

        // Verify file was created
        let apo_path = output_path.parent().unwrap().join("iir-autoeq-flat.txt");
        assert!(apo_path.exists());

        // Verify content
        let content = fs::read_to_string(&apo_path).unwrap();
        assert!(content.contains("AutoEQ"));
        let filters = parse_apo_filters(&content).expect("generated APO must parse");
        assert_eq!(filters.len(), 3);
        for (actual, (index, freq_hz, gain_db, q)) in filters.iter().zip([
            (1, 500.0, -3.0, 2.0),
            (2, 1000.0, 2.0, 5.0),
            (3, 3000.0, -1.0, 3.0),
        ]) {
            assert_eq!(actual.index, index);
            assert_eq!(actual.kind, "PK");
            assert!(
                (actual.freq_hz - freq_hz).abs() < 1e-9,
                "frequency: actual={}, expected={freq_hz}",
                actual.freq_hz
            );
            assert!(
                (actual.gain_db - gain_db).abs() < 1e-9,
                "gain: actual={}, expected={gain_db}",
                actual.gain_db
            );
            assert!(
                (actual.q - q).abs() < 1e-9,
                "Q: actual={}, expected={q}",
                actual.q
            );
        }
    }

    #[tokio::test]
    async fn test_save_peq_to_file_score_loss() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-score"]);

        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerScore, None)
            .await
            .expect("speaker-score filters should save");

        // Verify filename for score loss
        let apo_path = output_path.parent().unwrap().join("iir-autoeq-score.txt");
        assert!(apo_path.exists());
        let content = fs::read_to_string(&apo_path).unwrap();
        let filters = parse_apo_filters(&content).expect("generated APO must parse");
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].kind, "PK");
        assert!(
            (filters[0].freq_hz - 500.0).abs() < 1e-9,
            "frequency: actual={}, expected=500",
            filters[0].freq_hz
        );
        assert!((filters[0].gain_db + 2.0).abs() < 1e-9);
        assert!((filters[0].q - 2.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_save_peq_creates_multiple_formats() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);

        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("all configured filter formats should save");

        // Check APO format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.txt")
                .exists()
        );

        // Check RME format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.tmreq")
                .exists()
        );

        // Check Apple format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.aupreset")
                .exists()
        );
    }

    #[tokio::test]
    async fn test_save_peq_with_pareto_export_writes_sidecar() {
        use crate::autoeq_command::save::ParetoExport;
        use autoeq::optim::pareto::ParetoFilter;

        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);
        let x = vec![500.0f64.log10(), 2.0, -2.0];

        let filters = vec![
            ParetoFilter {
                params: x.clone(),
                flatness_loss: 1.5,
                score_loss: Some(0.7),
                num_filters: 1,
                converged: true,
            },
            ParetoFilter {
                params: x.clone(),
                flatness_loss: 1.2,
                score_loss: None,
                num_filters: 2,
                converged: false,
            },
        ];
        let export = ParetoExport::from_pareto_filters(
            &filters,
            vec!["flatness_loss".to_string(), "score_loss".to_string()],
            "fewest filters within tolerance",
            0,
            vec![
                vec![("mic-1".to_string(), 1.4), ("mic-2".to_string(), 1.6)],
                vec![("mic-1".to_string(), 1.1)],
            ],
        );

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, Some(&export))
            .await
            .expect("preset with pareto export should save");

        let sidecar = output_path
            .parent()
            .unwrap()
            .join("iir-autoeq-flat-pareto.json");
        assert!(sidecar.exists(), "pareto sidecar must sit next to the preset");
        let parsed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(
            parsed["objective_labels"],
            serde_json::json!(["flatness_loss", "score_loss"])
        );
        assert_eq!(
            parsed["selection_policy"],
            serde_json::json!("fewest filters within tolerance")
        );
        assert_eq!(parsed["selected_index"], serde_json::json!(0));
        assert_eq!(parsed["candidates"][0]["objectives"], serde_json::json!([1.5, 0.7]));
        assert!(parsed["candidates"][1]["objectives"][1].is_null());
        assert_eq!(
            parsed["candidates"][0]["per_measurement_losses"],
            serde_json::json!([[ "mic-1", 1.4 ], [ "mic-2", 1.6 ]])
        );
        assert_eq!(parsed["candidates"][1]["num_filters"], serde_json::json!(2));
    }

    #[tokio::test]
    async fn test_save_peq_without_pareto_writes_no_sidecar() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);
        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("preset without pareto export should save");

        assert!(
            !output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat-pareto.json")
                .exists(),
            "no sidecar expected without pareto metadata"
        );
    }

    #[test]
    fn test_apo_roundtrip_gap_small_for_integer_hz_filters() {
        use crate::autoeq_command::save::{APO_ROUNDTRIP_WARN_THRESHOLD, apo_roundtrip_objective_gap};
        use autoeq::PeqModel;
        use autoeq::optim::{ObjectiveData, ObjectiveDataBuilder};
        use ndarray::Array1;

        let freqs = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let deviation = Array1::from_vec(vec![2.0, 1.5, 1.0, 1.2, 0.8]);
        let target = Array1::zeros(freqs.len());
        let objective: ObjectiveData =
            ObjectiveDataBuilder::new(freqs, target, deviation, 48000.0, PeqModel::Pk, LossType::SpeakerFlat)
                .min_spacing_oct(0.1)
                .max_db(10.0)
                .min_db(-10.0)
                .freq_range(20.0, 20000.0)
                .smoothing(false, 3)
                .build()
                .expect("valid test objective data");

        // Exact integer-Hz centers: serialization is (near-)lossless.
        let x = vec![
            500.0f64.log10(),
            2.0,
            -3.0,
            1000.0f64.log10(),
            5.0,
            2.0,
        ];
        let gap = apo_roundtrip_objective_gap(&x, 48000.0, PeqModel::Pk, &objective)
            .expect("finite objectives must produce a gap");
        assert!(
            gap < APO_ROUNDTRIP_WARN_THRESHOLD,
            "integer-Hz filters must stay under the warn threshold, got {gap:.3e}"
        );
    }

    #[test]
    fn test_apo_roundtrip_gap_finite_for_fractional_low_freq_high_q() {
        use crate::autoeq_command::save::apo_roundtrip_objective_gap;
        use autoeq::PeqModel;
        use autoeq::optim::{ObjectiveData, ObjectiveDataBuilder};
        use ndarray::Array1;

        let freqs = Array1::from_vec(vec![20.0, 30.0, 45.0, 60.0, 90.0, 120.0]);
        let deviation = Array1::from_vec(vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.8]);
        let target = Array1::zeros(freqs.len());
        let objective: ObjectiveData =
            ObjectiveDataBuilder::new(freqs, target, deviation, 48000.0, PeqModel::Pk, LossType::SpeakerFlat)
                .min_spacing_oct(0.01)
                .max_db(10.0)
                .min_db(-10.0)
                .freq_range(20.0, 20000.0)
                .smoothing(false, 3)
                .build()
                .expect("valid test objective data");

        // Fractional center with high Q: worst case for integer-Hz rounding.
        let x = vec![31.7f64.log10(), 12.0, -6.0];
        let gap = apo_roundtrip_objective_gap(&x, 48000.0, PeqModel::Pk, &objective)
            .expect("finite objectives must produce a gap");
        assert!(
            gap.is_finite(),
            "low-frequency/high-Q rounding gap must stay measurable, got {gap:?}"
        );
    }
}
