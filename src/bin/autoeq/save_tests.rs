#[cfg(test)]
#[path = "../../../tests/common/apo.rs"]
mod apo;

#[cfg(test)]
mod tests {
    use super::apo::parse_apo_filters;
    use crate::save::save_peq_to_file;
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

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat)
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

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerScore)
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

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat)
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
}
