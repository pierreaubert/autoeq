#[cfg(test)]
mod tests {
    use crate::load::load_and_prepare;
    use autoeq::cli::Args;
    use clap::Parser;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_load_and_prepare_with_csv() {
        let temp_dir = TempDir::new().unwrap();

        // Create test CSV file
        let csv_content = "freq,spl\n20,75.0\n50,78.0\n100,80.0\n200,82.0\n500,80.0\n1000,78.0\n2000,75.0\n5000,72.0\n10000,70.0\n20000,68.0\n";
        let csv_path = temp_dir.path().join("test.csv");
        fs::write(&csv_path, csv_content).unwrap();

        let args = Args::parse_from([
            "autoeq-test",
            "--curve",
            &csv_path.to_string_lossy(),
            "--num-filters",
            "5",
        ]);

        let (freq, input, target, deviation, spin) = load_and_prepare(&args)
            .await
            .expect("CSV measurement should load and prepare");

        // Verify frequency grid
        assert!(freq.len() >= 100);
        assert!(freq[0] >= 20.0);
        assert!(freq[freq.len() - 1] <= 20000.0 + 1e-5);

        // Verify curves have same length
        assert_eq!(input.freq.len(), freq.len());
        assert_eq!(target.freq.len(), freq.len());
        assert_eq!(deviation.freq.len(), freq.len());
        assert_eq!(input.freq, freq);
        assert_eq!(target.freq, freq);
        assert_eq!(deviation.freq, freq);
        assert!(freq.iter().all(|value| value.is_finite()));
        assert!(freq.windows(2).into_iter().all(|pair| pair[0] < pair[1]));
        for ((input_spl, target_spl), deviation_spl) in input
            .spl
            .iter()
            .zip(target.spl.iter())
            .zip(deviation.spl.iter())
        {
            assert!((deviation_spl - (target_spl - input_spl)).abs() < 1e-9);
        }

        // Verify spin data is None for simple CSV
        assert!(spin.is_none());
    }

    #[tokio::test]
    async fn test_load_and_prepare_headphone_mode() {
        let temp_dir = TempDir::new().unwrap();

        // Create test CSV
        let csv_content = "freq,spl\n20,80.0\n100,75.0\n1000,70.0\n10000,65.0\n20000,60.0\n";
        let csv_path = temp_dir.path().join("headphone.csv");
        fs::write(&csv_path, csv_content).unwrap();

        let args = Args::parse_from([
            "autoeq-test",
            "--curve",
            &csv_path.to_string_lossy(),
            "--loss",
            "headphone-flat",
            "--num-filters",
            "8",
        ]);

        let (freq, input, target, deviation, spin) = load_and_prepare(&args)
            .await
            .expect("headphone measurement should load and prepare");

        // Headphone mode should use 120 points
        assert_eq!(input.freq.len(), 120);
        assert_eq!(input.freq, freq);
        assert_eq!(target.freq, freq);
        assert_eq!(deviation.freq, freq);
        assert!(input.spl.iter().all(|value| value.is_finite()));
        assert!(target.spl.iter().all(|value| value.is_finite()));
        assert!(deviation.spl.iter().all(|value| value.is_finite()));
        assert!(spin.is_none());
    }
}
