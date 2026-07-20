//! Deterministically upgrade a measurement-provenance sidecar to the current
//! interchange schema.
//!
//! Usage: `migrate-provenance <input.provenance.json> [output.provenance.json]`

use autoeq_measurements::{PROVENANCE_SCHEMA_VERSION, read_sidecar_file};
use std::path::PathBuf;

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf), String> {
    if !(2..=3).contains(&args.len()) {
        let program = args
            .first()
            .map(String::as_str)
            .unwrap_or("migrate-provenance");
        return Err(format!(
            "Usage: {program} <input.provenance.json> [output.provenance.json]"
        ));
    }
    let input = PathBuf::from(&args[1]);
    let output = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| input.clone());
    Ok((input, output))
}

fn write_migrated_sidecar(
    input: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let record = read_sidecar_file(input)?;
    let encoded = serde_json::to_vec_pretty(&record)?;
    if input == output {
        let mut backup = input.to_path_buf();
        backup.set_extension(format!(
            "{}.bak",
            input
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("json")
        ));
        std::fs::copy(input, &backup)?;
        println!("Backup: {}", backup.display());
    }
    std::fs::write(output, encoded)?;
    println!(
        "Migrated {} to provenance schema v{}: {}",
        input.display(),
        PROVENANCE_SCHEMA_VERSION,
        output.display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let (input, output) = parse_args(&args).map_err(std::io::Error::other)?;
    write_migrated_sidecar(&input, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_measurements::{Curve, MeasurementRecord};
    use ndarray::Array1;
    use tempfile::tempdir;

    #[test]
    fn defaults_output_to_input() {
        let (input, output) = parse_args(&["tool".into(), "sidecar.json".into()]).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn migration_upgrades_v0_and_keeps_a_backup() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("measurement.provenance.json");
        let mut record = MeasurementRecord::legacy(Curve {
            freq: Array1::from_vec(vec![20.0, 1_000.0]),
            spl: Array1::from_vec(vec![0.0, 1.0]),
            ..Default::default()
        })
        .unwrap();
        record.provenance.schema_version = 0;
        record.provenance.schema.clear();
        std::fs::write(&input, serde_json::to_vec(&record).unwrap()).unwrap();

        write_migrated_sidecar(&input, &input).unwrap();
        let migrated = read_sidecar_file(&input).unwrap();
        assert_eq!(
            migrated.provenance.schema_version,
            PROVENANCE_SCHEMA_VERSION
        );
        assert!(dir.path().join("measurement.provenance.json.bak").exists());
    }
}
