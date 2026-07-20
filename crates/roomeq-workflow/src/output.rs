//! RoomEQ output persistence adapters.

use std::path::Path;

use roomeq_model::DspChainOutput;

/// Save a DSP chain as pretty-printed JSON.
pub fn save_dsp_chain(
    output: &DspChainOutput,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(output)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_pretty_json() {
        let directory = tempfile::TempDir::new().expect("temp output directory");
        let path = directory.path().join("dsp.json");
        let output = DspChainOutput {
            version: "1".to_string(),
            global_plugins: Vec::new(),
            channels: std::collections::HashMap::new(),
            metadata: None,
        };

        save_dsp_chain(&output, &path).expect("save DSP chain");

        let json = std::fs::read_to_string(path).expect("read DSP chain");
        assert!(json.contains("\n  \"version\": \"1\""));
    }
}
