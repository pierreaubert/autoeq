//! Refresh derived report pairs for a saved bundle without rerunning optimization.

use roomeq_workflow::{output_bundle, symmetric_report};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::PathBuf::from(std::env::args_os().nth(1).ok_or("provide DSP JSON path")?);
    let output = output_bundle::load_output_bundle(&path)?;
    let index_path = output_bundle::assets_dir_for(&path).join("measurements_index.json");
    let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    index["symmetric_pairs"] = symmetric_report::pairs(&output);
    std::fs::write(index_path, serde_json::to_vec_pretty(&index)?)?;
    println!("Refreshed symmetric report pairs; DSP JSON and filters unchanged.");
    Ok(())
}
