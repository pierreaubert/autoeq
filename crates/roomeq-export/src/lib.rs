//! Experimental exporters for the canonical `roomeq-model` DSP graph.
//!
//! Production RoomEQ formats remain in the root `autoeq` crate while they are
//! migrated. These exporters validate the graph and never emit an empty
//! placeholder artifact.

#![forbid(unsafe_code)]

mod format;
pub use format::ExternalExportFormat;

use roomeq_model::{DspGraph, Plugin, ProvenanceConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    EqualizerApo,
}

/// Privacy profile applied to a packaged provenance manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestRedaction {
    Private,
    Shareable,
    Anonymous,
}

/// Sidecar emitted alongside a RoomEQ DSP export.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExportProvenanceManifest {
    pub schema: String,
    pub schema_version: u32,
    pub export_format: String,
    pub graph_sha256: String,
    /// Hash of the bytes delivered to the target renderer.  This is distinct
    /// from the graph hash because target formatting can change independently.
    pub export_sha256: String,
    /// Named package members (routing/config/filter renderings) and their
    /// byte-level identities.  The primary export is always present.
    #[serde(default)]
    pub artifact_hashes: BTreeMap<String, String>,
    pub measurement_hashes: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurement_sidecars: Option<BTreeMap<String, PathBuf>>,
}

pub fn export(graph: &DspGraph, format: ExportFormat) -> Result<String, String> {
    graph.validate()?;
    match format {
        ExportFormat::Json => serde_json::to_string_pretty(graph).map_err(|e| e.to_string()),
        ExportFormat::EqualizerApo => export_equalizer_apo(graph),
    }
}

/// Render an export and its independent provenance manifest. The measurement
/// content hashes stay stable across redaction; only private sidecar locations
/// are removed for shareable/anonymous packages.
pub fn export_with_provenance_manifest(
    graph: &DspGraph,
    format: ExportFormat,
    provenance: &ProvenanceConfig,
    redaction: ManifestRedaction,
) -> Result<(String, ExportProvenanceManifest), String> {
    let content = export(graph, format)?;
    let graph_json = serde_json::to_vec(graph).map_err(|error| error.to_string())?;
    let export_sha256 = hash_bytes(content.as_bytes());
    let artifact_hashes = BTreeMap::from([("primary_export".into(), export_sha256.clone())]);
    let measurement_hashes = provenance
        .measurements
        .iter()
        .map(|(name, reference)| (name.clone(), reference.content_hash.clone()))
        .collect();
    let measurement_sidecars = matches!(redaction, ManifestRedaction::Private).then(|| {
        provenance
            .measurements
            .iter()
            .filter_map(|(name, reference)| {
                reference
                    .sidecar_path
                    .as_ref()
                    .map(|path| (name.clone(), path.clone()))
            })
            .collect()
    });
    Ok((
        content,
        ExportProvenanceManifest {
            schema: "autoeq.roomeq-export-provenance".into(),
            schema_version: 1,
            export_format: format_name(format).into(),
            graph_sha256: hash_bytes(&graph_json),
            export_sha256,
            artifact_hashes,
            measurement_hashes,
            measurement_sidecars,
        },
    ))
}

/// Write the primary export and a `.provenance.json` sidecar as one package.
pub fn write_export_package(
    graph: &DspGraph,
    format: ExportFormat,
    output_path: &Path,
    provenance: &ProvenanceConfig,
    redaction: ManifestRedaction,
) -> Result<PathBuf, String> {
    let (content, manifest) =
        export_with_provenance_manifest(graph, format, provenance, redaction)?;
    std::fs::write(output_path, content).map_err(|error| error.to_string())?;
    let manifest_path = PathBuf::from(format!("{}.provenance.json", output_path.display()));
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(manifest_path)
}

fn format_name(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Json => "json",
        ExportFormat::EqualizerApo => "equalizer_apo",
    }
}

fn hash_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn export_equalizer_apo(graph: &DspGraph) -> Result<String, String> {
    let mut out = format!("# RoomEQ export v{}\n", graph.version);
    for (channel, chain) in &graph.channels {
        out.push_str(&format!("\nChannel: {channel}\n"));
        for plugin in &chain.plugins {
            write_plugin(&mut out, plugin)?;
        }
    }
    Ok(out)
}

fn write_plugin(out: &mut String, plugin: &Plugin) -> Result<(), String> {
    match plugin.kind.as_str() {
        "gain" => {
            let gain = plugin
                .parameters
                .get("gain_db")
                .and_then(|v| v.as_f64())
                .ok_or("gain plugin requires gain_db")?;
            out.push_str(&format!("Preamp: {gain:+.2} dB\n"));
        }
        "eq" => {
            let p = &plugin.parameters;
            let freq = p
                .get("freq")
                .and_then(|v| v.as_f64())
                .ok_or("eq plugin requires freq")?;
            let gain = p.get("gain_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let q = p.get("q").and_then(|v| v.as_f64()).unwrap_or(1.0);
            out.push_str(&format!(
                "Filter: ON PK Fc {freq:.2} Hz Gain {gain:+.2} dB Q {q:.4}\n"
            ));
        }
        "delay" => {
            let delay = plugin
                .parameters
                .get("delay_ms")
                .and_then(|v| v.as_f64())
                .ok_or("delay plugin requires delay_ms")?;
            out.push_str(&format!("Delay: {delay:.3} ms\n"));
        }
        other => return Err(format!("unsupported plugin kind '{other}'")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_graph_for_every_format() {
        let graph = DspGraph::new("1");
        for format in [ExportFormat::Json, ExportFormat::EqualizerApo] {
            let error = export(&graph, format).expect_err("empty graph must not export");
            assert!(
                error.contains("at least one channel"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn exports_json_and_apo() {
        let mut graph = DspGraph::new("1");
        graph.add_channel(
            "L",
            vec![Plugin {
                kind: "gain".into(),
                parameters: serde_json::json!({"gain_db": -3.0}),
            }],
        );
        assert!(
            export(&graph, ExportFormat::Json)
                .unwrap()
                .contains("channels")
        );
        assert!(
            export(&graph, ExportFormat::EqualizerApo)
                .unwrap()
                .contains("Preamp")
        );
    }

    #[test]
    fn provenance_manifest_preserves_hashes_but_redacts_sidecars() {
        let mut graph = DspGraph::new("1");
        graph.add_channel(
            "L",
            vec![Plugin {
                kind: "gain".into(),
                parameters: serde_json::json!({"gain_db": -3.0}),
            }],
        );
        let mut provenance = ProvenanceConfig::default();
        provenance.measurements.insert(
            "L".into(),
            roomeq_model::MeasurementProvenanceReference {
                record_id: "record-l".into(),
                content_hash: "a".repeat(64),
                schema_version: 1,
                sidecar_path: Some("private/measurement.provenance.json".into()),
            },
        );
        let (_, private) = export_with_provenance_manifest(
            &graph,
            ExportFormat::Json,
            &provenance,
            ManifestRedaction::Private,
        )
        .unwrap();
        let (_, shared) = export_with_provenance_manifest(
            &graph,
            ExportFormat::Json,
            &provenance,
            ManifestRedaction::Shareable,
        )
        .unwrap();
        assert_eq!(private.measurement_hashes, shared.measurement_hashes);
        assert_eq!(
            private.artifact_hashes["primary_export"],
            private.export_sha256
        );
        assert!(private.measurement_sidecars.is_some());
        assert!(shared.measurement_sidecars.is_none());
    }
}
