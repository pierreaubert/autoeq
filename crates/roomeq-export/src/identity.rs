//! Canonical DSP identity for exported RoomEQ graphs (plan task X1).
//!
//! [`canonical_dsp_identity`] binds the exact delivered processing — sample
//! rate, ordered plugin parameters, routing, gains/delays, physical outputs,
//! and convolution content identities — to one stable hash. Workflow calls it
//! before packaging so a final report can never be bound to an earlier
//! candidate or to convolution bytes replaced after the report was produced.
//!
//! The DSP identity is not a byte or package identity: two equivalent
//! renderings may share a DSP identity while their package fingerprints
//! differ. Parse-back success alone is not playback proof, and no identity
//! here is a listening-benefit claim.

// Rust guideline compliant 2026-02-21

use super::hash::sha256_hex;
use super::package::{ConvolutionResource, ExportPackage};
use roomeq_model::{BassManagementRoutingGraph, ChannelDspChain, DspGraph, PluginConfigWrapper};
use serde_json::Value;
use std::collections::BTreeMap;

/// Version of the canonical DSP identity document.
pub const DSP_IDENTITY_VERSION: u32 = 1;

/// One convolution resource bound into a DSP identity by content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundConvolution {
    /// Graph-declared reference (`ir_file`).
    pub reference: String,
    /// SHA-256 of the exact resource bytes bound at identity time.
    pub sha256: String,
}

/// Canonical identity of the delivered DSP described by a graph.
///
/// The hash covers only realized processing. Explanatory metadata
/// (scores, timestamps, algorithm names, advisories), display curves,
/// the K4 decision ledger (which references the graph, so hashing it
/// would be circular), and the output schema version are excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct DspIdentity {
    /// Sample rate the identity was computed for, in Hz.
    pub sample_rate_hz: f64,
    /// Stable hash over the canonical DSP document (hex SHA-256).
    pub dsp_identity: String,
    /// Convolution content bound into the hash, sorted by reference.
    pub convolution: Vec<BoundConvolution>,
}

/// Per-target representability result for workflow handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportReadiness {
    /// Target the verdict applies to.
    pub format: super::ExportFormat,
    /// Whether the target can represent the graph without silent changes.
    pub supported: bool,
    /// Rejection reason when `supported` is false.
    pub reason: Option<String>,
}

/// Compute the canonical DSP identity of `graph` at `sample_rate_hz`.
///
/// Every convolution reference declared by the graph must have exactly one
/// matching entry in `resources`; the content hash of those bytes is part of
/// the hashed document. A missing resource is unavailable evidence, never a
/// bound identity. When the graph carries a final-workflow convolution
/// inventory (`final_convolution_sha256`), the supplied bytes must match it:
/// replaced or stale-candidate bytes fail instead of rebinding silently.
///
/// # Errors
///
/// Returns an error for a nonfinite or nonpositive sample rate, an invalid
/// graph, a missing convolution resource, or a final-inventory mismatch.
pub fn canonical_dsp_identity(
    graph: &DspGraph,
    sample_rate_hz: f64,
    resources: &[ConvolutionResource],
) -> anyhow::Result<DspIdentity> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        anyhow::bail!("DSP identity requires a finite positive sample rate");
    }
    graph.validate().map_err(anyhow::Error::msg)?;
    super::package::validate_final_convolution_identity(graph, resources)?;
    let supplied = super::package::resource_map(resources)?;
    let mut convolution = Vec::new();
    for reference in super::package::checked_convolution_resource_references(graph)? {
        let bytes = supplied.get(reference.as_str()).ok_or_else(|| {
            anyhow::anyhow!(
                "convolution resource '{reference}' is unavailable: cannot bind DSP identity"
            )
        })?;
        convolution.push(BoundConvolution {
            reference: reference.clone(),
            sha256: sha256_hex(bytes),
        });
    }
    convolution.sort_by(|left, right| left.reference.cmp(&right.reference));

    let mut channels = BTreeMap::new();
    for (name, chain) in &graph.channels {
        channels.insert(name.clone(), canonical_channel(chain));
    }
    let document = serde_json::json!({
        "format": "roomeq_dsp_identity",
        "version": DSP_IDENTITY_VERSION,
        "sample_rate_hz": sample_rate_hz,
        "global_plugins": canonical_plugins(&graph.global_plugins),
        "channels": channels.into_iter().map(|(name, value)| {
            let mut entry = BTreeMap::new();
            entry.insert("channel".to_string(), Value::String(name));
            entry.insert("chain".to_string(), value);
            Value::Object(entry.into_iter().collect())
        }).collect::<Vec<_>>(),
        "routing": canonical_routing(graph),
        "convolution": convolution.iter().map(|bound| {
            serde_json::json!({"reference": bound.reference, "sha256": bound.sha256})
        }).collect::<Vec<_>>(),
    });
    let bytes = serde_json::to_vec(&canonical_json(&document))?;
    Ok(DspIdentity {
        sample_rate_hz,
        dsp_identity: sha256_hex(&bytes),
        convolution,
    })
}

/// Fingerprint the exact packaged bytes of an export package.
///
/// This is a byte/package identity, deliberately distinct from
/// [`DspIdentity::dsp_identity`]: re-rendering the same DSP (different
/// decimal formatting, reordered sidecars, new timestamps in comments)
/// changes the fingerprint while the DSP identity is unchanged.
///
/// # Panics
///
/// Never panics; member hashes are recomputed from the stored bytes.
pub fn package_fingerprint(package: &ExportPackage) -> String {
    let mut entries: Vec<(String, String)> = package
        .members
        .iter()
        .map(|member| {
            (
                member.relative_path.to_string_lossy().into_owned(),
                sha256_hex(&member.bytes),
            )
        })
        .collect();
    entries.sort();
    let mut input = Vec::new();
    for (path, hash) in entries {
        input.extend_from_slice(path.as_bytes());
        input.push(0);
        input.extend_from_slice(hash.as_bytes());
        input.push(0);
    }
    sha256_hex(&input)
}

/// Report per-target representability for `formats` without rendering.
///
/// Unsupported targets fail explicitly (unsupported routing, required
/// limiters, unavailable convolution resources); this helper never drops,
/// rescales, or converts DSP silently. It performs no rendering and no I/O.
pub fn export_readiness(
    graph: &DspGraph,
    formats: &[super::ExportFormat],
) -> Vec<ExportReadiness> {
    formats
        .iter()
        .map(|format| match super::external_export_supported(graph, *format) {
            Ok(()) => ExportReadiness {
                format: *format,
                supported: true,
                reason: None,
            },
            Err(error) => ExportReadiness {
                format: *format,
                supported: false,
                reason: Some(error.to_string()),
            },
        })
        .collect()
}

/// Canonical JSON with object keys sorted recursively.
///
/// Arrays keep declared order (plugin order is DSP); objects hash
/// independently of insertion order so `HashMap` iteration order can never
/// leak into the identity.
fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), canonical_json(item)))
                .collect::<BTreeMap<String, Value>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonical_json).collect()),
        _ => value.clone(),
    }
}

fn canonical_plugins(plugins: &[PluginConfigWrapper]) -> Vec<Value> {
    plugins
        .iter()
        .map(|plugin| {
            let mut entry = BTreeMap::new();
            entry.insert(
                "type".to_string(),
                Value::String(plugin.plugin_type.clone()),
            );
            entry.insert(
                "parameters".to_string(),
                canonical_json(&plugin.parameters),
            );
            Value::Object(entry.into_iter().collect())
        })
        .collect()
}

fn canonical_channel(chain: &ChannelDspChain) -> Value {
    let mut drivers: Vec<(usize, String, Vec<Value>)> = chain
        .drivers
        .iter()
        .flatten()
        .map(|driver| {
            (
                driver.index,
                driver.name.clone(),
                canonical_plugins(&driver.plugins),
            )
        })
        .collect();
    drivers.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut entry = BTreeMap::new();
    entry.insert(
        "plugins".to_string(),
        Value::Array(canonical_plugins(&chain.plugins)),
    );
    entry.insert(
        "drivers".to_string(),
        Value::Array(
            drivers
                .into_iter()
                .map(|(index, name, plugins)| {
                    let mut driver = BTreeMap::new();
                    driver.insert(
                        "index".to_string(),
                        Value::Number(index.into()),
                    );
                    driver.insert("name".to_string(), Value::String(name));
                    driver.insert("plugins".to_string(), Value::Array(plugins));
                    Value::Object(driver.into_iter().collect())
                })
                .collect(),
        ),
    );
    Value::Object(entry.into_iter().collect())
}

/// Realized routing only: channel layout, ordered routes, matrix, and trims.
///
/// Selection evidence (candidate lists, advisories, stereo-selection basis)
/// is excluded: it explains how the route set was chosen, not what the
/// delivered processing is. The same exclusion applies to the K4 ledger and
/// to every display curve on the graph.
fn canonical_routing(graph: &DspGraph) -> Value {
    let routing: Option<&BassManagementRoutingGraph> = graph
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.bass_management.as_ref())
        .and_then(|report| report.routing_graph.as_ref());
    let Some(routing) = routing else {
        return Value::Null;
    };
    let routes: Vec<Value> = routing
        .routes
        .iter()
        .map(|route| {
            canonical_json(&serde_json::json!({
                "source_channel": route.source_channel,
                "source_index": route.source_index,
                "destination": route.destination,
                "destination_index": route.destination_index,
                "pre_chain_channel": route.pre_chain_channel,
                "post_chain_channel": route.post_chain_channel,
                "route_kind": route.route_kind,
                "crossover_type": route.crossover_type,
                "high_pass_hz": route.high_pass_hz,
                "low_pass_hz": route.low_pass_hz,
                "gain_db": route.gain_db,
                "gain_linear": route.gain_linear,
                "matrix_gain": route.matrix_gain,
                "delay_ms": route.delay_ms,
                "polarity_inverted": route.polarity_inverted,
                "group_id": route.group_id,
            }))
        })
        .collect();
    canonical_json(&serde_json::json!({
        "input_channels": routing.input_channels,
        "output_channels": routing.output_channels,
        "physical_sub_outputs": routing.physical_sub_outputs,
        "routes": routes,
        "matrix": routing.matrix,
        "input_trim_db": routing.input_trim_db,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let value = serde_json::json!({"b": 1, "a": {"d": 1, "c": 2}, "list": [{"y": 1, "x": 0}]});
        let canonical = canonical_json(&value);
        let text = serde_json::to_string(&canonical).unwrap();
        assert_eq!(text, r#"{"a":{"c":2,"d":1},"b":1,"list":[{"x":0,"y":1}]}"#);
    }

    #[test]
    fn package_fingerprint_ignores_member_order() {
        let first = crate::ExportPackageMember::new("a.txt", Vec::from(b"a")).unwrap();
        let second = crate::ExportPackageMember::new("b.txt", Vec::from(b"b")).unwrap();
        let forward =
            crate::ExportPackage::new(vec![first.clone(), second.clone()]).unwrap();
        let mut reversed = crate::ExportPackage::new(vec![second, first]).unwrap();
        reversed.members.reverse();
        assert_eq!(package_fingerprint(&forward), package_fingerprint(&reversed));
        let altered = crate::ExportPackage::new(vec![
            crate::ExportPackageMember::new("a.txt", Vec::from(b"altered")).unwrap(),
        ])
        .unwrap();
        assert_ne!(package_fingerprint(&forward), package_fingerprint(&altered));
    }
}
