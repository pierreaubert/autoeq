use super::hash::sha256_hex;
use anyhow::Context;
use roomeq_model::{DspGraph, PluginConfigWrapper};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Explicit resource supplied by a workflow adapter for a convolution path in
/// the canonical graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionResource {
    pub reference: String,
    pub bytes: Arc<[u8]>,
}

impl ConvolutionResource {
    pub fn sha256(&self) -> String {
        sha256_hex(&self.bytes)
    }
}

pub(crate) fn validate_final_convolution_identity(graph: &DspGraph, resources: &[ConvolutionResource]) -> anyhow::Result<()> {
    let references = checked_convolution_resource_references(graph)?;
    let Some(inventory) = graph.metadata.as_ref().and_then(|metadata| metadata.final_convolution_sha256.as_ref()) else {
        return Ok(()); // Legacy/manual graphs have no final-workflow binding.
    };
    anyhow::ensure!(references.iter().collect::<BTreeSet<_>>() == inventory.keys().collect::<BTreeSet<_>>(),
        "final convolution inventory does not match graph references");
    let supplied = resource_map(resources)?;
    for reference in references {
        let expected = inventory[&reference].as_ref().with_context(|| format!("final convolution artifact '{reference}' is unbound"))?;
        let bytes = supplied.get(reference.as_str()).with_context(|| format!("missing final convolution artifact '{reference}'"))?;
        anyhow::ensure!(sha256_hex(bytes) == *expected, "final convolution artifact '{reference}' changed since workflow completion");
    }
    Ok(())
}

/// One deterministic export package member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPackageMember {
    pub relative_path: PathBuf,
    pub bytes: Arc<[u8]>,
    pub sha256: String,
}

impl ExportPackageMember {
    pub fn new(
        relative_path: impl Into<PathBuf>,
        bytes: impl Into<Arc<[u8]>>,
    ) -> anyhow::Result<Self> {
        let relative_path = relative_path.into();
        validate_member_path(&relative_path)?;
        let bytes = bytes.into();
        let sha256 = sha256_hex(&bytes);
        Ok(Self {
            relative_path,
            bytes,
            sha256,
        })
    }
}

/// Complete in-memory export package. Persistence belongs to workflow or
/// artifact-store adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPackage {
    pub members: Vec<ExportPackageMember>,
}

impl ExportPackage {
    /// Revalidate public members before persistence. The package may have been
    /// modified since construction; a filename/hash pair is not proof of bytes.
    pub fn validate_integrity(&self) -> anyhow::Result<()> {
        let mut paths = BTreeSet::new();
        for member in &self.members {
            validate_member_path(&member.relative_path)?;
            anyhow::ensure!(paths.insert(&member.relative_path), "duplicate export package member");
            anyhow::ensure!(sha256_hex(&member.bytes) == member.sha256,
                "export package content hash mismatch for '{}'", member.relative_path.display());
        }
        Ok(())
    }
    /// Read the typed delay contract from the exact packaged CamillaDSP
    /// artifact, including its backend-only additional latency. Other formats
    /// return None. Malformed or ambiguous declarations fail explicitly.
    pub fn camilladsp_delay_realization(&self) -> anyhow::Result<Option<crate::CamillaDspDelayRealization>> {
        let mut report = None;
        for member in &self.members {
            let Ok(text) = std::str::from_utf8(&member.bytes) else { continue };
            for line in text.lines() {
                if let Some(json) = line.strip_prefix("# roomeq_delay_realization: ") {
                    anyhow::ensure!(report.is_none(), "multiple CamillaDSP delay reports in one package");
                    report = Some(serde_json::from_str(json).context("invalid packaged CamillaDSP delay report")?);
                }
            }
        }
        Ok(report)
    }

    pub fn new(mut members: Vec<ExportPackageMember>) -> anyhow::Result<Self> {
        members.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        for pair in members.windows(2) {
            if pair[0].relative_path == pair[1].relative_path {
                anyhow::bail!(
                    "export package contains duplicate member '{}'",
                    pair[0].relative_path.display()
                );
            }
        }
        Ok(Self { members })
    }

    pub fn member(&self, relative_path: &Path) -> Option<&ExportPackageMember> {
        self.members
            .iter()
            .find(|member| member.relative_path == relative_path)
    }
}

/// Best-effort reference discovery for legacy inspection callers.
/// Finalization and export must use `checked_convolution_resource_references`.
pub fn convolution_resource_references(graph: &DspGraph) -> Vec<String> {
    let mut references = BTreeSet::new();
    collect_references(&graph.global_plugins, &mut references);
    for chain in graph.channels.values() {
        collect_references(&chain.plugins, &mut references);
        if let Some(drivers) = &chain.drivers {
            for driver in drivers {
                collect_references(&driver.plugins, &mut references);
            }
        }
    }
    references.into_iter().collect()
}

/// Collect every declared FIR resource, rejecting malformed stages rather than
/// silently omitting them from the final playback inventory.
pub fn checked_convolution_resource_references(graph: &DspGraph) -> anyhow::Result<Vec<String>> {
    let validate = |plugins: &[PluginConfigWrapper], owner: &str| -> anyhow::Result<()> {
        for (index, plugin) in plugins.iter().enumerate() {
            if plugin.plugin_type != "convolution" {
                continue;
            }
            let reference = plugin.parameters.get("ir_file")
                .and_then(serde_json::Value::as_str)
                .with_context(|| format!("{owner} convolution stage {index} requires string field 'ir_file'"))?;
            anyhow::ensure!(!reference.trim().is_empty() && !reference.contains('\0'),
                "{owner} convolution stage {index} requires a nonblank, NUL-free 'ir_file'");
        }
        Ok(())
    };
    validate(&graph.global_plugins, "global")?;
    for (name, chain) in &graph.channels {
        validate(&chain.plugins, &format!("channel '{name}'"))?;
        for driver in chain.drivers.iter().flatten() {
            validate(&driver.plugins, &format!("channel '{name}' driver '{}'", driver.name))?;
        }
    }
    Ok(convolution_resource_references(graph))
}

/// Rewrite convolution references to package-local member names and return
/// the sidecar members without touching the filesystem.
pub fn package_convolution_sidecars(
    graph: &DspGraph,
    resources: &[ConvolutionResource],
    occupied_names: &BTreeSet<String>,
    reusable_names: &HashMap<String, String>,
) -> anyhow::Result<(DspGraph, Vec<ExportPackageMember>)> {
    validate_final_convolution_identity(graph, resources)?;
    let resources = resource_map(resources)?;
    let mut packaged_by_reference = HashMap::new();
    let mut assigned = occupied_names.clone();
    let mut members = BTreeMap::<String, ExportPackageMember>::new();

    // Group references by content hash so each unique asset is streamed,
    // hashed, and copied once no matter how many plugins reference it. The
    // byte comparison inside a hash bucket only guards against collisions.
    let mut content_groups = Vec::<(String, Arc<[u8]>, Vec<String>)>::new();
    let mut group_by_hash = HashMap::<String, usize>::new();
    for reference in checked_convolution_resource_references(graph)? {
        let bytes = resources.get(reference.as_str()).copied().ok_or_else(|| {
            anyhow::anyhow!("missing explicit convolution resource '{reference}'")
        })?;
        let hash = sha256_hex(bytes);
        let group = group_by_hash.get(&hash).copied().filter(|index| {
            content_groups
                .get(*index)
                .is_some_and(|(_, packaged, _)| packaged.as_ref() == bytes.as_ref())
        });
        let group = match group {
            Some(index) => index,
            None => {
                let index = content_groups.len();
                group_by_hash.insert(hash.clone(), index);
                content_groups.push((hash, Arc::clone(bytes), Vec::new()));
                index
            }
        };
        content_groups[group].2.push(reference);
    }
    for (hash, bytes, references) in content_groups {
        let reusable = references
            .iter()
            .filter_map(|reference| reusable_names.get(reference))
            .min()
            .cloned();
        let packaged_name = if let Some(existing) = reusable {
            if !assigned.contains(&existing) {
                anyhow::bail!(
                    "reusable convolution member '{existing}' is not an occupied destination"
                );
            }
            existing
        } else {
            let preferred = Path::new(&references[0])
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
                .unwrap_or("room_eq_ir.wav");
            let packaged_name = unique_member_name(preferred, &assigned);
            assigned.insert(packaged_name.clone());
            // Reuse the grouping hash: the sidecar bytes are hashed once per
            // unique asset instead of once per reference plus once per member.
            validate_member_path(Path::new(&packaged_name))?;
            members.insert(
                packaged_name.clone(),
                ExportPackageMember {
                    relative_path: packaged_name.clone().into(),
                    bytes: Arc::clone(&bytes),
                    sha256: hash,
                },
            );
            packaged_name
        };
        for reference in references {
            packaged_by_reference.insert(reference, packaged_name.clone());
        }
    }

    let mut graph = graph.clone();

    rewrite_plugins(&mut graph.global_plugins, &packaged_by_reference)?;
    for chain in graph.channels.values_mut() {
        rewrite_plugins(&mut chain.plugins, &packaged_by_reference)?;
        if let Some(drivers) = chain.drivers.as_mut() {
            for driver in drivers {
                rewrite_plugins(&mut driver.plugins, &packaged_by_reference)?;
            }
        }
    }

    if let Some(inventory) = graph.metadata.as_mut().and_then(|metadata| metadata.final_convolution_sha256.as_mut()) {
        *inventory = inventory.iter().map(|(reference, hash)|
            (packaged_by_reference[reference].clone(), hash.clone())).collect();
    }
    Ok((graph, members.into_values().collect()))
}

pub(super) fn resource_map(
    resources: &[ConvolutionResource],
) -> anyhow::Result<HashMap<&str, &Arc<[u8]>>> {
    let mut by_reference = HashMap::new();
    for resource in resources {
        if resource.reference.trim().is_empty() {
            anyhow::bail!("convolution resource reference must not be empty");
        }
        if let Some(previous) = by_reference.insert(resource.reference.as_str(), &resource.bytes)
            && previous.as_ref() != resource.bytes.as_ref()
        {
            anyhow::bail!(
                "convolution resource '{}' was supplied with conflicting bytes",
                resource.reference
            );
        }
    }
    Ok(by_reference)
}

fn rewrite_plugins(
    plugins: &mut [PluginConfigWrapper],
    packaged_by_reference: &HashMap<String, String>,
) -> anyhow::Result<()> {
    for plugin in plugins {
        if plugin.plugin_type != "convolution" {
            continue;
        }
        let reference = plugin
            .parameters
            .get("ir_file")
            .and_then(serde_json::Value::as_str)
            .context("convolution plugin requires string field 'ir_file'")?
            .to_string();
        let packaged_name = packaged_by_reference
            .get(&reference)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("missing explicit convolution resource '{reference}'")
            })?;
        let parameters = plugin
            .parameters
            .as_object_mut()
            .context("convolution plugin parameters must be a JSON object")?;
        parameters.insert("ir_file".to_string(), serde_json::json!(packaged_name));
    }
    Ok(())
}

fn collect_references(plugins: &[PluginConfigWrapper], references: &mut BTreeSet<String>) {
    for plugin in plugins {
        if plugin.plugin_type == "convolution"
            && let Some(reference) = plugin
                .parameters
                .get("ir_file")
                .and_then(serde_json::Value::as_str)
        {
            references.insert(reference.to_string());
        }
    }
}

fn unique_member_name(preferred: &str, assigned: &BTreeSet<String>) -> String {
    if !assigned.contains(preferred) {
        return preferred.to_string();
    }
    let preferred = Path::new(preferred);
    let stem = preferred
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("room_eq_ir");
    let extension = preferred
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    for suffix in 2_u64..=u64::MAX {
        let candidate = format!("{stem}_{suffix:03}{extension}");
        if !assigned.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("u64 package-member namespace exhausted")
}

fn validate_member_path(path: &Path) -> anyhow::Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!(
            "export package member '{}' must be a safe relative path",
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::ChannelDspChain;

    fn convolution_chain(name: &str, ir_file: &str) -> (String, ChannelDspChain) {
        (
            name.to_string(),
            ChannelDspChain {
                channel: name.to_string(),
                plugins: vec![PluginConfigWrapper {
                    plugin_type: "convolution".to_string(),
                    parameters: serde_json::json!({"ir_file": ir_file}),
                }],
                drivers: None,
                initial_curve: None,
                final_curve: None,
                eq_response: None,
                target_curve: None,
                pre_ir: None,
                post_ir: None,
                fir_temporal_masking: None,
                direct_early_late_correction: None,
            },
        )
    }

    #[test]
    fn malformed_convolution_stages_cannot_disappear_from_resource_inventory() {
        for scope in ["global", "channel", "driver"] {
            for parameters in [
                serde_json::json!({}),
                serde_json::json!({"ir_file": null}),
                serde_json::json!({"ir_file": 42}),
                serde_json::json!({"ir_file": ""}),
                serde_json::json!({"ir_file": "  "}),
                serde_json::json!({"ir_file": "bad\u{0}.wav"}),
            ] {
                let plugin = PluginConfigWrapper {
                    plugin_type: "convolution".into(), parameters,
                };
                let (name, mut chain) = convolution_chain("left", "valid.wav");
                chain.plugins.clear();
                let mut graph = DspGraph {
                    version: "1.3.0".into(), global_plugins: Vec::new(),
                    channels: HashMap::new(), metadata: None,
                    deployed_source_curves: Default::default(),
                };
                match scope {
                    "global" => graph.global_plugins.push(plugin),
                    "channel" => chain.plugins.push(plugin),
                    _ => chain.drivers = Some(vec![roomeq_model::DriverDspChain {
                        name: "woofer".into(), index: 0, plugins: vec![plugin],
                        initial_curve: None,
                    }]),
                }
                graph.channels.insert(name, chain);
                let error = checked_convolution_resource_references(&graph).unwrap_err();
                assert!(error.to_string().contains(scope), "{error}");
                // Legacy graphs also must not package a malformed stage.
                assert!(package_convolution_sidecars(
                    &graph, &[], &BTreeSet::new(), &HashMap::new()
                ).is_err());
            }
        }
    }

    #[test]
    fn identical_sidecars_are_hashed_and_copied_once() {
        let shared: Arc<[u8]> = Arc::from(b"shared-impulse".as_slice());
        let other: Arc<[u8]> = Arc::from(b"other-impulse".as_slice());
        let graph = DspGraph {
            deployed_source_curves: Default::default(),
            version: "1.3.0".to_string(),
            global_plugins: Vec::new(),
            channels: HashMap::from([
                convolution_chain("left", "a.wav"),
                convolution_chain("right", "b.wav"),
                convolution_chain("center", "c.wav"),
            ]),
            metadata: None,
        };
        let resources = vec![
            ConvolutionResource {
                reference: "a.wav".to_string(),
                bytes: Arc::clone(&shared),
            },
            ConvolutionResource {
                reference: "b.wav".to_string(),
                bytes: Arc::clone(&shared),
            },
            ConvolutionResource {
                reference: "c.wav".to_string(),
                bytes: Arc::clone(&other),
            },
        ];

        let (packaged, members) =
            package_convolution_sidecars(&graph, &resources, &BTreeSet::new(), &HashMap::new())
                .unwrap();

        assert_eq!(members.len(), 2, "identical assets must share one sidecar");
        for member in &members {
            assert_eq!(member.sha256, sha256_hex(&member.bytes));
        }
        let rewritten: Vec<&str> = ["left", "right", "center"]
            .iter()
            .map(|channel| {
                packaged.channels[*channel].plugins[0]
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
                    .unwrap()
            })
            .collect();
        assert_eq!(rewritten[0], rewritten[1]);
        assert_ne!(rewritten[0], rewritten[2]);
    }
}
