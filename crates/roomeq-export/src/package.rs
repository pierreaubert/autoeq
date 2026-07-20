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

/// Return every convolution reference in deterministic order so a workflow
/// can load exactly the resources required by a graph.
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

/// Rewrite convolution references to package-local member names and return
/// the sidecar members without touching the filesystem.
pub fn package_convolution_sidecars(
    graph: &DspGraph,
    resources: &[ConvolutionResource],
    occupied_names: &BTreeSet<String>,
    reusable_names: &HashMap<String, String>,
) -> anyhow::Result<(DspGraph, Vec<ExportPackageMember>)> {
    let resources = resource_map(resources)?;
    let mut packaged_by_reference = HashMap::new();
    let mut assigned = occupied_names.clone();
    let mut members = BTreeMap::<String, ExportPackageMember>::new();

    let mut content_groups = Vec::<(Arc<[u8]>, Vec<String>)>::new();
    for reference in convolution_resource_references(graph) {
        let bytes = resources.get(reference.as_str()).copied().ok_or_else(|| {
            anyhow::anyhow!("missing explicit convolution resource '{reference}'")
        })?;
        if let Some((_, references)) = content_groups
            .iter_mut()
            .find(|(packaged, _)| packaged.as_ref() == bytes.as_ref())
        {
            references.push(reference);
        } else {
            content_groups.push((Arc::clone(bytes), vec![reference]));
        }
    }
    for (bytes, references) in content_groups {
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
            members.insert(
                packaged_name.clone(),
                ExportPackageMember::new(&packaged_name, Arc::clone(&bytes))?,
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
