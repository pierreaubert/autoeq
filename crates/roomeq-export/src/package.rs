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

pub(crate) fn validate_final_convolution_identity(
    graph: &DspGraph,
    resources: &[ConvolutionResource],
) -> anyhow::Result<()> {
    let references = checked_convolution_resource_references(graph)?;
    let Some(inventory) = graph
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.final_convolution_sha256.as_ref())
    else {
        return Ok(()); // Legacy/manual graphs have no final-workflow binding.
    };
    anyhow::ensure!(
        references.iter().collect::<BTreeSet<_>>() == inventory.keys().collect::<BTreeSet<_>>(),
        "final convolution inventory does not match graph references"
    );
    let supplied = resource_map(resources)?;
    for reference in references {
        let expected = inventory[&reference]
            .as_ref()
            .with_context(|| format!("final convolution artifact '{reference}' is unbound"))?;
        let bytes = supplied
            .get(reference.as_str())
            .with_context(|| format!("missing final convolution artifact '{reference}'"))?;
        anyhow::ensure!(
            sha256_hex(bytes) == *expected,
            "final convolution artifact '{reference}' changed since workflow completion"
        );
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
            anyhow::ensure!(
                paths.insert(portable_member_key(&member.relative_path)?),
                "duplicate or case-insensitive export package member collision"
            );
            anyhow::ensure!(
                sha256_hex(&member.bytes) == member.sha256,
                "export package content hash mismatch for '{}'",
                member.relative_path.display()
            );
        }
        Ok(())
    }
    /// Read the typed delay contract from the exact packaged CamillaDSP
    /// artifact, including its backend-only additional latency. Other formats
    /// return None. Malformed or ambiguous declarations fail explicitly.
    pub fn camilladsp_delay_realization(
        &self,
    ) -> anyhow::Result<Option<crate::CamillaDspDelayRealization>> {
        let mut report = None;
        for member in &self.members {
            let Ok(text) = std::str::from_utf8(&member.bytes) else {
                continue;
            };
            for line in text.lines() {
                if let Some(json) = line.strip_prefix("# roomeq_delay_realization: ") {
                    anyhow::ensure!(
                        report.is_none(),
                        "multiple CamillaDSP delay reports in one package"
                    );
                    report = Some(
                        serde_json::from_str(json)
                            .context("invalid packaged CamillaDSP delay report")?,
                    );
                }
            }
        }
        Ok(report)
    }

    pub fn new(mut members: Vec<ExportPackageMember>) -> anyhow::Result<Self> {
        members.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        let mut paths = BTreeSet::new();
        for member in &members {
            validate_member_path(&member.relative_path)?;
            anyhow::ensure!(
                paths.insert(portable_member_key(&member.relative_path)?),
                "export package contains duplicate or case-insensitive member '{}'",
                member.relative_path.display()
            );
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
            let reference = plugin
                .parameters
                .get("ir_file")
                .and_then(serde_json::Value::as_str)
                .with_context(|| {
                    format!("{owner} convolution stage {index} requires string field 'ir_file'")
                })?;
            anyhow::ensure!(
                !reference.trim().is_empty() && !reference.contains('\0'),
                "{owner} convolution stage {index} requires a nonblank, NUL-free 'ir_file'"
            );
        }
        Ok(())
    };
    validate(&graph.global_plugins, "global")?;
    for (name, chain) in &graph.channels {
        validate(&chain.plugins, &format!("channel '{name}'"))?;
        for driver in chain.drivers.iter().flatten() {
            validate(
                &driver.plugins,
                &format!("channel '{name}' driver '{}'", driver.name),
            )?;
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
    validate_source_ledger(graph)?;
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
            validate_member_path(Path::new(&existing))?;
            if !contains_member_name(&assigned, &existing) {
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

    if let Some(inventory) = graph
        .metadata
        .as_mut()
        .and_then(|metadata| metadata.final_convolution_sha256.as_mut())
    {
        *inventory = inventory
            .iter()
            .map(|(reference, hash)| (packaged_by_reference[reference].clone(), hash.clone()))
            .collect();
    }
    rewrite_final_phase_references(&mut graph, &packaged_by_reference);
    rebind_ledger_to_packaged_graph(&mut graph)?;
    Ok((graph, members.into_values().collect()))
}

/// Preserve final phase resource references through package-local renaming.
fn rewrite_final_phase_references(
    graph: &mut DspGraph,
    packaged_by_reference: &HashMap<String, String>,
) {
    let Some(ledger) = graph.correction_decisions.as_mut() else {
        return;
    };
    for record in &mut ledger.decisions {
        // Provisional rows describe the original assessment, not deployment.
        if record.stage != roomeq_model::decision_ledger::DecisionStage::Final {
            continue;
        }
        for evidence in &mut record.evidence_refs {
            let renamed = if let Some(reference) = evidence.strip_prefix("phase-fir:") {
                packaged_by_reference
                    .get(reference)
                    .map(|name| format!("phase-fir:{name}"))
            } else if let Some((driver, reference)) = evidence
                .strip_prefix("phase-fir-driver:")
                .and_then(|binding| binding.split_once(':'))
            {
                packaged_by_reference
                    .get(reference)
                    .map(|name| format!("phase-fir-driver:{driver}:{name}"))
            } else {
                None
            };
            // Removed resources and unrelated evidence remain historical facts.
            if let Some(renamed) = renamed {
                *evidence = renamed;
            }
        }
    }
}

/// Rebind a carried decision ledger to the packaged bytes.
///
/// Sidecar rewriting renames references without changing the processing
/// the ledger accepted. Carried Final rows are rebound to the packaged
/// graph instead of shipping the stale pre-package fingerprint; rows
/// without a binding (provisional history) are untouched. A graph with
/// no ledger passes through unchanged.
fn rebind_ledger_to_packaged_graph(graph: &mut DspGraph) -> anyhow::Result<()> {
    for chain in graph.channels.values_mut() {
        roomeq_model::joint_sub_report::refresh_joint_sub_binding(chain);
    }
    let Some(mut carried) = graph.correction_decisions.take() else {
        return Ok(());
    };
    let value = serde_json::to_value(&*graph)
        .map_err(|error| anyhow::anyhow!("packaged graph does not serialize: {error}"))?;
    let identity = roomeq_model::decision_ledger::canonical_value_identity(&value);
    if let Some(evidence) = &carried.acceptance_evidence {
        let sample_rate = evidence
            .payload
            .get("sample_rate_hz")
            .and_then(serde_json::Value::as_f64)
            .context("acceptance diagnostics lack their analysis sample rate")?;
        carried.acceptance_evidence = Some(
            roomeq_engine::quality::graph_acceptance_evidence(graph, sample_rate)
                .map_err(anyhow::Error::msg)?,
        );
    }
    roomeq_model::decision_ledger::rebind_ledger_to_repackaged_graph(&mut carried, &identity)
        .map_err(anyhow::Error::msg)?;
    carried.payload_binding = Some(roomeq_model::payload_binding::PayloadBinding::new(
        &value,
        &identity.fingerprint,
    ));
    graph.correction_decisions = Some(carried);
    Ok(())
}

/// Repackaging can rename resources, but cannot validate a stale source claim.
pub(crate) fn validate_source_ledger(graph: &DspGraph) -> anyhow::Result<()> {
    use roomeq_model::decision_ledger::{DecisionStage, canonical_value_identity};
    let Some(ledger) = &graph.correction_decisions else {
        return Ok(());
    };
    ledger.validate().map_err(anyhow::Error::msg)?;
    let mut value = serde_json::to_value(graph)?;
    value
        .as_object_mut()
        .expect("graph is a JSON object")
        .remove("correction_decisions");
    let identity = canonical_value_identity(&value);
    if let Some(evidence) = &ledger.acceptance_evidence {
        anyhow::ensure!(
            evidence.matches(&identity.fingerprint),
            "stale source acceptance evidence cannot be rebound by packaging"
        );
    }
    for record in &ledger.decisions {
        anyhow::ensure!(
            record.stage != DecisionStage::Final
                || record.final_graph_identity.as_deref() == Some(identity.fingerprint.as_str()),
            "stale source decision '{}' cannot be rebound by packaging",
            record.decision_id
        );
    }
    if let Some(binding) = &ledger.payload_binding {
        anyhow::ensure!(
            binding.matches(&value, &identity.fingerprint),
            "stale source payload binding cannot be refreshed by packaging"
        );
    }
    Ok(())
}

pub(crate) fn resource_map(
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
    // Package members travel across operating systems, so keep generated
    // filenames portable even when the source IR basename is valid only on
    // the host filesystem (for example, a colon on Unix).
    let portable_name = preferred
        .chars()
        .map(|character| {
            if character.is_control()
                || matches!(
                    character,
                    '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
                )
            {
                '_'
            } else {
                character
            }
        })
        .collect::<String>();
    let preferred = portable_name.trim_end_matches([' ', '.']);
    let mut preferred = if preferred.is_empty() || matches!(preferred, "." | "..") {
        String::from("room_eq_ir.wav")
    } else {
        preferred.to_string()
    };
    if is_reserved_windows_device_name(&preferred) {
        preferred.insert(0, '_');
    }
    if !contains_member_name(assigned, &preferred) {
        return preferred;
    }
    let preferred_path = Path::new(&preferred);
    let stem = preferred_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("room_eq_ir");
    let extension = preferred_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    for suffix in 2_u64..=u64::MAX {
        let candidate = format!("{stem}_{suffix:03}{extension}");
        if !contains_member_name(assigned, &candidate) {
            return candidate;
        }
    }
    unreachable!("u64 package-member namespace exhausted")
}

fn contains_member_name(assigned: &BTreeSet<String>, candidate: &str) -> bool {
    let candidate = candidate.to_lowercase();
    assigned.iter().any(|name| name.to_lowercase() == candidate)
}

fn portable_member_key(path: &Path) -> anyhow::Result<String> {
    let text = path
        .to_str()
        .context("export package member path is not valid UTF-8")?;
    Ok(text.to_lowercase())
}

fn is_reserved_windows_device_name(component: &str) -> bool {
    let stem = component
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches([' ', '.'])
        .to_lowercase();
    matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
        || ["com", "lpt"].iter().any(|prefix| {
            matches!(
                stem.strip_prefix(prefix).unwrap_or_default(),
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
}

fn validate_member_path(path: &Path) -> anyhow::Result<()> {
    let path_text = path
        .to_str()
        .context("export package member path is not valid UTF-8")?;
    let portable_components = path_text.split('/').all(|component| {
        !component.is_empty()
            && component != "."
            && component != ".."
            && !component.ends_with([' ', '.'])
            && !component.chars().any(|character| {
                character.is_control()
                    || matches!(
                        character,
                        '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
                    )
            })
            && !is_reserved_windows_device_name(component)
    });
    if path_text.is_empty()
        || path_text.starts_with('/')
        || !portable_components
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        anyhow::bail!(
            "export package member '{}' must be a safe portable relative path",
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
                physical_correction_target: None,
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
                joint_sub: None,
                early_reflections: None,
                t60_octaves: None,
                waterfall: None,
                resonance_decays: None,
                wavelet: None,
                early_late_curves: None,
            },
        )
    }

    #[test]
    fn member_paths_reject_windows_traversal_and_reserved_names_on_unix() {
        for path in [
            r"..\\escape.wav",
            r"C:\\escape.wav",
            "nested/name?.wav",
            "nested/name*.wav",
            "nested/name<.wav",
            "nested/name>.wav",
            "nested/name|.wav",
            "nested/name\".wav",
            "nested/trailing-dot.",
            "nested/trailing-space ",
            "nested//empty.wav",
            "CON.wav",
            "aux",
            "nested/COM1.wav",
            "nested/LPT9.wav",
            "nested/COM¹.wav",
        ] {
            assert!(
                validate_member_path(Path::new(path)).is_err(),
                "accepted non-portable member {path:?}"
            );
        }
    }

    #[test]
    fn package_member_names_sanitize_device_names_and_preserve_resource_bytes() {
        let graph = DspGraph {
            version: "1.3.0".into(),
            artifact_bundle_schema_version: None,
            global_plugins: Vec::new(),
            channels: HashMap::from([
                convolution_chain("left", "source/CON.wav"),
                convolution_chain("right", "source/COM1.WAV"),
            ]),
            metadata: None,
            correction_decisions: None,
            deployed_source_curves: Default::default(),
        };
        let resources = vec![
            ConvolutionResource {
                reference: "source/CON.wav".into(),
                bytes: Arc::from(b"convolution bytes one".as_slice()),
            },
            ConvolutionResource {
                reference: "source/COM1.WAV".into(),
                bytes: Arc::from(b"convolution bytes two".as_slice()),
            },
        ];

        let (packaged, members) =
            package_convolution_sidecars(&graph, &resources, &BTreeSet::new(), &HashMap::new())
                .unwrap();
        let left_name = packaged.channels["left"].plugins[0].parameters["ir_file"]
            .as_str()
            .unwrap();
        let right_name = packaged.channels["right"].plugins[0].parameters["ir_file"]
            .as_str()
            .unwrap();

        assert_eq!(left_name, "_CON.wav");
        assert_eq!(right_name, "_COM1.WAV");
        assert_eq!(members.len(), 2);
        for (name, expected) in [
            (left_name, resources[0].bytes.as_ref()),
            (right_name, resources[1].bytes.as_ref()),
        ] {
            let member = members
                .iter()
                .find(|member| member.relative_path == Path::new(name))
                .unwrap();
            assert_eq!(member.bytes.as_ref(), expected);
        }
    }

    #[test]
    fn package_member_names_avoid_case_insensitive_collisions() {
        let graph = DspGraph {
            version: "1.3.0".into(),
            artifact_bundle_schema_version: None,
            global_plugins: Vec::new(),
            channels: HashMap::from([
                convolution_chain("left", "source/fir.wav"),
                convolution_chain("right", "source/FIR.WAV"),
            ]),
            metadata: None,
            correction_decisions: None,
            deployed_source_curves: Default::default(),
        };
        let resources = vec![
            ConvolutionResource {
                reference: "source/fir.wav".into(),
                bytes: Arc::from(b"first impulse".as_slice()),
            },
            ConvolutionResource {
                reference: "source/FIR.WAV".into(),
                bytes: Arc::from(b"second impulse".as_slice()),
            },
        ];
        let occupied = BTreeSet::from(["FIR.WAV".to_string()]);

        let (packaged, members) =
            package_convolution_sidecars(&graph, &resources, &occupied, &HashMap::new()).unwrap();
        let names = members
            .iter()
            .map(|member| member.relative_path.to_string_lossy().to_lowercase())
            .collect::<BTreeSet<_>>();
        assert_eq!(members.len(), 2);
        assert_eq!(names.len(), 2);
        assert!(!names.contains("fir.wav"));
        for (channel, expected) in [
            ("left", resources[0].bytes.as_ref()),
            ("right", resources[1].bytes.as_ref()),
        ] {
            let name = packaged.channels[channel].plugins[0].parameters["ir_file"]
                .as_str()
                .unwrap();
            let member = members
                .iter()
                .find(|member| member.relative_path == Path::new(name))
                .unwrap();
            assert_eq!(member.bytes.as_ref(), expected);
        }

        let variants = ExportPackage::new(vec![
            ExportPackageMember::new("Sound.wav", b"one".to_vec()).unwrap(),
            ExportPackageMember::new("sound.WAV", b"two".to_vec()).unwrap(),
        ]);
        assert!(variants.is_err(), "case variants must collide on Windows");
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
                    plugin_type: "convolution".into(),
                    parameters,
                };
                let (name, mut chain) = convolution_chain("left", "valid.wav");
                chain.plugins.clear();
                let mut graph = DspGraph {
                    version: "1.3.0".into(),
                    artifact_bundle_schema_version: None,
                    global_plugins: Vec::new(),
                    channels: HashMap::new(),
                    metadata: None,
                    correction_decisions: None,
                    deployed_source_curves: Default::default(),
                };
                match scope {
                    "global" => graph.global_plugins.push(plugin),
                    "channel" => chain.plugins.push(plugin),
                    _ => {
                        chain.drivers = Some(vec![roomeq_model::DriverDspChain {
                            measured_acoustics: None,
                            name: "woofer".into(),
                            index: 0,
                            plugins: vec![plugin],
                            initial_curve: None,
                            measured_band_hz: None,
                        }])
                    }
                }
                graph.channels.insert(name, chain);
                let error = checked_convolution_resource_references(&graph).unwrap_err();
                assert!(error.to_string().contains(scope), "{error}");
                // Legacy graphs also must not package a malformed stage.
                assert!(
                    package_convolution_sidecars(&graph, &[], &BTreeSet::new(), &HashMap::new())
                        .is_err()
                );
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
            artifact_bundle_schema_version: None,
            global_plugins: Vec::new(),
            channels: HashMap::from([
                convolution_chain("left", "a.wav"),
                convolution_chain("right", "b.wav"),
                convolution_chain("center", "c.wav"),
            ]),
            metadata: None,
            correction_decisions: None,
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

    /// A carried ledger survives packaging rebound to the packaged bytes:
    /// Final rows bind the new fingerprint, provisional history is
    /// untouched, and the rebound ledger validates.
    #[test]
    fn packaged_graph_rebinds_carried_ledger() {
        use roomeq_model::decision_ledger::{
            CorrectionDecisionLedger, DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord,
            DecisionStage, DecisionStatus, canonical_graph_identity,
        };
        let impulse: Arc<[u8]> = Arc::from(b"rebind-impulse".as_slice());
        let mut graph = DspGraph {
            deployed_source_curves: Default::default(),
            version: "1.3.0".to_string(),
            artifact_bundle_schema_version: None,
            global_plugins: Vec::new(),
            channels: HashMap::from([convolution_chain("left", "a.wav")]),
            metadata: None,
            correction_decisions: None,
        };
        // Reference rewriting renames a.wav, so the pre-package binding
        // would go stale if it were carried forward unchanged.
        let stale_identity = canonical_graph_identity(&graph);
        let evidence = roomeq_engine::quality::graph_acceptance_evidence(&graph, 48_000.0).unwrap();
        let provisional = DecisionRecord {
            decision_id: "dec-history".to_string(),
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            stage: DecisionStage::Provisional,
            logical_input: "left".to_string(),
            physical_output: "left".to_string(),
            measurement_refs: Vec::new(),
            seat_refs: Vec::new(),
            frequency_band_hz: Some([40.0, 400.0]),
            filter_center_hz: None,
            action: DecisionAction::Equalize,
            status: DecisionStatus::InsufficientEvidence,
            reason_codes: vec!["no_timing_reference".to_string()],
            observed: Vec::new(),
            limits: Vec::new(),
            evidence_refs: Vec::new(),
            confidence: roomeq_model::AssessmentConfidence::Low,
            related_decision_ids: Vec::new(),
            supersedes_ids: Vec::new(),
            final_graph_identity: None,
        };
        let mut applied = provisional.clone();
        applied.decision_id = "dec-eq-left".to_string();
        applied.stage = DecisionStage::Final;
        applied.status = DecisionStatus::Applied;
        applied.final_graph_identity = Some(stale_identity.fingerprint.clone());
        graph.correction_decisions = Some(CorrectionDecisionLedger {
            acceptance_evidence: Some(evidence),
            payload_binding: None,
            ledger_version: DECISION_LEDGER_VERSION.to_string(),
            decisions: vec![applied, provisional],
            channel_summaries: Vec::new(),
        });
        let resources = vec![ConvolutionResource {
            reference: "a.wav".to_string(),
            bytes: Arc::clone(&impulse),
        }];
        // Occupy the original member name so packaging must rename the
        // reference; otherwise the packaged bytes would equal the input.
        let occupied = BTreeSet::from(["a.wav".to_string()]);
        let (packaged, _) =
            package_convolution_sidecars(&graph, &resources, &occupied, &HashMap::new()).unwrap();
        let ledger = packaged
            .correction_decisions
            .as_ref()
            .expect("packaged graph keeps its ledger");
        assert!(ledger.validate().is_ok());
        let fresh = canonical_graph_identity(&{
            let mut cleared = packaged.clone();
            cleared.correction_decisions = None;
            cleared
        });
        assert_ne!(fresh.fingerprint, stale_identity.fingerprint);
        let evidence = ledger.acceptance_evidence.as_ref().unwrap();
        assert!(evidence.matches(&fresh.fingerprint));
        assert!(!evidence.matches(&stale_identity.fingerprint));
        assert_eq!(
            evidence.payload["channels"]["left"]["bundle"]["disposition"]["channels"][0]["fir_references"]
                [0],
            packaged.channels["left"].plugins[0].parameters["ir_file"],
        );
        let mut payload = serde_json::to_value(&packaged).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .remove("correction_decisions");
        assert!(
            ledger
                .payload_binding
                .as_ref()
                .unwrap()
                .matches(&payload, &fresh.fingerprint)
        );
        for record in &ledger.decisions {
            if record.decision_id == "dec-eq-left" {
                assert_eq!(record.stage, DecisionStage::Final);
                assert_eq!(
                    record.final_graph_identity.as_deref(),
                    Some(fresh.fingerprint.as_str())
                );
            } else {
                assert_eq!(record.stage, DecisionStage::Provisional);
                assert!(record.final_graph_identity.is_none());
            }
        }
    }

    #[test]
    fn roadmap_correction_packaged_phase_references_follow_resources() {
        use roomeq_model::decision_ledger::{
            CorrectionDecisionLedger, DECISION_LEDGER_VERSION, DecisionAction, DecisionRecord,
            DecisionStage, DecisionStatus, canonical_graph_identity,
        };
        let mut graph = DspGraph::new("test");
        let (_, mut chain) = convolution_chain("left", "source/a:phase.wav");
        chain.drivers = Some(vec![roomeq_model::DriverDspChain {
            measured_acoustics: None,
            name: "woofer".into(),
            index: 2,
            plugins: chain.plugins.clone(),
            initial_curve: None,
            measured_band_hz: None,
        }]);
        chain.plugins.clear();
        graph.channels.insert("left".into(), chain);
        let mut record = DecisionRecord::example(DecisionStatus::Applied);
        record.action = DecisionAction::PhaseCorrect;
        record.logical_input = "left".into();
        record.physical_output = "left".into();
        record.stage = DecisionStage::Final;
        record.final_graph_identity = Some(canonical_graph_identity(&graph).fingerprint);
        record.evidence_refs = vec![
            "phase-fir:source/a:phase.wav".into(),
            "phase-fir-driver:2:source/a:phase.wav".into(),
            "capture:source/a:phase.wav".into(),
        ];
        let mut history = record.clone();
        history.decision_id = "history".into();
        history.stage = DecisionStage::Provisional;
        history.final_graph_identity = None;
        graph.correction_decisions = Some(CorrectionDecisionLedger {
            ledger_version: DECISION_LEDGER_VERSION.into(),
            decisions: vec![record, history.clone()],
            acceptance_evidence: None,
            payload_binding: None,
            channel_summaries: Vec::new(),
        });
        let resources = [ConvolutionResource {
            reference: "source/a:phase.wav".into(),
            bytes: Arc::from(b"resource-reference-fixture".as_slice()),
        }];
        let (packaged, members) = package_convolution_sidecars(
            &graph,
            &resources,
            &BTreeSet::from(["a:phase.wav".into()]),
            &HashMap::new(),
        )
        .unwrap();
        let filename = packaged.channels["left"].drivers.as_ref().unwrap()[0].plugins[0].parameters
            ["ir_file"]
            .as_str()
            .unwrap();
        assert_ne!(filename, "source/a:phase.wav");
        assert!(
            !filename.contains(':'),
            "package names stay portable across Windows"
        );
        assert_eq!(members[0].relative_path, Path::new(filename));
        assert_eq!(members[0].bytes.as_ref(), resources[0].bytes.as_ref());
        let ledger = packaged.correction_decisions.as_ref().unwrap();
        assert_eq!(
            ledger.decisions[0].evidence_refs,
            vec![
                format!("phase-fir:{filename}"),
                format!("phase-fir-driver:2:{filename}"),
                "capture:source/a:phase.wav".into(),
            ]
        );
        assert_eq!(ledger.decisions[1], history);
        validate_source_ledger(&packaged).unwrap();
        assert_eq!(
            graph.correction_decisions.as_ref().unwrap().decisions[0].evidence_refs[0],
            "phase-fir:source/a:phase.wav"
        );
    }

    #[test]
    fn roadmap_correction_package_cannot_rebind_stale_delivery_claims() {
        use roomeq_model::decision_ledger::{
            CorrectionDecisionLedger, DECISION_LEDGER_VERSION, DecisionRecord, DecisionStatus,
            canonical_graph_identity,
        };
        let mut graph = DspGraph::new("test");
        graph.add_channel("left", Vec::new());
        let identity = canonical_graph_identity(&graph);
        let payload = serde_json::to_value(&graph).unwrap();
        let mut record = DecisionRecord::example(DecisionStatus::Applied);
        record.stage = roomeq_model::decision_ledger::DecisionStage::Final;
        record.final_graph_identity = Some(identity.fingerprint.clone());
        for portable in [false, true] {
            let mut changed = graph.clone();
            changed.correction_decisions = Some(CorrectionDecisionLedger {
                acceptance_evidence: None,
                payload_binding: portable.then(|| {
                    roomeq_model::payload_binding::PayloadBinding::new(
                        &payload,
                        &identity.fingerprint,
                    )
                }),
                ledger_version: DECISION_LEDGER_VERSION.to_owned(),
                decisions: vec![record.clone()],
                channel_summaries: Vec::new(),
            });
            changed.global_plugins.push(PluginConfigWrapper {
                plugin_type: "gain".to_owned(),
                parameters: serde_json::json!({"gain_db": 12.0}),
            });
            let error =
                package_convolution_sidecars(&changed, &[], &BTreeSet::new(), &HashMap::new())
                    .expect_err("packaging must not bless preexisting stale claims");
            assert!(error.to_string().contains("stale"), "{error}");
        }
    }
}
