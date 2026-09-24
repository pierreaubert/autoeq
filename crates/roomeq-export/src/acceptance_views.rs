//! Binding of multi-view acceptance summaries to exported artifacts.
//!
//! The quality crate owns the measured views; this module owns the binding:
//! every bound view carries the settings hash it was computed with, a payload
//! hash over its serialized content, and the immutable graph identity it
//! assesses. Binding recomputes payload digests and checks shared provenance.
//! This establishes internal binding consistency, not acoustic acceptance or
//! capture authenticity. Production result/report integration remains separate.

// Rust guideline compliant 2026-02-21

use roomeq_model::payload_binding::PayloadBinding;
use serde::{Deserialize, Serialize};

/// One acceptance view summary offered for binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewBindingInput {
    /// View name (`magnitude`, `ir_step`, `etc`, `decay`, `ambient_noise`, `headroom`).
    pub view_name: String,
    /// Settings hash the view was computed with.
    pub settings_hash: String,
    /// `sha256-json-typed-v1` digest of the payload and graph identity.
    pub payload_hash: String,
    /// Immutable DSP graph identity the view assesses.
    pub graph_identity: String,
    /// Actual serialized quality view; legacy hash-only inputs cannot bind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// One view bound to an exported artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundView {
    /// View name.
    pub view_name: String,
    /// Payload hash verified at bind time.
    pub payload_hash: String,
    /// Graph identity the view assesses.
    pub graph_identity: String,
}

/// Acceptance views bound to one exported artifact identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundAcceptanceViews {
    /// Fingerprint of the exported artifact (see `package_fingerprint`).
    pub artifact_fingerprint: String,
    /// Settings hash shared by every bound view.
    pub settings_hash: String,
    /// Bound views in binding order.
    pub views: Vec<BoundView>,
}

/// Bind acceptance view summaries to an exported artifact.
///
/// Requires a non-empty artifact fingerprint, at least one view, identical
/// settings hashes across every view and the bundle settings, recomputed
/// payload hashes, unique names, and identical nonempty graph identities.
/// Payload settings and provenance must agree with their binding metadata.
/// The caller must additionally match the graph to the exported artifact;
/// a supplied artifact fingerprint alone does not prove that relationship.
///
/// # Examples
///
/// ```
/// use roomeq_export::acceptance_views::{ViewBindingInput, bind_acceptance_views};
/// use roomeq_model::payload_binding::PayloadBinding;
/// let payload = serde_json::json!({"settings_hash": "s", "provenance": {"graph_identity": "g"}});
/// let views = vec![ViewBindingInput {
///     view_name: "magnitude".to_string(),
///     settings_hash: "s".to_string(),
///     payload_hash: PayloadBinding::new(&payload, "g").sha256,
///     graph_identity: "g".to_string(),
///     payload: Some(payload),
/// }];
/// let bound = bind_acceptance_views("artifact", "s", views).unwrap();
/// assert_eq!(bound.views.len(), 1);
/// ```
///
/// # Errors
///
/// Returns an error string when any binding invariant is violated.
pub fn bind_acceptance_views(
    artifact_fingerprint: &str,
    settings_hash: &str,
    views: Vec<ViewBindingInput>,
) -> Result<BoundAcceptanceViews, String> {
    if artifact_fingerprint.trim().is_empty() {
        return Err("binding needs a non-empty artifact fingerprint".to_string());
    }
    if settings_hash.trim().is_empty() {
        return Err("binding needs a non-empty settings hash".to_string());
    }
    if views.is_empty() {
        return Err("binding needs at least one view".to_string());
    }
    let graph = &views[0].graph_identity;
    let mut names = std::collections::HashSet::new();
    for view in &views {
        if view.view_name.trim().is_empty() || !names.insert(view.view_name.trim()) {
            return Err("bound views need unique nonempty names".to_string());
        }
        if view.settings_hash != settings_hash {
            return Err(format!(
                "view `{}` carries mismatched settings",
                view.view_name
            ));
        }
        if view.payload_hash.is_empty() {
            return Err(format!(
                "view `{}` has an empty payload hash",
                view.view_name
            ));
        }
        if view.graph_identity.trim().is_empty() || &view.graph_identity != graph {
            return Err(format!(
                "view `{}` has missing or mismatched graph identity",
                view.view_name
            ));
        }
        let payload = view
            .payload
            .as_ref()
            .ok_or_else(|| format!("view `{}` has no payload to verify", view.view_name))?;
        if payload
            .get("settings_hash")
            .and_then(serde_json::Value::as_str)
            != Some(settings_hash)
            || payload
                .pointer("/provenance/graph_identity")
                .and_then(serde_json::Value::as_str)
                != Some(graph.as_str())
        {
            return Err(format!(
                "view `{}` payload provenance disagrees with binding",
                view.view_name
            ));
        }
        if PayloadBinding::new(payload, graph).sha256 != view.payload_hash {
            return Err(format!(
                "view `{}` payload hash does not match its contents",
                view.view_name
            ));
        }
    }
    Ok(BoundAcceptanceViews {
        artifact_fingerprint: artifact_fingerprint.to_string(),
        settings_hash: settings_hash.to_string(),
        views: views
            .into_iter()
            .map(|view| BoundView {
                view_name: view.view_name,
                payload_hash: view.payload_hash,
                graph_identity: view.graph_identity,
            })
            .collect(),
    })
}

/// Bind carried diagnostics to the actual rendered artifact bytes.
///
/// Returns no sidecar for legacy graphs without production diagnostics.
/// The association is not an independent backend replay or acoustic verdict.
pub(crate) fn acceptance_views_member(
    graph: &roomeq_model::DspGraph,
    artifact: &crate::ExportPackageMember,
) -> anyhow::Result<Option<crate::ExportPackageMember>> {
    let Some(evidence) = graph
        .correction_decisions
        .as_ref()
        .and_then(|ledger| ledger.acceptance_evidence.as_ref())
    else {
        return Ok(None);
    };
    let mut value = serde_json::to_value(graph)?;
    value
        .as_object_mut()
        .expect("serialized graph")
        .remove("correction_decisions");
    let identity = roomeq_model::decision_ledger::canonical_value_identity(&value);
    anyhow::ensure!(
        evidence.matches(&identity.fingerprint),
        "stale acceptance-view binding"
    );
    let channels = evidence
        .payload
        .get("channels")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("acceptance diagnostics lack channel payloads"))?;
    anyhow::ensure!(
        channels.len() == graph.channels.len()
            && channels
                .keys()
                .all(|name| graph.channels.contains_key(name)),
        "acceptance diagnostics do not cover exported channels"
    );
    let mut bindings = std::collections::BTreeMap::new();
    for (name, channel) in channels {
        let bundle: roomeq_engine::quality::AcceptanceBundle =
            serde_json::from_value(channel["bundle"].clone())?;
        let settings_hash = bundle.settings.settings_hash();
        let mut views = Vec::new();
        for view_name in [
            "magnitude",
            "ir_step",
            "etc",
            "decay",
            "ambient_noise",
            "headroom",
            "disposition",
            "fine_magnitude",
        ] {
            let payload = if view_name == "fine_magnitude" {
                &channel[view_name]
            } else {
                &channel["bundle"][view_name]
            };
            if payload.is_null() {
                continue;
            }
            views.push(ViewBindingInput {
                view_name: view_name.into(),
                settings_hash: settings_hash.clone(),
                graph_identity: identity.fingerprint.clone(),
                payload_hash: PayloadBinding::new(payload, &identity.fingerprint).sha256,
                payload: Some(payload.clone()),
            });
        }
        if !views.is_empty() {
            bindings.insert(
                name,
                bind_acceptance_views(&artifact.sha256, &settings_hash, views)
                    .map_err(anyhow::Error::msg)?,
            );
        }
    }
    let payload = serde_json::json!({
        "version": "export-acceptance-views-v1",
        "artifact": artifact.relative_path,
        "artifact_sha256": artifact.sha256,
        "scope": "bound retained predictions; not independent backend replay or acoustic verification",
        "evidence": evidence,
        "bindings": bindings,
    });
    let mut path = artifact.relative_path.clone();
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("artifact filename is not UTF-8"))?;
    path.set_file_name(format!("{filename}.acceptance-views.json"));
    Ok(Some(crate::ExportPackageMember::new(
        path,
        serde_json::to_vec_pretty(&payload)?,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str) -> ViewBindingInput {
        let payload = serde_json::json!({
            "settings_hash": "settings",
            "provenance": {"graph_identity": "graph"},
            "samples": [0.0, 1.0]
        });
        ViewBindingInput {
            view_name: name.to_string(),
            settings_hash: "settings".to_string(),
            payload_hash: PayloadBinding::new(&payload, "graph").sha256,
            graph_identity: "graph".to_string(),
            payload: Some(payload),
        }
    }

    #[test]
    fn roadmap_correction_export_binds_production_views_to_artifact_bytes() {
        use roomeq_model::decision_ledger::{CorrectionDecisionLedger, DECISION_LEDGER_VERSION};
        let mut graph = roomeq_model::DspGraph::new("1");
        graph.add_channel("L", Vec::new());
        let evidence = roomeq_engine::quality::graph_acceptance_evidence(&graph, 48_000.0).unwrap();
        let value = serde_json::to_value(&graph).unwrap();
        let binding = PayloadBinding::new(&value, &evidence.binding.graph_identity);
        graph.correction_decisions = Some(CorrectionDecisionLedger {
            acceptance_evidence: Some(evidence),
            payload_binding: Some(binding),
            ledger_version: DECISION_LEDGER_VERSION.into(),
            decisions: Vec::new(),
            channel_summaries: Vec::new(),
        });
        let build = |graph: &roomeq_model::DspGraph| {
            crate::build_export_package(
                graph,
                crate::ExportFormat::BiquadCoefficients,
                std::path::Path::new("filters.json"),
                48_000.0,
                &[],
                &Default::default(),
                &Default::default(),
            )
        };
        let package = build(&graph).unwrap();
        assert!(
            crate::build_export_package(
                &graph,
                crate::ExportFormat::BiquadCoefficients,
                std::path::Path::new("filters.json"),
                44_100.0,
                &[],
                &Default::default(),
                &Default::default(),
            )
            .is_err()
        );
        let artifact = package
            .members
            .iter()
            .find(|member| member.relative_path == std::path::Path::new("filters.json"))
            .unwrap();
        let views = package
            .members
            .iter()
            .find(|member| {
                member.relative_path == std::path::Path::new("filters.json.acceptance-views.json")
            })
            .unwrap();
        let views: serde_json::Value = serde_json::from_slice(&views.bytes).unwrap();
        assert_eq!(views["artifact_sha256"], artifact.sha256);
        assert_eq!(
            views["bindings"]["L"]["artifact_fingerprint"],
            artifact.sha256
        );
        assert!(
            views["evidence"]["payload"]["channels"]["L"]["unavailable"]["headroom"].is_string()
        );
        graph
            .correction_decisions
            .as_mut()
            .unwrap()
            .acceptance_evidence
            .as_mut()
            .unwrap()
            .payload["scope"] = "tampered".into();
        assert!(build(&graph).is_err());
    }

    #[test]
    fn roadmap_correction_acceptance_binding_rejects_duplicate_views() {
        assert!(
            bind_acceptance_views(
                "artifact",
                "settings",
                vec![input("magnitude"), input("magnitude")]
            )
            .is_err()
        );
    }

    #[test]
    fn roadmap_correction_acceptance_binding_rejects_mixed_graphs() {
        let mut other = input("headroom");
        other.graph_identity = "other-graph".into();
        other.payload.as_mut().unwrap()["provenance"]["graph_identity"] = "other-graph".into();
        other.payload_hash =
            PayloadBinding::new(other.payload.as_ref().unwrap(), "other-graph").sha256;
        assert!(
            bind_acceptance_views("artifact", "settings", vec![input("magnitude"), other]).is_err()
        );
    }

    #[test]
    fn acceptance_binding_keeps_matched_views() {
        let bound = bind_acceptance_views(
            "artifact",
            "settings",
            vec![input("magnitude"), input("headroom")],
        )
        .unwrap();
        assert_eq!(bound.views.len(), 2);
        assert_eq!(bound.artifact_fingerprint, "artifact");
    }

    #[test]
    fn roadmap_correction_acceptance_binding_recomputes_payload_hash() {
        let mut view = input("magnitude");
        view.payload.as_mut().unwrap()["samples"][0] = 6.0.into();
        let error = bind_acceptance_views("artifact", "settings", vec![view.clone()]).unwrap_err();
        assert!(error.contains("payload hash"));
        view.payload_hash = PayloadBinding::new(view.payload.as_ref().unwrap(), "graph").sha256;
        let json = serde_json::to_string(&view).unwrap();
        let roundtrip = serde_json::from_str(&json).unwrap();
        assert!(bind_acceptance_views("artifact", "settings", vec![roundtrip]).is_ok());
    }

    #[test]
    fn roadmap_correction_acceptance_binding_legacy_hash_is_not_evidence() {
        let mut value = serde_json::to_value(input("magnitude")).unwrap();
        value.as_object_mut().unwrap().remove("payload");
        let legacy = serde_json::from_value(value).unwrap();
        let error = bind_acceptance_views("artifact", "settings", vec![legacy]).unwrap_err();
        assert!(error.contains("no payload"));
    }

    #[test]
    fn roadmap_correction_acceptance_binding_rejects_payload_provenance() {
        for pointer in ["/settings_hash", "/provenance/graph_identity"] {
            let mut view = input("magnitude");
            *view.payload.as_mut().unwrap().pointer_mut(pointer).unwrap() = "wrong".into();
            view.payload_hash = PayloadBinding::new(view.payload.as_ref().unwrap(), "graph").sha256;
            let error = bind_acceptance_views("artifact", "settings", vec![view]).unwrap_err();
            assert!(error.contains("payload provenance"));
        }
    }

    #[test]
    fn acceptance_binding_rejects_mismatched_settings() {
        let mut views = vec![input("magnitude")];
        views.push(ViewBindingInput {
            settings_hash: "other".to_string(),
            ..input("etc")
        });
        assert!(bind_acceptance_views("artifact", "settings", views).is_err());
    }

    #[test]
    fn acceptance_binding_rejects_empty_payload_or_identity() {
        let mut views = vec![input("magnitude")];
        views[0].payload_hash.clear();
        assert!(bind_acceptance_views("artifact", "settings", views.clone()).is_err());
        views[0].payload_hash = "payload".to_string();
        views[0].graph_identity.clear();
        assert!(bind_acceptance_views("artifact", "settings", views).is_err());
    }

    #[test]
    fn acceptance_binding_rejects_empty_artifact_or_no_views() {
        assert!(bind_acceptance_views("", "settings", vec![input("magnitude")]).is_err());
        assert!(bind_acceptance_views("artifact", "settings", Vec::new()).is_err());
    }
}
