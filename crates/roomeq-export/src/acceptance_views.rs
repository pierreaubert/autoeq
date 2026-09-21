//! Binding of multi-view acceptance summaries to exported artifacts.
//!
//! The quality crate owns the measured views; this module owns the binding:
//! every bound view carries the settings hash it was computed with, a payload
//! hash over its serialized content, and the immutable graph identity it
//! assesses. Binding refuses mismatched settings and empty payloads so an
//! export can never cite views it was not built with. The root report consumes
//! [`BoundAcceptanceViews`] verbatim (see the Wave 2 handoff note).

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

/// One acceptance view summary offered for binding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewBindingInput {
    /// View name (`magnitude`, `ir_step`, `etc`, `decay`, `ambient_noise`, `headroom`).
    pub view_name: String,
    /// Settings hash the view was computed with.
    pub settings_hash: String,
    /// Hash over the serialized view payload.
    pub payload_hash: String,
    /// Immutable DSP graph identity the view assesses.
    pub graph_identity: String,
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
/// settings hashes across every view and the bundle settings, non-empty
/// payload hashes, and non-empty graph identities.
///
/// # Examples
///
/// ```
/// use roomeq_export::acceptance_views::{ViewBindingInput, bind_acceptance_views};
/// let views = vec![ViewBindingInput {
///     view_name: "magnitude".to_string(),
///     settings_hash: "s".to_string(),
///     payload_hash: "p".to_string(),
///     graph_identity: "g".to_string(),
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
    if artifact_fingerprint.is_empty() {
        return Err("binding needs a non-empty artifact fingerprint".to_string());
    }
    if settings_hash.is_empty() {
        return Err("binding needs a non-empty settings hash".to_string());
    }
    if views.is_empty() {
        return Err("binding needs at least one view".to_string());
    }
    for view in &views {
        if view.view_name.is_empty() {
            return Err("bound views need names".to_string());
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
        if view.graph_identity.is_empty() {
            return Err(format!("view `{}` has no graph identity", view.view_name));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str) -> ViewBindingInput {
        ViewBindingInput {
            view_name: name.to_string(),
            settings_hash: "settings".to_string(),
            payload_hash: "payload".to_string(),
            graph_identity: "graph".to_string(),
        }
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
