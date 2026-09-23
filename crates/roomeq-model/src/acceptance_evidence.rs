//! Versioned transport for quality-owned acceptance diagnostics.
//!
//! The payload keeps canonical quality view serialization without a model-to-
//! quality dependency. It lives inside the excluded decision-ledger attachment
//! and has its own digest: graph and view payloads cannot hash themselves.

use crate::payload_binding::PayloadBinding;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Current serialized acceptance diagnostic format.
pub const ACCEPTANCE_EVIDENCE_VERSION: &str = "acceptance-evidence-v1";

/// Graph-bound diagnostic payload, not a safety or acoustic acceptance certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AcceptanceEvidence {
    /// Version controlling interpretation of the quality-owned payload.
    pub version: String,
    /// Serialized quality bundles and explicit unavailable reasons by channel.
    pub payload: serde_json::Value,
    /// Independent digest covering both the view payload and delivered graph ID.
    pub binding: PayloadBinding,
}

impl AcceptanceEvidence {
    /// Bind quality diagnostics to their source graph.
    pub fn new(payload: serde_json::Value, graph_identity: &str) -> Self {
        Self {
            version: ACCEPTANCE_EVIDENCE_VERSION.into(),
            binding: PayloadBinding::new(&payload, graph_identity),
            payload,
        }
    }

    /// Check format and payload identity without asserting acoustic correctness.
    pub fn matches(&self, graph_identity: &str) -> bool {
        self.version == ACCEPTANCE_EVIDENCE_VERSION
            && self.binding.matches(&self.payload, graph_identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roadmap_correction_acceptance_transport_binds_graph_and_contents() {
        let mut evidence = AcceptanceEvidence::new(serde_json::json!({"channels": {}}), "graph-1");
        assert!(evidence.matches("graph-1"));
        assert!(!evidence.matches("graph-2"));
        let roundtrip: AcceptanceEvidence =
            serde_json::from_str(&serde_json::to_string(&evidence).unwrap()).unwrap();
        assert_eq!(evidence, roundtrip);
        evidence.payload["channels"] = serde_json::json!({"invented": {}});
        assert!(!evidence.matches("graph-1"));
    }
}
