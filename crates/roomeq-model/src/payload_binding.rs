//! Portable decision-payload binding for independent report consumers.
//!
//! The digest supplements the existing graph identity; it is not a signature
//! or evidence of acoustic correctness. Its input excludes the entire ledger.
//! Encoding v1 starts with the algorithm's UTF-8 bytes and a zero byte, then
//! encodes the graph identity as a string and the JSON payload as follows:
//! `n`/`t`/`f` for null/true/false; `i` + ASCII decimal + `;` for integers;
//! `d` + eight IEEE-754 binary64 big-endian bytes for floating-point numbers;
//! `s` + u64 big-endian UTF-8 byte count + string bytes; `a`/`o` + u64
//! big-endian element/pair count + recursively encoded children. Object keys
//! are strings sorted by Unicode scalar order; array order is significant.
//! Integers and floats, including signed zero, remain distinct. SHA-256 is
//! emitted as 64 lowercase hexadecimal digits. Python's counterpart lives in
//! `scripts/src/payload_binding.py`; cross-language integration tests bind
//! actual Rust-generated workflow outputs rather than only canned digests.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Versioned, format-independent digest codec for delivered JSON values.
pub const PAYLOAD_BINDING_ALGORITHM: &str = "sha256-json-typed-v1";

/// Binding between a delivered payload and its existing decision graph identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PayloadBinding {
    /// Exact encoding version; unknown versions cannot authorize report claims.
    pub algorithm: String,
    /// Existing graph identity used by all final decision records.
    pub graph_identity: String,
    /// SHA-256 over the typed payload encoding and graph identity.
    pub sha256: String,
}

impl PayloadBinding {
    /// Bind an already serialized payload, excluding its decision ledger.
    pub fn new(payload: &Value, graph_identity: &str) -> Self {
        let mut hash = Sha256::new();
        hash.update(PAYLOAD_BINDING_ALGORITHM.as_bytes());
        hash.update([0]);
        encode(&mut hash, &Value::String(graph_identity.to_owned()));
        encode(&mut hash, payload);
        Self {
            algorithm: PAYLOAD_BINDING_ALGORITHM.to_owned(),
            graph_identity: graph_identity.to_owned(),
            sha256: hash
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        }
    }

    /// Recompute the binding; supplied labels alone never establish agreement.
    pub fn matches(&self, payload: &Value, graph_identity: &str) -> bool {
        !graph_identity.is_empty() && *self == Self::new(payload, graph_identity)
    }
}

fn encode(hash: &mut Sha256, value: &Value) {
    match value {
        Value::Null => hash.update(b"n"),
        Value::Bool(true) => hash.update(b"t"),
        Value::Bool(false) => hash.update(b"f"),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            hash.update(b"i");
            hash.update(number.to_string().as_bytes());
            hash.update(b";");
        }
        Value::Number(number) => {
            hash.update(b"d");
            // serde_json Number is a finite binary64 or an integer in this API.
            hash.update(
                number
                    .as_f64()
                    .expect("JSON number is representable")
                    .to_be_bytes(),
            );
        }
        Value::String(text) => {
            hash.update(b"s");
            hash.update((text.len() as u64).to_be_bytes());
            hash.update(text.as_bytes());
        }
        Value::Array(values) => {
            hash.update(b"a");
            hash.update((values.len() as u64).to_be_bytes());
            for value in values {
                encode(hash, value);
            }
        }
        Value::Object(values) => {
            hash.update(b"o");
            hash.update((values.len() as u64).to_be_bytes());
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort();
            for key in keys {
                encode(hash, &Value::String(key.clone()));
                encode(hash, &values[key]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roadmap_correction_payload_binding_checks_content_and_numeric_types() {
        let value = json!({"unicode": "é𝄞", "small": 1e-20, "large": u64::MAX,
            "nested": [null, true, false, -0.0, 1.0, -7]});
        let binding = PayloadBinding::new(&value, "graph-1");
        // Independently encoded by Python; shared regression vector, not an
        // acoustic/reference-model validation claim.
        assert_eq!(
            binding.sha256,
            "ea5287ff38c712e5de32deb7ce8ecdc89c8d9309eb30dd35e2c6616c7a75c029"
        );
        assert!(binding.matches(&value, "graph-1"));
        assert!(!binding.matches(&value, "graph-2"));
        let mut changed = value.clone();
        changed["nested"][4] = json!(1);
        assert!(!binding.matches(&changed, "graph-1"));
        changed = value.clone();
        changed["nested"][3] = json!(0.0);
        assert!(!binding.matches(&changed, "graph-1"));
        let mut unsupported = binding.clone();
        unsupported.algorithm = "unknown".to_owned();
        assert!(!unsupported.matches(&value, "graph-1"));
    }
}
