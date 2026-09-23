//! Wolfram cross-check: byte-exact artifact transport (AR01).
//!
//! Oracle: `wolfram/ar01_impulse_store.wls` supplies the fixture of
//! record (a 16-sample impulse plus rate/channel/encoding metadata).
//! The test stores those exact samples as f32 little-endian bytes plus
//! a metadata sidecar through the artifact API (`MemoryArtifactStore`
//! and `FsArtifactStore`) and requires byte-identical reloads. Decoded
//! f32 values must sit within half an f32 quantum of the golden f64
//! values (Q); missing keys read back as absent (X). No signal
//! transform is asserted: transport identity only.

use autoeq_artifacts::{ArtifactStore, FsArtifactStore, MemoryArtifactStore};
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance, require_reference};
use std::path::PathBuf;

const CASE: &str = "ar01_impulse_store";
const CASE_ID: &str = "autoeq-qa.ar01-impulse-store.v1";
const TOL: f64 = 1e-7;

fn encode_f32_le(samples: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for &sample in samples {
        bytes.extend_from_slice(&(sample as f32).to_le_bytes());
    }
    bytes
}

#[test]
fn wolfram_ar01_impulse_store() {
    let ref_json = require_reference(CASE, "ar01_impulse_store.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);

    let samples: Vec<f64> = serde_json::from_value(ref_json["impulse"].clone()).unwrap();
    let frames: usize = serde_json::from_value(ref_json["frames"].clone()).unwrap();
    let channels: usize = serde_json::from_value(ref_json["channels"].clone()).unwrap();
    let rate: f64 = serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap();
    let encoding: String = serde_json::from_value(ref_json["encoding"].clone()).unwrap();
    assert_eq!(samples.len(), 16, "{CASE}: expected 16 fixture samples");
    assert_eq!(frames, 16);
    assert_eq!(channels, 1);
    assert_eq!(rate, 48_000.0);
    assert_eq!(encoding, "f32-le");
    assert!(samples.iter().all(|v| v.is_finite()));

    let payload = encode_f32_le(&samples);
    assert_eq!(payload.len(), frames * channels * 4);
    let sidecar = serde_json::json!({
        "sample_rate_hz": rate,
        "channels": channels,
        "encoding": encoding,
        "frames": frames,
    });
    let sidecar_bytes = serde_json::to_vec(&sidecar).unwrap();

    // In-memory transport is byte-exact; missing keys stay missing.
    let memory = MemoryArtifactStore::new();
    let impulse_path = PathBuf::from("room/coax/impulse.f32");
    let sidecar_path = PathBuf::from("room/coax/impulse.json");
    memory
        .create_dir_all(PathBuf::from("room/coax").as_path())
        .unwrap();
    memory.write(&impulse_path, &payload).unwrap();
    memory.write(&sidecar_path, &sidecar_bytes).unwrap();
    assert_eq!(
        memory.read(&impulse_path).unwrap(),
        Some(payload.clone()),
        "{CASE}: memory transport altered impulse bytes"
    );
    assert_eq!(
        memory.read(&sidecar_path).unwrap(),
        Some(sidecar_bytes.clone()),
        "{CASE}: memory transport altered sidecar bytes"
    );
    assert_eq!(
        memory
            .read(&PathBuf::from("room/coax/missing.f32"))
            .unwrap(),
        None,
        "{CASE}: absent key must read as absent"
    );

    // Filesystem transport is byte-exact as well.
    let dir = std::env::temp_dir().join(format!("autoeq-qa-ar01-{}", std::process::id()));
    let fs = FsArtifactStore::new();
    fs.create_dir_all(&dir).unwrap();
    let fs_impulse = dir.join("impulse.f32");
    let fs_sidecar = dir.join("impulse.json");
    fs.write(&fs_impulse, &payload).unwrap();
    fs.write(&fs_sidecar, &sidecar_bytes).unwrap();
    assert_eq!(
        fs.read(&fs_impulse).unwrap(),
        Some(payload.clone()),
        "{CASE}: filesystem transport altered impulse bytes"
    );
    assert_eq!(
        fs.read(&fs_sidecar).unwrap(),
        Some(sidecar_bytes.clone()),
        "{CASE}: filesystem transport altered sidecar bytes"
    );
    assert_eq!(
        fs.read(&dir.join("missing.f32")).unwrap(),
        None,
        "{CASE}: absent file must read as absent"
    );

    // Decoded f32 values sit within half a quantum of the golden f64s.
    let mut worst = 0.0f64;
    for (index, &expected) in samples.iter().enumerate() {
        let decoded = f32::from_le_bytes([
            payload[4 * index],
            payload[4 * index + 1],
            payload[4 * index + 2],
            payload[4 * index + 3],
        ]) as f64;
        let err = (decoded - expected).abs();
        worst = worst.max(err);
        assert!(
            err <= 6e-8 * expected.abs() + 1e-12,
            "{CASE}: sample {index} decode drift {err:.3e}"
        );
    }
    assert!(
        worst <= TOL,
        "{CASE}: worst decode drift {worst:.3e} exceeds {TOL:.1e}"
    );
    std::fs::remove_file(&fs_impulse).ok();
    std::fs::remove_file(&fs_sidecar).ok();
    std::fs::remove_dir(&dir).ok();

    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: worst,
        tolerance: TOL,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
