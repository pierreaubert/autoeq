use super::super::dsp_response_cache::DspResponseCache;
use super::super::dsp_response_cache::channel_chain_response;
use super::super::types::CtcArtifact;
use num_complex::Complex64;
use roomeq_engine::ctc::fft_real_to_half_spectrum_f64;
use roomeq_model::{ChannelDspChain, PluginConfigWrapper};

#[test]
fn joint_room_eq_path_models_mixed_fir_iir_band_split() {
    let chain = test_channel_chain(
        vec![
            PluginConfigWrapper {
                plugin_type: "band_split".to_string(),
                parameters: serde_json::json!({
                    "frequency": 1_000.0,
                    "type": "LR24"
                }),
            },
            PluginConfigWrapper {
                plugin_type: "gain".to_string(),
                parameters: serde_json::json!({
                    "gain_db": -12.0,
                    "channels": [0, 1]
                }),
            },
            PluginConfigWrapper {
                plugin_type: "band_merge".to_string(),
                parameters: serde_json::json!({
                    "bands": 2
                }),
            },
        ],
        None,
    );
    let mut cache = DspResponseCache::new(48_000);
    let low = channel_chain_response(&chain, 100.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    let high = channel_chain_response(&chain, 10_000.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    assert!(low < 0.35, "low-band gain should attenuate LF, got {low}");
    assert!(high > 0.8, "high band should pass through, got {high}");
}

#[test]
fn joint_room_eq_path_models_driver_crossover_branches() {
    let low_driver = roomeq_model::DriverDspChain {
        name: "woofer".to_string(),
        index: 0,
        plugins: vec![PluginConfigWrapper {
            plugin_type: "crossover".to_string(),
            parameters: serde_json::json!({
                "type": "LR24",
                "frequency": 1_000.0,
                "output": "low"
            }),
        }],
        initial_curve: None,
    };
    let chain = test_channel_chain(Vec::new(), Some(vec![low_driver]));
    let mut cache = DspResponseCache::new(48_000);
    let low = channel_chain_response(&chain, 100.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    let high = channel_chain_response(&chain, 10_000.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    assert!(low > 0.9, "lowpass driver should pass LF, got {low}");
    assert!(high < 0.05, "lowpass driver should reject HF, got {high}");
}

#[test]
fn crossover_plugin_honors_declared_type_order() {
    let make = |type_str: &str| {
        test_channel_chain(
            vec![PluginConfigWrapper {
                plugin_type: "crossover".to_string(),
                parameters: serde_json::json!({
                    "type": type_str,
                    "frequency": 80.0,
                    "output": "high"
                }),
            }],
            None,
        )
    };
    let mut cache = DspResponseCache::new(48_000);
    let lr24 = channel_chain_response(&make("LR24"), 20.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    let lr48 = channel_chain_response(&make("LR48"), 20.0, 48_000.0, &mut cache)
        .unwrap()
        .norm();
    let lr24_db = 20.0 * lr24.log10();
    let lr48_db = 20.0 * lr48.log10();
    // Two octaves below an 80 Hz cutoff: an LR24 high-pass sits around
    // -40 dB while an LR48 high-pass is down ~90 dB. The realization must
    // follow the declared type, not a fixed LR4 response.
    assert!(lr24_db > -60.0, "LR24 high-pass at 20 Hz, got {lr24_db} dB");
    assert!(
        lr48_db < lr24_db - 30.0,
        "LR48 must attenuate well below LR24 at 20 Hz: {lr48_db} vs {lr24_db} dB"
    );
}

pub(super) fn test_channel_chain(
    plugins: Vec<PluginConfigWrapper>,
    drivers: Option<Vec<roomeq_model::DriverDspChain>>,
) -> ChannelDspChain {
    ChannelDspChain {
        channel: "left".to_string(),
        plugins,
        drivers,
        initial_curve: None,
        final_curve: None,
        eq_response: None,
        pre_ir: None,
        post_ir: None,
        fir_temporal_masking: None,
        direct_early_late_correction: None,
        target_curve: None,
    }
}

pub(super) fn artifact_filter_spectrum(
    artifact: &CtcArtifact,
    speaker: &str,
    target_ear: &str,
) -> Vec<Complex64> {
    let filter = artifact
        .filters
        .iter()
        .find(|filter| filter.speaker == speaker && filter.target_ear == target_ear)
        .unwrap_or_else(|| panic!("missing filter speaker='{speaker}', target_ear='{target_ear}'"));
    fft_real_to_half_spectrum_f64(&filter.taps, artifact.fir_taps)
}
