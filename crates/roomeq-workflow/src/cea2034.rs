use autoeq_measurements::read;
use autoeq_measurements::read::Cea2034Data;
use log::{info, warn};
use roomeq_model::RoomConfig;
use std::collections::HashMap;

/// Fetch CEA2034 data for a speaker from spinorama.org (blocking).
///
/// Creates a tokio runtime internally to call the async API.
/// Reuses the existing disk cache in `~/.local/share/autoeq/data_cached/speakers/`.
pub fn fetch_cea2034_blocking(
    speaker_name: &str,
    version: &str,
) -> std::result::Result<Cea2034Data, Box<dyn std::error::Error>> {
    let fetch = async {
        let plot_data = read::fetch_measurement_plot_data(speaker_name, version, "CEA2034").await?;
        let curves = read::extract_cea2034_curves_original(&plot_data, "CEA2034")?;
        read::build_cea2034_data(curves)
    };

    // If already inside a tokio runtime (e.g. called from async context),
    // use the existing runtime to avoid "Cannot start a runtime from within a runtime" panic.
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(fetch))
    } else {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(fetch)
    }
}

/// Pre-fetch CEA2034 data for all speakers that have `speaker_name` set.
///
/// Iterates the room config's speakers, resolves speaker names, and fetches
/// CEA2034 data for each. Returns a cache keyed by speaker name.
/// Logs warnings for any speaker whose data cannot be fetched.
pub fn pre_fetch_all_cea2034(config: &RoomConfig) -> HashMap<String, Cea2034Data> {
    let cea_config = match &config.optimizer.cea2034_correction {
        Some(c) if c.enabled => c,
        _ => return HashMap::new(),
    };

    let mut cache = HashMap::new();

    for speaker_config in config.speakers.values() {
        // Resolve speaker name: cea2034_correction.speaker_name overrides per-speaker name
        let speaker_name = cea_config
            .speaker_name
            .as_deref()
            .or_else(|| speaker_config.speaker_name());

        if let Some(name) = speaker_name {
            if cache.contains_key(name) {
                continue; // Already fetched (e.g., same speaker model for L and R)
            }

            info!("  Fetching CEA2034 data for speaker '{}'...", name);
            match fetch_cea2034_blocking(name, &cea_config.version) {
                Ok(data) => {
                    info!(
                        "  CEA2034 data loaded: {} frequency points",
                        data.listening_window.freq.len()
                    );
                    cache.insert(name.to_string(), data);
                }
                Err(e) => {
                    warn!(
                        "  Failed to fetch CEA2034 data for '{}': {}. \
                         Speaker correction will be skipped for this speaker.",
                        name, e
                    );
                }
            }
        }
    }

    cache
}
