use roomeq_model::RoomConfig;
use std::fs;
use std::path::Path;

/// Validate configuration before running roomeq
pub(super) fn validate_config(config: &RoomConfig) -> Result<(), String> {
    // Validate optimizer config
    if config.optimizer.num_filters == 0 {
        return Err("num_filters must be greater than 0 for quality fuzzing".to_string());
    }

    if config.optimizer.min_freq >= config.optimizer.max_freq {
        return Err(format!(
            "min_freq ({}) must be less than max_freq ({})",
            config.optimizer.min_freq, config.optimizer.max_freq
        ));
    }

    if config.optimizer.min_q > config.optimizer.max_q {
        return Err(format!(
            "min_q ({}) must be less than or equal to max_q ({})",
            config.optimizer.min_q, config.optimizer.max_q
        ));
    }

    if config.optimizer.min_db > config.optimizer.max_db {
        return Err(format!(
            "min_db ({}) must be less than or equal to max_db ({})",
            config.optimizer.min_db, config.optimizer.max_db
        ));
    }

    if config.optimizer.max_iter == 0 {
        return Err("max_iter must be greater than 0".to_string());
    }

    // Validate speakers
    if config.speakers.is_empty() {
        return Err("No speakers configured".to_string());
    }

    Ok(())
}

pub(super) fn validate_roomeq_output(output_json_path: &Path) -> Result<f64, String> {
    let output_json = fs::read_to_string(output_json_path)
        .map_err(|e| format!("failed to read output JSON: {}", e))?;
    let output: serde_json::Value = serde_json::from_str(&output_json)
        .map_err(|e| format!("failed to parse output JSON: {}", e))?;

    let channels = output
        .get("channels")
        .and_then(|value| value.as_object())
        .ok_or_else(|| "output JSON is missing object field 'channels'".to_string())?;
    if channels.is_empty() {
        return Err("output JSON contains no channels".to_string());
    }

    // If any channel looks like a supporting-source output, ensure a Convolution
    // plugin was emitted for it.
    let support_channels: Vec<&String> = channels
        .keys()
        .filter(|k| k.ends_with("_support"))
        .collect();
    if !support_channels.is_empty() {
        let has_convolution = channels.values().any(|ch| {
            ch.get("plugins")
                .and_then(|p| p.as_array())
                .map(|plugins| {
                    plugins.iter().any(|plugin| {
                        plugin.get("plugin_type").and_then(|t| t.as_str()) == Some("convolution")
                    })
                })
                .unwrap_or(false)
        });
        if !has_convolution {
            return Err(
                "supporting-source channels present but no Convolution plugin found".to_string(),
            );
        }
    }

    if let Some(metadata) = output.get("metadata").and_then(|value| value.as_object()) {
        for key in ["pre_score", "post_score"] {
            if let Some(value) = metadata.get(key) {
                let score = value
                    .as_f64()
                    .ok_or_else(|| format!("metadata.{} is not numeric", key))?;
                if !score.is_finite() {
                    return Err(format!("metadata.{} is not finite", key));
                }
            }
        }
    }

    let metadata = output
        .get("metadata")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "output metadata is missing".to_string())?;
    let score = |key: &str| -> Result<f64, String> {
        let value = metadata
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| format!("metadata.{key} is missing or non-numeric"))?;
        if !value.is_finite() {
            return Err(format!("metadata.{key} is not finite"));
        }
        Ok(value)
    };
    let pre = score("pre_score")?;
    let post = score("post_score")?;
    if post >= pre {
        return Err(format!(
            "randomized quality case did not improve: post {post:.4} >= pre {pre:.4}"
        ));
    }

    let has_correction = channels.values().any(|channel| {
        let direct = channel
            .get("plugins")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten();
        let drivers = channel
            .get("drivers")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .flat_map(|driver| {
                driver
                    .get("plugins")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
            });
        direct.chain(drivers).any(|plugin| {
            plugin
                .get("plugin_type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        "eq" | "convolution" | "warped_biquad" | "kautz_filter"
                    )
                })
        })
    });
    if !has_correction {
        return Err("randomized quality case emitted no corrective filters".to_string());
    }

    Ok(post)
}
