//! Scale the correction bank while preserving the Kautz dry path and poles.

use roomeq_model::Result;
use serde_json::Value;

use super::failed;

/// Interpolate linear bank weights without reinterpreting them as dB gains.
pub(super) fn scale_weights(filter: &mut Value, strength: f64) -> Result<()> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(failed(
            "Kautz correction strength must be finite and in [0, 1]",
        ));
    }
    // Validate and mutate a copy so a malformed later section cannot leave
    // earlier weights scaled. The outer selection also retains its snapshot.
    let mut scaled = filter.clone();
    let key = match (scaled.get("kautz_sections"), scaled.get("sections")) {
        (Some(_), Some(_)) => return Err(failed("duplicate Kautz section fields")),
        (Some(_), None) => Some("kautz_sections"),
        (None, Some(_)) => Some("sections"),
        (None, None) => None,
    };
    let mut has_sections = false;
    if let Some(key) = key {
        let sections = scaled
            .get_mut(key)
            .and_then(Value::as_array_mut)
            .ok_or_else(|| failed("Kautz sections must be an array"))?;
        has_sections = !sections.is_empty();
        for section in sections {
            scale_weight(section, "gain", strength)?;
        }
    }
    if !has_sections {
        // The playback consumer's legacy single-section db_gain spelling
        // also holds a linear weight. Empty lists use this same fallback.
        scale_weight(&mut scaled, "db_gain", strength)?;
    }
    *filter = scaled;
    Ok(())
}

fn scale_weight(object: &mut Value, field: &str, strength: f64) -> Result<()> {
    let object = object
        .as_object_mut()
        .ok_or_else(|| failed("Kautz section must be an object"))?;
    if let Some(value) = object.get_mut(field) {
        let weight = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| failed("Kautz weight must be a finite number"))?;
        *value = serde_json::json!(weight * strength);
    }
    // Omitted weights remain omitted; playback defaults them to zero.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kautz_strength_preserves_aliases_defaults_poles_and_order() {
        for key in ["kautz_sections", "sections"] {
            let mut filter = json!({
                "topology": "kautz_filter", "db_gain": 99.0,
                key: [
                    {"pole_freq": 75.0, "q": 8.0, "gain": -0.04},
                    {"frequency": 95.0, "q": 4.0},
                    {"pole_freq_hz": 135.0, "q": 10.0, "gain": 0.02}
                ]
            });
            let original = filter.clone();
            scale_weights(&mut filter, 0.5).unwrap();
            let mut expected = original;
            expected[key][0]["gain"] = json!(-0.02);
            expected[key][2]["gain"] = json!(0.01);
            assert_eq!(filter, expected);
            assert_eq!(filter[key].as_array().unwrap().len(), 3);
        }
    }

    #[test]
    fn kautz_strength_scales_legacy_fallback_linear_weight() {
        for sections in [None, Some("kautz_sections"), Some("sections")] {
            let mut filter =
                json!({"topology": "kautz_filter", "freq": 75.0, "q": 8.0, "db_gain": -0.04});
            if let Some(key) = sections {
                filter[key] = json!([]);
            }
            scale_weights(&mut filter, 0.5).unwrap();
            assert_eq!(filter["db_gain"], -0.02);
            scale_weights(&mut filter, 0.0).unwrap();
            assert_eq!(filter["db_gain"], 0.0);
        }
    }

    #[test]
    fn kautz_strength_rejects_malformed_weights_without_partial_mutation() {
        for mut filter in [
            json!({"kautz_sections": null}),
            json!({"kautz_sections": [], "sections": []}),
            json!({"kautz_sections": [{"gain": 1.0}, {"gain": "bad"}]}),
            json!({"kautz_sections": [null]}),
            json!({"sections": [{"gain": null}]}),
            json!({"db_gain": "bad"}),
        ] {
            let original = filter.clone();
            assert!(scale_weights(&mut filter, 0.5).is_err());
            assert_eq!(filter, original);
        }
        for strength in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut filter = json!({"db_gain": 0.04});
            let original = filter.clone();
            assert!(scale_weights(&mut filter, strength).is_err());
            assert_eq!(filter, original);
        }
    }
}
