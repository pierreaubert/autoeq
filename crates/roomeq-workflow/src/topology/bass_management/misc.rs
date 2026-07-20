use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::{MeasurementSource, RoomConfig, SpeakerConfig, SystemConfig};

/// Resolve a logical role to the single measurement source owned by workflow.
pub(crate) fn resolve_single_source<'a>(
    role: &str,
    config: &'a RoomConfig,
    sys: &SystemConfig,
) -> Result<&'a MeasurementSource> {
    let meas_key = sys
        .speakers
        .get(role)
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: format!("Missing speaker mapping for '{role}'"),
        })?;
    let speaker =
        config
            .speakers
            .get(meas_key)
            .ok_or_else(|| AutoeqError::InvalidConfiguration {
                message: format!("Missing speaker config for key '{meas_key}'"),
            })?;
    match speaker {
        SpeakerConfig::Single(source) => Ok(source),
        _ => Err(AutoeqError::InvalidConfiguration {
            message: format!("Workflow requires Single speaker config for '{role}'"),
        }),
    }
}
