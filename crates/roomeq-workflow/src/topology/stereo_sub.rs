//! Stereo bass management uses the same per-logical-input route model as
//! home cinema, including per-driver sub filters and deployed acceptance.

use super::types::{WorkflowAssembly, WorkflowExecutor};
use roomeq_engine::error::Result;
use roomeq_engine::room_result::RoomOptimizationResult;

pub(in super::super) struct Stereo21Executor;

impl WorkflowExecutor for Stereo21Executor {
    fn execute<'cfg, 'p, 's>(
        &self,
        assembly: &mut WorkflowAssembly<'cfg, 'p, 's>,
    ) -> Result<RoomOptimizationResult> {
        for role in ["L", "R"] {
            let key = assembly.sys.speakers.get(role).ok_or_else(|| {
                roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker mapping '{role}'"),
                }
            })?;
            let speaker = assembly.config.speakers.get(key).ok_or_else(|| {
                roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config key '{key}'"),
                }
            })?;
            if !matches!(speaker, roomeq_model::SpeakerConfig::Single(_)) {
                return Err(roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!("'{role}' must be a Single speaker config"),
                });
            }
        }

        let subwoofers = assembly.sys.subwoofers.as_ref().ok_or_else(|| {
            roomeq_engine::error::AutoeqError::InvalidConfiguration {
                message: "stereo bass routing requires system.subwoofers.outputs".to_string(),
            }
        })?;
        if subwoofers.outputs.is_empty() {
            return Err(roomeq_engine::error::AutoeqError::InvalidConfiguration {
                message: "stereo bass routing requires system.subwoofers.outputs".to_string(),
            });
        }
        for output in &subwoofers.outputs {
            if !assembly.config.speakers.contains_key(&output.speaker) {
                return Err(roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!(
                        "physical sub output '{}' references missing speaker config '{}'",
                        output.id, output.speaker
                    ),
                });
            }
        }

        super::home_cinema::HomeCinemaExecutor.execute(assembly)
    }
}
