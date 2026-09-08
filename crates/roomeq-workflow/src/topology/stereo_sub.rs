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
        let sub_role = roomeq_engine::home_cinema::bass_output_role(assembly.config, assembly.sys);
        for role in ["L", "R", sub_role.as_str()] {
            let key = assembly.sys.speakers.get(role).ok_or_else(|| {
                roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: if role == sub_role {
                        format!("Missing speaker mapping for '{role}'")
                    } else {
                        format!("Missing speaker mapping '{role}'")
                    },
                }
            })?;
            let speaker = assembly.config.speakers.get(key).ok_or_else(|| {
                roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!("Missing speaker config for key '{key}'"),
                }
            })?;
            if role != sub_role && !matches!(speaker, roomeq_model::SpeakerConfig::Single(_)) {
                return Err(roomeq_engine::error::AutoeqError::InvalidConfiguration {
                    message: format!("'{role}' must be a Single speaker config"),
                });
            }
        }
        super::home_cinema::HomeCinemaExecutor.execute(assembly)
    }
}
