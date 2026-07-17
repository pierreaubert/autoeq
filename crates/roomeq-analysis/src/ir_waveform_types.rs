use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Impulse response waveform (time-domain).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IrWaveform {
    /// Time axis in milliseconds.
    pub time_ms: Vec<f64>,
    /// Amplitude normalized so the pre-correction peak is one.
    pub amplitude: Vec<f64>,
}
