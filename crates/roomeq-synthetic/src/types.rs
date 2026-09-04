use crate::Curve;
use math_audio_iir_fir::Biquad;

/// Full synthetic scenario with known ground truth.
#[derive(Debug, Clone)]
pub struct SyntheticScenario {
    /// Human-readable name for the scenario
    pub name: String,
    /// The original target curve (what we want to achieve)
    pub perfect_curve: Curve,
    /// The degraded measurement (after noise + room modes)
    pub degraded_curve: Curve,
    /// The room modes that were applied
    pub known_modes: Vec<Biquad>,
    /// Pre-mode noise RMS in dB
    pub pre_noise_rms_db: f64,
    /// Post-mode noise RMS in dB
    pub post_noise_rms_db: f64,
}

/// Synthetic multi-sub scenario with per-subwoofer measurements.
#[derive(Debug, Clone)]
pub struct MultiSubSyntheticScenario {
    pub name: String,
    /// Perfect flat bass target (what we want the combined response to be)
    pub perfect_curve: Curve,
    /// Per-subwoofer degraded measurements (with phase from simulated delays)
    pub sub_curves: Vec<Curve>,
    /// Number of subwoofers
    pub n_subs: usize,
    /// Room modes shared by all subs (room resonances)
    pub shared_modes: Vec<Biquad>,
    /// Per-sub unique modes (placement-dependent resonances)
    pub per_sub_modes: Vec<Vec<Biquad>>,
}

/// Synthetic cardioid subwoofer scenario (front + rear sub with separation).
#[derive(Debug, Clone)]
pub struct CardioidSyntheticScenario {
    pub name: String,
    pub perfect_curve: Curve,
    pub front_curve: Curve,
    pub rear_curve: Curve,
    pub separation_meters: f64,
}

/// Synthetic Double Bass Array scenario (front array + rear array).
#[derive(Debug, Clone)]
pub struct DbaSyntheticScenario {
    pub name: String,
    pub perfect_curve: Curve,
    pub front_curves: Vec<Curve>,
    pub rear_curves: Vec<Curve>,
}

/// Synthetic modal-room scenario with known correctable and non-correctable features.
///
/// The room has fixed modal-peak frequencies (minimum-phase resonances an
/// optimizer should correct) and a fixed SBIR cancellation-notch frequency
/// (a position-dependent null no optimizer should boost into). Seat curves
/// share those frequencies but differ in depth, modelling seat-to-seat
/// variance of a real in-room capture. `seats[0]` is the training (main)
/// position; the rest are held-out seats.
#[derive(Debug, Clone)]
pub struct ModalRoomScenario {
    pub name: String,
    pub perfect_curve: Curve,
    pub seats: Vec<Curve>,
    pub correctable_peak_hz: Vec<f64>,
    pub non_correctable_notch_hz: Vec<f64>,
}
