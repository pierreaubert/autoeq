use ndarray::Array1;
use num_complex::Complex64;

/// One independently generated source in a parallel acoustic sum.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelSourceParameters {
    /// Linear level adjustment in dB.
    pub gain_db: f64,
    /// Propagation or DSP delay in milliseconds.
    pub delay_ms: f64,
    /// Whether polarity is inverted.
    pub inverted: bool,
}

/// Ground-truth parameters used to generate an acoustic oracle.
#[derive(Debug, Clone, PartialEq)]
pub enum OracleParameters {
    Identity,
    Delay {
        delay_ms: f64,
    },
    Polarity {
        inverted: bool,
    },
    AllPass {
        center_hz: f64,
        q: f64,
    },
    LinkwitzRiley4 {
        crossover_hz: f64,
    },
    RoomMode {
        center_hz: f64,
        q: f64,
        peak_gain_db: f64,
    },
    ParallelWoofers {
        sources: Vec<ParallelSourceParameters>,
    },
    ExcessPhase {
        sections: Vec<(f64, f64)>,
    },
    CombNull {
        reflection_gain: f64,
        reflection_delay_ms: f64,
    },
    RoomTransition {
        volume_m3: f64,
        rt60_seconds: f64,
        schroeder_hz: f64,
        transition_octaves: f64,
    },
    BiquadCascade {
        section_count: usize,
    },
}

/// Behaviour that a candidate correction is forbidden to introduce.
#[derive(Debug, Clone, PartialEq)]
pub enum ProhibitedBehavior {
    NonFiniteTransfer,
    CorrectionOutsideRegion {
        max_abs_db: f64,
    },
    BoostIntoNull {
        center_hz: f64,
        half_width_octaves: f64,
        max_boost_db: f64,
    },
    GroupDelayResidual {
        max_rms_ms: f64,
    },
    Latency {
        max_ms: f64,
    },
    PreRinging {
        max_energy_db: f64,
    },
}

/// Analytic fixture with complex ground truth and safety metadata.
#[derive(Debug, Clone)]
pub struct AcousticOracle {
    pub name: String,
    pub frequencies_hz: Array1<f64>,
    pub expected_transfer: Array1<Complex64>,
    /// Independently useful branches, such as low/high crossover paths or
    /// individual parallel woofer responses.
    pub components: Vec<Array1<Complex64>>,
    pub generating_parameters: OracleParameters,
    pub valid_correction_region_hz: (f64, f64),
    pub prohibited_behaviors: Vec<ProhibitedBehavior>,
}

impl AcousticOracle {
    /// Validate the fixture boundary before it is used as ground truth.
    pub fn validate(&self) -> Result<(), String> {
        if !crate::roomeq::frequency_grid::is_valid_frequency_grid(&self.frequencies_hz) {
            return Err(format!(
                "oracle '{}' has an invalid frequency grid",
                self.name
            ));
        }
        if self.expected_transfer.len() != self.frequencies_hz.len() {
            return Err(format!(
                "oracle '{}' transfer length {} does not match frequency length {}",
                self.name,
                self.expected_transfer.len(),
                self.frequencies_hz.len()
            ));
        }
        if self.components.iter().any(|component| {
            component.len() != self.frequencies_hz.len()
                || component
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
        }) {
            return Err(format!("oracle '{}' has an invalid component", self.name));
        }
        if self
            .expected_transfer
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(format!("oracle '{}' has a non-finite transfer", self.name));
        }
        let (lo, hi) = self.valid_correction_region_hz;
        if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi <= lo {
            return Err(format!(
                "oracle '{}' has invalid correction region [{lo}, {hi}]",
                self.name
            ));
        }
        Ok(())
    }
}

/// Optional time-domain evidence evaluated alongside a complex transfer.
#[derive(Debug, Clone, Copy)]
pub struct ImpulseEvidence<'a> {
    pub samples: &'a [f64],
    pub sample_rate: f64,
}

/// Candidate DSP response to compare against an oracle.
#[derive(Debug, Clone, Copy)]
pub struct CandidateTransfer<'a> {
    pub transfer: &'a [Complex64],
    pub impulse: Option<ImpulseEvidence<'a>>,
}
