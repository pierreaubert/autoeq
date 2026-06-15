use crate::loss::{DriversLossData, drivers_flat_loss, multisub_flat_loss};
use crate::optim::loss::{Objective, ObjectiveContext};

/// Mode selector for the driver/subwoofer objective.
#[derive(Debug, Clone, Copy)]
pub enum DriversMode {
    /// Multi-driver crossover optimization.
    Flat,
    /// Multi-subwoofer gain/delay optimization.
    MultiSub,
}

/// Multi-driver / multi-subwoofer objective for [`LossType::DriversFlat`] and
/// [`LossType::MultiSubFlat`].
#[derive(Debug, Clone)]
pub struct DriversStrategy {
    pub data: DriversLossData,
    pub mode: DriversMode,
    pub fixed_crossover_freqs: Option<Vec<f64>>,
}

impl Objective for DriversStrategy {
    fn compute(&self, x: &[f64], ctx: &ObjectiveContext) -> f64 {
        let n_drivers = self.data.drivers.len();
        let gains = &x[0..n_drivers];
        let delays = &x[n_drivers..2 * n_drivers];

        match self.mode {
            DriversMode::Flat => {
                let xover_freqs: Vec<f64> = if let Some(ref fixed) = self.fixed_crossover_freqs {
                    fixed.clone()
                } else {
                    let xover_freqs_log10 = &x[2 * n_drivers..];
                    xover_freqs_log10.iter().map(|f| 10.0_f64.powf(*f)).collect()
                };
                drivers_flat_loss(
                    &self.data,
                    gains,
                    &xover_freqs,
                    Some(delays),
                    ctx.srate,
                    ctx.min_freq,
                    ctx.max_freq,
                )
            }
            DriversMode::MultiSub => {
                multisub_flat_loss(&self.data, gains, delays, ctx.srate, ctx.min_freq, ctx.max_freq)
            }
        }
    }
}
