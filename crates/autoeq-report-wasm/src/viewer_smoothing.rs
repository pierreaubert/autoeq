//! Optional display smoothing of the sampled curves already in a report.

use crate::schema::{Figure, Series, XScale};

/// Additional smoothing applied by a report viewer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ViewerSmoothing {
    #[default]
    None,
    Octave48,
    Octave24,
    Octave12,
    Octave6,
    Octave3,
    Octave2,
    Octave,
    Variable,
    Psychoacoustic,
    Erb,
}

impl ViewerSmoothing {
    /// Viewer choices in display order, with stable browser values and labels.
    pub const OPTIONS: [(Self, &'static str, &'static str); 11] = [
        (Self::None, "none", "None"),
        (Self::Octave48, "48", "1/48 octave"),
        (Self::Octave24, "24", "1/24 octave"),
        (Self::Octave12, "12", "1/12 octave"),
        (Self::Octave6, "6", "1/6 octave"),
        (Self::Octave3, "3", "1/3 octave"),
        (Self::Octave2, "2", "1/2 octave"),
        (Self::Octave, "1", "1/1 octave"),
        (Self::Variable, "var", "Var"),
        (Self::Psychoacoustic, "psycho", "Psychoacoustic"),
        (Self::Erb, "erb", "ERB"),
    ];

    /// Updates displayed samples from the original figure.
    ///
    /// Both figures represent the same plot. The destination retains its
    /// viewport, series visibility, and other presentation state.
    pub fn apply(self, original: &Figure, destination: &mut Figure) {
        if !matches!(original.x.scale, XScale::Log) {
            return;
        }
        for (source, target) in original.series.iter().zip(&mut destination.series) {
            target.y.clone_from(&source.y);
            if self != Self::None {
                smooth_samples(self, source, &mut target.y);
            }
        }
    }

    fn edges(self, hz: f64, center: f64) -> [f64; 2] {
        let half_width = match self {
            Self::None => 0.0,
            Self::Octave48 => 0.5 / 48.0,
            Self::Octave24 => 0.5 / 24.0,
            Self::Octave12 => 0.5 / 12.0,
            Self::Octave6 => 0.5 / 6.0,
            Self::Octave3 => 0.5 / 3.0,
            Self::Octave2 => 0.5 / 2.0,
            Self::Octave => 0.5,
            Self::Variable => 0.5 * interpolate(hz, 100.0, 10_000.0, 1.0 / 48.0, 1.0 / 3.0),
            Self::Psychoacoustic => 0.5 * interpolate(hz, 100.0, 1_000.0, 1.0 / 3.0, 1.0 / 6.0),
            // Glasberg–Moore equivalent rectangular bandwidth, with f in kHz.
            Self::Erb => 0.5 * (107.77 * (hz / 1_000.0) + 24.673),
        };
        [center - half_width, center + half_width]
    }
}

fn interpolate(hz: f64, low_hz: f64, high_hz: f64, low: f64, high: f64) -> f64 {
    let fraction = (hz.clamp(low_hz, high_hz) / low_hz).ln() / (high_hz / low_hz).ln();
    low + (high - low) * fraction
}

fn smooth_samples(mode: ViewerSmoothing, series: &Series, output: &mut [Option<f64>]) {
    if series.x.len() != series.y.len()
        || series.x.iter().any(|&hz| !hz.is_finite() || hz <= 0.0)
        || series.x.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return;
    }
    let psychoacoustic = mode == ViewerSmoothing::Psychoacoustic;
    let keys: Vec<_> = series
        .x
        .iter()
        .map(|&hz| {
            if mode == ViewerSmoothing::Erb {
                hz
            } else {
                hz.log2()
            }
        })
        .collect();
    let samples: Vec<_> = series
        .y
        .iter()
        .map(|value| {
            value
                .map(|db| {
                    if psychoacoustic {
                        10.0_f64.powf(0.15 * db)
                    } else {
                        db
                    }
                })
                .filter(|value| value.is_finite())
        })
        .collect();

    // Both edges advance monotonically for every mode, even on irregular grids.
    let (mut left, mut right, mut sum, mut count) = (0, 0, 0.0, 0);
    for (index, target) in output.iter_mut().enumerate() {
        if samples[index].is_none() {
            continue;
        }
        let [low, high] = mode.edges(series.x[index], keys[index]);
        while right < keys.len() && keys[right] <= high {
            if let Some(value) = samples[right] {
                sum += value;
                count += 1;
            }
            right += 1;
        }
        while left < right && keys[left] < low {
            if let Some(value) = samples[left] {
                sum -= value;
                count -= 1;
            }
            left += 1;
        }
        let mean = sum / f64::from(count);
        let smoothed = if psychoacoustic {
            (20.0 / 3.0) * mean.log10()
        } else {
            mean
        };
        if smoothed.is_finite() {
            *target = Some(smoothed);
        }
    }
}
