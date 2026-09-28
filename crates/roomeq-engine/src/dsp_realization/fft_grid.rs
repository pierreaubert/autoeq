//! Exact FIR transfer samples for complete one-sided FFT grids.

use super::{ConvolutionIrProvider, Result};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::collections::HashMap;

pub(super) fn fft_size(frequencies: &[f64], sample_rate: f64) -> Option<usize> {
    let upper_size = frequencies.len().checked_sub(1)?.checked_mul(2)?;
    if upper_size < 2 || frequencies.windows(2).any(|pair| pair[0] >= pair[1]) {
        return None;
    }
    // Electrical validation adds narrow EQ/crossover centers to a uniform
    // grid. Recognize its complete underlying grid without dropping anchors.
    let size = 1usize << (usize::BITS - 1 - upper_size.leading_zeros());
    let mut next_bin = 0;
    for frequency in frequencies {
        if *frequency == next_bin as f64 * sample_rate / size as f64 {
            next_bin += 1;
        }
    }
    (next_bin == size / 2 + 1).then_some(size)
}

pub(super) struct FftGrid<'a, P> {
    source: &'a mut P,
    size: usize,
    spectra: HashMap<String, Vec<Complex64>>,
    planner: FftPlanner<f64>,
}

impl<'a, P> FftGrid<'a, P> {
    pub(super) fn new(source: &'a mut P, size: usize) -> Self {
        Self {
            source,
            size,
            spectra: HashMap::new(),
            planner: FftPlanner::new(),
        }
    }
}

impl<P: ConvolutionIrProvider> ConvolutionIrProvider for FftGrid<'_, P> {
    fn taps(&mut self, ir_file: &str, sample_rate: u32) -> Result<&[f64]> {
        self.source.taps(ir_file, sample_rate)
    }

    fn response(
        &mut self,
        ir_file: &str,
        frequency_hz: f64,
        sample_rate: f64,
    ) -> Result<Complex64> {
        let index = (frequency_hz * self.size as f64 / sample_rate).round() as usize;
        if index > self.size / 2 || frequency_hz != index as f64 * sample_rate / self.size as f64 {
            // Preserve the exact requested off-grid frequency, including
            // narrow electrical peaks. Never interpolate safety evidence.
            return self.source.response(ir_file, frequency_hz, sample_rate);
        }
        if !self.spectra.contains_key(ir_file) {
            let taps = self
                .source
                .taps(ir_file, super::checked_sample_rate(sample_rate)?)?;
            let mut spectrum = vec![Complex64::default(); self.size];
            // The scalar evaluator defines an empty FIR as bypass.
            if taps.is_empty() {
                spectrum[0].re = 1.0;
            }
            // DFT samples are periodic in the tap index. Folding preserves
            // filters longer than the FFT instead of silently truncating them.
            for (index, tap) in taps.iter().enumerate() {
                spectrum[index % self.size].re += tap;
            }
            self.planner
                .plan_fft_forward(self.size)
                .process(&mut spectrum);
            spectrum.truncate(self.size / 2 + 1);
            self.spectra.insert(ir_file.to_owned(), spectrum);
        }
        Ok(self.spectra[ir_file][index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_complete_exact_fft_grids_qualify() {
        let grid: Vec<_> = (0..=32).map(|i| i as f64 * 48000.0 / 64.0).collect();
        assert_eq!(fft_size(&grid, 48000.0), Some(64));
        assert_eq!(fft_size(&[], 48000.0), None);
        assert_eq!(fft_size(&[0.0], 48000.0), None);
        assert_eq!(fft_size(&grid[1..], 48000.0), None);
        assert_eq!(fft_size(&grid[..32], 48000.0), None);
        assert_eq!(fft_size(&grid, 44100.0), None);
        let mut perturbed = grid.clone();
        perturbed[4] += 1e-8;
        assert_eq!(fft_size(&perturbed, 48000.0), None);
        let mut anchored = grid.clone();
        anchored.extend([30.1, 1000.3, 23999.0]);
        anchored.sort_by(f64::total_cmp);
        assert_eq!(fft_size(&anchored, 48000.0), Some(64));
    }
}
