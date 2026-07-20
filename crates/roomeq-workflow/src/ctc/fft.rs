//! FFT adapters for CTC resource processing.

use num_complex::Complex64;
use rustfft::FftPlanner;

pub(super) fn fft_real_to_half_spectrum(input: &[f32], fft_size: usize) -> Vec<Complex64> {
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];
    for (dst, value) in buffer.iter_mut().zip(input.iter().copied()) {
        dst.re = value as f64;
    }
    FftPlanner::<f64>::new()
        .plan_fft_forward(fft_size)
        .process(&mut buffer);
    buffer.truncate(fft_size / 2 + 1);
    buffer
}
