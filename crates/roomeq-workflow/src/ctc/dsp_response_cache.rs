use super::misc::{checked_sample_rate, read_wav_channels_f64};
use super::types::MatrixSpectrum;
use num_complex::Complex64;
use roomeq_engine::Curve;
use roomeq_engine::dsp_realization::{ConvolutionIrProvider, RealizedDsp};
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_model::{ChannelDspChain, SystemConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub(super) fn apply_room_eq_dsp_to_spectrum(
    spectrum: &mut MatrixSpectrum,
    sys: &SystemConfig,
    channels: &HashMap<String, ChannelDspChain>,
    sample_rate: f64,
    sidecar_dir: &Path,
) -> Result<()> {
    let speakers = spectrum.speakers.len();
    let fft_size = (spectrum.bins.len() - 1) * 2;
    let mut cache =
        DspResponseCache::with_sidecar_dir(checked_sample_rate(sample_rate)?, sidecar_dir);
    let mut responses_by_speaker = Vec::with_capacity(speakers);

    for speaker in &spectrum.speakers {
        let Some(channel_name) = sys.speakers.get(speaker) else {
            responses_by_speaker.push(vec![Complex64::new(1.0, 0.0); spectrum.bins.len()]);
            continue;
        };
        let Some(chain) = channels.get(channel_name) else {
            responses_by_speaker.push(vec![Complex64::new(1.0, 0.0); spectrum.bins.len()]);
            continue;
        };
        let mut realized = RealizedDsp::new(chain, sample_rate, &mut cache)?;
        let mut responses = Vec::with_capacity(spectrum.bins.len());
        for bin in 0..spectrum.bins.len() {
            let frequency_hz = bin as f64 * sample_rate / fft_size as f64;
            responses.push(realized.response_at(frequency_hz)?);
        }
        responses_by_speaker.push(responses);
    }

    #[allow(clippy::needless_range_loop)]
    for bin in 0..spectrum.bins.len() {
        for position in &mut spectrum.bins[bin] {
            for speaker_index in 0..speakers {
                let correction = responses_by_speaker[speaker_index][bin];
                for ear_index in 0..2 {
                    position.values[ear_index * speakers + speaker_index] *= correction;
                }
            }
        }
    }
    Ok(())
}

/// Apply the canonical serialized channel chain to an arbitrary measurement.
pub fn apply_channel_dsp_chain_to_curve(
    chain: &ChannelDspChain,
    curve: &Curve,
    sample_rate: f64,
) -> Result<Curve> {
    apply_channel_dsp_chain_to_curve_with_sidecar_dir(chain, curve, sample_rate, Path::new("."))
}

pub fn apply_channel_dsp_chain_to_curve_with_sidecar_dir(
    chain: &ChannelDspChain,
    curve: &Curve,
    sample_rate: f64,
    sidecar_dir: &Path,
) -> Result<Curve> {
    let mut cache =
        DspResponseCache::with_sidecar_dir(checked_sample_rate(sample_rate)?, sidecar_dir);
    RealizedDsp::new(chain, sample_rate, &mut cache)?.apply_to_curve(curve)
}

/// Apply a serialized channel chain while resolving not-yet-exported
/// convolution sidecars from their in-memory FIR taps.
///
/// RoomEQ builds convolution plugins before the output writer materializes the
/// referenced WAV files. Final acceptance checks must therefore replay the
/// exact chain from the retained taps instead of requiring the sidecar early.
pub fn apply_channel_dsp_chain_to_curve_with_embedded_irs(
    chain: &ChannelDspChain,
    curve: &Curve,
    sample_rate: f64,
    sidecar_dir: &Path,
    embedded_irs: &HashMap<String, Vec<f64>>,
) -> Result<Curve> {
    let mut cache =
        DspResponseCache::with_sidecar_dir(checked_sample_rate(sample_rate)?, sidecar_dir);
    for (ir_file, taps) in embedded_irs {
        let path = Path::new(ir_file);
        let resolved_path = if path.is_relative() {
            sidecar_dir.join(path)
        } else {
            path.to_path_buf()
        };
        cache.convolution_ir.insert(resolved_path, taps.clone());
    }
    RealizedDsp::new(chain, sample_rate, &mut cache)?.apply_to_curve(curve)
}

pub(super) struct DspResponseCache {
    sample_rate: u32,
    sidecar_dir: PathBuf,
    convolution_ir: HashMap<PathBuf, Vec<f64>>,
}

impl DspResponseCache {
    #[cfg(test)]
    pub(super) fn new(sample_rate: u32) -> Self {
        Self::with_sidecar_dir(sample_rate, Path::new("."))
    }

    pub(super) fn with_sidecar_dir(sample_rate: u32, sidecar_dir: &Path) -> Self {
        Self {
            sample_rate,
            sidecar_dir: sidecar_dir.to_path_buf(),
            convolution_ir: HashMap::new(),
        }
    }

    fn convolution_taps(&mut self, path: &Path) -> Result<&[f64]> {
        let resolved_path = if path.is_relative() {
            self.sidecar_dir.join(path)
        } else {
            path.to_path_buf()
        };
        if !self.convolution_ir.contains_key(&resolved_path) {
            let channels = read_wav_channels_f64(
                &resolved_path,
                self.sample_rate,
                "RoomEQ convolution IR WAV",
            )?;
            let Some(first_channel) = channels.into_iter().next() else {
                return Err(AutoeqError::InvalidMeasurement {
                    message: format!("convolution IR '{}' has no channels", path.display()),
                });
            };
            self.convolution_ir
                .insert(resolved_path.clone(), first_channel);
        }
        Ok(self
            .convolution_ir
            .get(&resolved_path)
            .map(Vec::as_slice)
            .expect("cached convolution IR"))
    }
}

impl ConvolutionIrProvider for DspResponseCache {
    fn taps(&mut self, ir_file: &str, sample_rate: u32) -> Result<&[f64]> {
        if sample_rate != self.sample_rate {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "DSP realization requested {sample_rate} Hz convolution from {} Hz cache",
                    self.sample_rate
                ),
            });
        }
        self.convolution_taps(Path::new(ir_file))
    }
}

#[cfg(test)]
pub(super) fn channel_chain_response(
    chain: &ChannelDspChain,
    frequency_hz: f64,
    sample_rate: f64,
    cache: &mut DspResponseCache,
) -> Result<Complex64> {
    RealizedDsp::new(chain, sample_rate, cache)?.response_at(frequency_hz)
}
