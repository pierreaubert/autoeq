//! Native preparation of canonical RoomEQ channel transfer responses.

// Rust guideline compliant 2026-02-21
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

use autoeq_core::Curve;
use autoeq_core::error::AutoeqError;
use autoeq_core::serialized_chain::ConvolutionIrProvider;
use roomeq_engine::dsp_realization::RealizedDsp;
use roomeq_model::{ChannelDspChain, CurveData, PluginConfigWrapper};
use roomeq_workflow::{FrozenBundleResource, OutputBundleVerification};
use serde::Serialize;

use crate::LoadedReportInput;

// Prevent very large hydrated grids from producing oversized host report payloads.
const MAX_CURVE_POINTS: usize = 100_000;
// Four million f32 WAV taps occupy at most 16 MiB on disk and 32 MiB in this f64 cache.
const MAX_FIR_TAPS: usize = 4_000_000;
// Bound retained tap memory across all referenced FIRs in one report request.
const MAX_TOTAL_FIR_TAPS: usize = 16_000_000;
// Scalar FIR evaluation is proportional to grid points times taps per FIR stage.
const MAX_RESPONSE_WORK: u64 = 100_000_000;

/// A captured FIR decoded from the immutable manifested bundle snapshot.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CapturedFirResource {
    /// Sample rate encoded by the WAV and checked against the requested replay rate.
    pub sample_rate_hz: u32,
    /// Exact mono float samples consumed by the canonical engine realization.
    pub taps: Vec<f64>,
}

/// Status of one channel's canonical small-signal DSP replay.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DspReplayState {
    /// The canonical engine evaluated the serialized chain on the measurement grid.
    Available {
        /// Electrical transfer magnitude and phase on the input measurement grid.
        transfer_curve: CurveData,
    },
    /// No replay was produced because the required evidence or realization was unavailable.
    Unavailable {
        /// Stable human-readable reason from the preparation boundary or engine.
        reason: String,
    },
}

/// Prepared channel responses and FIR evidence from one frozen report input.
#[derive(Debug, Clone, Serialize)]
pub struct DspReplayReport {
    /// Requested sample rate in hertz.
    pub sample_rate_hz: u32,
    /// Explicit statement of graph-wide limitations, if any.
    pub unavailable_reason: Option<String>,
    /// Channel replay results keyed by the graph's stable channel key.
    pub channels: BTreeMap<String, DspReplayState>,
    /// Captured FIR samples keyed by the exact graph ir_file reference.
    pub captured_fir_resources: BTreeMap<String, CapturedFirResource>,
}

/// Invalid top-level settings for native DSP replay preparation.
#[derive(Debug, thiserror::Error)]
pub enum DspReplayError {
    /// Replay rates must be finite, positive whole-number hertz values.
    #[error("invalid DSP replay sample rate {0}; expected a positive integer number of hertz")]
    InvalidSampleRate(f64),
}

/// Prepare small-signal transfer curves and captured FIR data without reopening bundle paths.
pub(super) fn prepare(
    input: &LoadedReportInput,
    sample_rate_hz: f64,
) -> Result<DspReplayReport, DspReplayError> {
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || sample_rate_hz > f64::from(u32::MAX)
        || sample_rate_hz.fract() != 0.0
    {
        return Err(DspReplayError::InvalidSampleRate(sample_rate_hz));
    }
    let sample_rate_hz = sample_rate_hz as u32;
    let graph = input.graph();
    let mut fir_provider = FrozenFirProvider::new(input);
    let mut preload_failures = BTreeMap::new();

    for (channel_key, channel) in &graph.channels {
        if let Err(reason) = fir_provider.preload_chain(channel, sample_rate_hz) {
            preload_failures.insert(channel_key.clone(), reason);
        }
    }

    let unavailable_reason = if graph.global_plugins.is_empty() {
        None
    } else {
        Some(
            "graph contains global routing plugins; channel-only replay cannot represent routed sources"
                .to_owned(),
        )
    };
    let mut channels = BTreeMap::new();
    for (channel_key, channel) in &graph.channels {
        let state = if let Some(reason) = &unavailable_reason {
            DspReplayState::Unavailable {
                reason: reason.clone(),
            }
        } else if let Some(reason) = preload_failures.get(channel_key) {
            DspReplayState::Unavailable {
                reason: reason.clone(),
            }
        } else if let Some(curve) = &channel.initial_curve {
            match replay_channel(channel, curve, sample_rate_hz, &mut fir_provider) {
                Ok(transfer_curve) => DspReplayState::Available { transfer_curve },
                Err(reason) => DspReplayState::Unavailable { reason },
            }
        } else {
            DspReplayState::Unavailable {
                reason: "channel has no hydrated initial_curve".to_owned(),
            }
        };
        channels.insert(channel_key.clone(), state);
    }

    Ok(DspReplayReport {
        sample_rate_hz,
        unavailable_reason,
        channels,
        captured_fir_resources: fir_provider.into_resources(),
    })
}

fn replay_channel(
    chain: &ChannelDspChain,
    curve_data: &CurveData,
    sample_rate_hz: u32,
    provider: &mut FrozenFirProvider<'_>,
) -> Result<CurveData, String> {
    let measurement_grid = Curve::from(curve_data.clone());
    measurement_grid
        .validate("native report DSP replay input curve")
        .map_err(|error| error.to_string())?;
    if measurement_grid.freq.len() > MAX_CURVE_POINTS {
        return Err(format!(
            "input curve contains {} points; replay limit is {MAX_CURVE_POINTS}",
            measurement_grid.freq.len()
        ));
    }
    let nyquist_hz = f64::from(sample_rate_hz) / 2.0;
    if measurement_grid
        .freq
        .iter()
        .any(|frequency| *frequency > nyquist_hz)
    {
        return Err(format!(
            "input curve extends above replay Nyquist frequency {nyquist_hz} Hz"
        ));
    }
    let estimated_work = estimate_response_work(chain, measurement_grid.freq.len(), provider)?;
    if estimated_work > MAX_RESPONSE_WORK {
        return Err(format!(
            "canonical FIR response work estimate {estimated_work} exceeds limit {MAX_RESPONSE_WORK}"
        ));
    }

    let mut realized = RealizedDsp::new(chain, f64::from(sample_rate_hz), provider)
        .map_err(|error| error.to_string())?;
    let unity_transfer = Curve::from(CurveData {
        freq: measurement_grid.freq.to_vec(),
        spl: vec![0.0; measurement_grid.freq.len()],
        phase: Some(vec![0.0; measurement_grid.freq.len()]),
        ..CurveData::default()
    });
    let replayed = realized
        .apply_to_curve(&unity_transfer)
        .map_err(|error| error.to_string())?;
    Ok(CurveData::from(&replayed))
}

fn estimate_response_work(
    chain: &ChannelDspChain,
    frequency_count: usize,
    provider: &FrozenFirProvider<'_>,
) -> Result<u64, String> {
    let mut work = 0_u64;
    for plugin in chain_plugins(chain) {
        if plugin.plugin_type != "convolution" {
            continue;
        }
        let ir_file = plugin
            .parameters
            .get("ir_file")
            .and_then(serde_json::Value::as_str)
            .filter(|path| !path.is_empty())
            .ok_or_else(|| "convolution plugin has no nonempty ir_file".to_owned())?;
        let taps = provider
            .resource(ir_file)
            .ok_or_else(|| format!("captured convolution resource '{ir_file}' is unavailable"))?
            .taps
            .len();
        work = work.saturating_add((frequency_count as u64).saturating_mul(taps as u64));
    }
    Ok(work)
}

fn chain_plugins(chain: &ChannelDspChain) -> impl Iterator<Item = &PluginConfigWrapper> {
    chain
        .drivers
        .iter()
        .flatten()
        .flat_map(|driver| driver.plugins.iter())
        .chain(chain.plugins.iter())
}

struct FrozenFirProvider<'a> {
    resources: BTreeMap<&'a str, &'a FrozenBundleResource>,
    loaded: BTreeMap<String, CapturedFirResource>,
    total_taps: usize,
    manifest_verified: bool,
}

impl<'a> FrozenFirProvider<'a> {
    fn new(input: &'a LoadedReportInput) -> Self {
        Self {
            resources: input
                .resources()
                .map(|resource| (resource.relative_path(), resource))
                .collect(),
            loaded: BTreeMap::new(),
            total_taps: 0,
            manifest_verified: input.verification() == OutputBundleVerification::ManifestVerified,
        }
    }

    fn preload_chain(
        &mut self,
        chain: &ChannelDspChain,
        sample_rate_hz: u32,
    ) -> Result<(), String> {
        let references: BTreeSet<&str> = chain_plugins(chain)
            .filter(|plugin| plugin.plugin_type == "convolution")
            .filter_map(|plugin| {
                plugin
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
            })
            .filter(|path| !path.is_empty())
            .collect();
        for reference in references {
            self.load(reference, sample_rate_hz)?;
        }
        Ok(())
    }

    fn load(&mut self, ir_file: &str, sample_rate_hz: u32) -> Result<(), String> {
        if self.loaded.contains_key(ir_file) {
            return Ok(());
        }
        if !self.manifest_verified {
            return Err(format!(
                "convolution resource '{ir_file}' is unavailable for legacy unverified input"
            ));
        }
        let resource = self
            .resources
            .get(ir_file)
            .ok_or_else(|| format!("manifested convolution resource '{ir_file}' is missing"))?;
        let decoded = decode_mono_float_wav(
            resource.bytes(),
            sample_rate_hz,
            ir_file,
            MAX_TOTAL_FIR_TAPS.saturating_sub(self.total_taps),
        )?;
        self.total_taps = self.total_taps.saturating_add(decoded.taps.len());
        self.loaded.insert(ir_file.to_owned(), decoded);
        Ok(())
    }

    fn resource(&self, ir_file: &str) -> Option<&CapturedFirResource> {
        self.loaded.get(ir_file)
    }

    fn into_resources(self) -> BTreeMap<String, CapturedFirResource> {
        self.loaded
    }
}

impl ConvolutionIrProvider for FrozenFirProvider<'_> {
    fn taps(&mut self, ir_file: &str, sample_rate: u32) -> autoeq_core::error::Result<&[f64]> {
        if let Err(message) = self.load(ir_file, sample_rate) {
            return Err(AutoeqError::InvalidConfiguration { message });
        }
        Ok(&self.loaded[ir_file].taps)
    }
}

fn decode_mono_float_wav(
    bytes: &[u8],
    expected_sample_rate_hz: u32,
    ir_file: &str,
    max_taps: usize,
) -> Result<CapturedFirResource, String> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|error| format!("captured FIR '{ir_file}' is not a readable WAV: {error}"))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!(
            "captured FIR '{ir_file}' must be mono; found {} channels",
            spec.channels
        ));
    }
    if spec.sample_rate != expected_sample_rate_hz {
        return Err(format!(
            "captured FIR '{ir_file}' has sample rate {} Hz; replay requires {expected_sample_rate_hz} Hz",
            spec.sample_rate
        ));
    }
    if spec.sample_format != hound::SampleFormat::Float || spec.bits_per_sample != 32 {
        return Err(format!(
            "captured FIR '{ir_file}' must use mono IEEE float32 samples; found {:?}/{}-bit",
            spec.sample_format, spec.bits_per_sample
        ));
    }
    let sample_count = usize::try_from(reader.duration())
        .map_err(|_| format!("captured FIR '{ir_file}' sample count does not fit this platform"))?;
    let supported_taps = MAX_FIR_TAPS.min(max_taps);
    if sample_count == 0 || sample_count > supported_taps {
        return Err(format!(
            "captured FIR '{ir_file}' has {sample_count} taps; remaining supported range is 1..={supported_taps}"
        ));
    }
    let taps = reader
        .samples::<f32>()
        .map(|sample| {
            sample
                .map(f64::from)
                .map_err(|error| format!("captured FIR '{ir_file}' has malformed samples: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if taps.iter().any(|tap| !tap.is_finite()) {
        return Err(format!(
            "captured FIR '{ir_file}' contains a nonfinite sample"
        ));
    }
    Ok(CapturedFirResource {
        sample_rate_hz: spec.sample_rate,
        taps,
    })
}
