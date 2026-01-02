//! Room EQ - Multi-channel room equalization optimizer
//!
//! Copyright (C) 2025 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use log::{debug, info, warn};
use rayon::prelude::*;
use schemars::schema_for;
use std::collections::HashMap;
use std::path::PathBuf;

// Include roomeq modules
mod crossover_optim;
mod dba_optim;
mod eq_optim;
mod fir_optim;
use autoeq::Curve;
use autoeq::read as load;
use num_complex::Complex64;
mod group_delay_optim;
mod multisub_optim;
mod output;
mod types;

use types::{ChannelDspChain, OptimizationMetadata, RoomConfig, SpeakerConfig};

/// Room EQ - Optimize multi-channel speaker systems
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to room configuration JSON file
    #[arg(short, long, required_unless_present = "schema")]
    config: Option<PathBuf>,

    /// Output DSP chain JSON file
    #[arg(short, long, required_unless_present = "schema")]
    output: Option<PathBuf>,

    /// Sample rate for filter design (default: 48000 Hz)
    #[arg(long, default_value_t = 48000.0)]
    sample_rate: f64,

    /// Verbose output (deprecated, use RUST_LOG env var)
    #[arg(short, long)]
    verbose: bool,

    /// Dump JSON schema for the output format
    #[arg(long)]
    schema: bool,
}

fn main() -> Result<()> {
    // Initialize logger safely
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if args.schema {
        let schema = schema_for!(types::DspChainOutput);
        println!("{}", serde_json::to_string_pretty(&schema).unwrap());
        return Ok(());
    }

    if args.verbose {
        warn!("The --verbose flag is deprecated. Use RUST_LOG=debug instead.");
    }

    // Unwrap required args (safe because of required_unless_present)
    let config_path = args
        .config
        .ok_or_else(|| anyhow!("Config file is required"))?;
    let output_path = args
        .output
        .ok_or_else(|| anyhow!("Output file is required"))?;

    run(args.sample_rate, config_path, output_path)
}

fn run(sample_rate: f64, config_path: PathBuf, output_path: PathBuf) -> Result<()> {
    // Load room configuration
    info!("Loading room configuration from {:?}", config_path);

    let config_json = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

    let room_config: RoomConfig = serde_json::from_str(&config_json)
        .with_context(|| "Failed to parse room configuration JSON")?;

    info!("Found {} speakers", room_config.speakers.len());

    let output_dir = output_path.parent().unwrap_or(std::path::Path::new("."));

    // Process each speaker
    let mut channel_chains = HashMap::new();
    let mut pre_scores = Vec::new();
    let mut post_scores = Vec::new();

    // Process in parallel
    let results: Vec<Result<(String, ChannelDspChain, f64, f64, Curve)>> = room_config
        .speakers
        .par_iter()
        .map(|(channel_name, speaker_config)| {
            info!("Processing channel: {}", channel_name);

            let (chain, pre_score, post_score, final_curve) = process_speaker(
                channel_name,
                speaker_config,
                &room_config,
                sample_rate,
                output_dir,
            )?;

            Ok((
                channel_name.clone(),
                chain,
                pre_score,
                post_score,
                final_curve,
            ))
        })
        .collect();

    let mut curves = HashMap::new();

    // Collect results
    for res in results {
        let (channel_name, chain, pre_score, post_score, final_curve) = res?;
        channel_chains.insert(channel_name.clone(), chain);
        curves.insert(channel_name, final_curve);
        pre_scores.push(pre_score);
        post_scores.push(post_score);
    }

    // Group Delay Optimization
    if let Some(gd_configs) = &room_config.group_delay {
        info!("Optimizing group delay alignments...");
        for gd_config in gd_configs {
            let sub_curve = match curves.get(&gd_config.subwoofer) {
                Some(c) => c,
                None => {
                    warn!(
                        "Subwoofer channel '{}' not found for group delay optimization",
                        gd_config.subwoofer
                    );
                    continue;
                }
            };

            for speaker_name in &gd_config.speakers {
                if let Some(speaker_curve) = curves.get(speaker_name) {
                    info!(
                        "  Aligning '{}' with '{}'",
                        speaker_name, gd_config.subwoofer
                    );
                    let delay_res = group_delay_optim::optimize_group_delay(
                        sub_curve,
                        speaker_curve,
                        gd_config.min_freq,
                        gd_config.max_freq,
                    );

                    match delay_res {
                        Ok(delay_ms) => {
                            info!(
                                "    Optimal relative delay: {:.3} ms (positive = delay speaker)",
                                delay_ms
                            );

                            // Apply delay
                            // If delay > 0, delay speaker.
                            // If delay < 0, delay subwoofer.
                            // BUT, we have one subwoofer and multiple speakers.
                            // And maybe multiple GD configs sharing the subwoofer.
                            // Simpler approach: Apply delay to speaker relative to sub.
                            // If speaker needs to be advanced (delay < 0), we delay the sub?
                            // This creates conflicts.
                            //
                            // Robust approach: delay the speaker by `delay_ms` if > 0.
                            // If < 0, we can't really do anything unless we delay the sub, which affects others.
                            // For now, let's just apply positive delays to speakers, or if negative, warn user.
                            // OR, we can add a 'global' sub delay if all speakers need it.

                            if delay_ms > 0.0 {
                                if let Some(chain) = channel_chains.get_mut(speaker_name) {
                                    output::add_delay_plugin(chain, delay_ms);
                                    info!(
                                        "    Applied {:.3} ms delay to '{}'",
                                        delay_ms, speaker_name
                                    );
                                }
                            } else if delay_ms < 0.0 {
                                // Try to delay sub?
                                // Only if this sub is not used by others or we accept global shift.
                                // For now, let's just warn.
                                warn!(
                                    "    Speaker '{}' should be advanced by {:.3} ms relative to sub. This requires delaying the sub, which is not automatically handled per-speaker pair.",
                                    speaker_name, -delay_ms
                                );
                                // Optional: Apply delay to sub if it's the only pair?
                                // Better: Apply delay to sub and shift all other speakers? Too complex for now.
                            }
                        }
                        Err(e) => {
                            warn!("    Group delay optimization failed: {}", e);
                        }
                    }
                } else {
                    warn!("Speaker channel '{}' not found", speaker_name);
                }
            }
        }
    }

    // Aggregate scores (average across channels)
    let avg_pre_score = if !pre_scores.is_empty() {
        pre_scores.iter().sum::<f64>() / pre_scores.len() as f64
    } else {
        0.0
    };
    let avg_post_score = if !post_scores.is_empty() {
        post_scores.iter().sum::<f64>() / post_scores.len() as f64
    } else {
        0.0
    };

    info!(
        "Average pre-score: {:.4}, post-score: {:.4}",
        avg_pre_score, avg_post_score
    );

    // Create DSP chain output
    let dsp_output = output::create_dsp_chain_output(
        channel_chains,
        Some(OptimizationMetadata {
            pre_score: avg_pre_score,
            post_score: avg_post_score,
            algorithm: room_config.optimizer.algorithm.clone(),
            iterations: room_config.optimizer.max_iter,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }),
    );

    // Save output
    info!("Saving DSP chain to {:?}", output_path);

    output::save_dsp_chain(&dsp_output, &output_path)
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| format!("Failed to save DSP chain to {:?}", output_path))?;

    info!("Done!");

    Ok(())
}

/// Process a single speaker (simple or group)
///
/// Returns: (DSP chain, pre_score, post_score)
fn process_speaker(
    channel_name: &str,
    speaker_config: &SpeakerConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &std::path::Path,
) -> Result<(ChannelDspChain, f64, f64, Curve)> {
    match speaker_config {
        SpeakerConfig::Single(source) => {
            // Simple case: single measurement
            process_single_speaker(channel_name, source, room_config, sample_rate, output_dir)
        }
        SpeakerConfig::Group(group) => {
            // Multi-driver case: optimize crossover and EQ
            process_speaker_group(channel_name, group, room_config, sample_rate, output_dir)
        }
        SpeakerConfig::MultiSub(group) => {
            // Multi-subwoofer optimization
            process_multisub_group(channel_name, group, room_config, sample_rate, output_dir)
        }
        SpeakerConfig::Dba(config) => {
            // DBA optimization
            process_dba(channel_name, config, room_config, sample_rate, output_dir)
        }
    }
}

/// Process a simple speaker with a single measurement
///
/// Returns: (DSP chain, pre_score, post_score)
fn process_single_speaker(
    channel_name: &str,
    source: &types::MeasurementSource,
    room_config: &RoomConfig,
    sample_rate: f64,
    output_dir: &std::path::Path,
) -> Result<(ChannelDspChain, f64, f64, Curve)> {
    // Load measurement
    let curve = load::load_source(source)
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| format!("Failed to load measurement for channel {}", channel_name))?;

    debug!(
        "  Loaded measurement: {:.1} Hz - {:.1} Hz",
        curve.freq[0],
        curve.freq[curve.freq.len() - 1]
    );

    // Compute pre-score: normalize curve and compute flat loss
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;

    // Normalize curve (subtract mean in evaluation range)
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..curve.freq.len() {
        if curve.freq[i] >= min_freq && curve.freq[i] <= max_freq {
            sum += curve.spl[i];
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let normalized_spl = &curve.spl - mean;
    let pre_score = autoeq::loss::flat_loss(&curve.freq, &normalized_spl, min_freq, max_freq);

    match room_config.optimizer.mode.as_str() {
        "fir" => {
            // FIR mode
            info!("  Generating FIR filter...");
            let coeffs = fir_optim::generate_fir_correction(
                &curve,
                &room_config.optimizer,
                room_config.target_curve.as_ref(),
                sample_rate,
            )
            .map_err(|e| anyhow!("FIR generation failed: {}", e))?;

            // Save WAV
            let filename = format!("{}_fir.wav", channel_name);
            let wav_path = output_dir.join(&filename);
            autoeq::fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                .map_err(|e| anyhow!("Failed to save FIR WAV: {}", e))?;

            info!("  Saved FIR filter to {}", wav_path.display());

            // Build chain
            let plugin = output::create_convolution_plugin(&filename);
            let chain = types::ChannelDspChain {
                channel: channel_name.to_string(),
                plugins: vec![plugin],
                drivers: None,
            };

            // TODO: Compute FIR response properly. For now returning input curve.
            Ok((chain, pre_score, 0.0, curve))
        }
        "mixed" => {
            // Mixed mode
            // 1. Optimize IIR
            let (eq_filters, post_iir_score) = eq_optim::optimize_channel_eq(
                &curve,
                &room_config.optimizer,
                room_config.target_curve.as_ref(),
                sample_rate,
            )
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("IIR optimization failed for channel {}", channel_name))?;

            info!(
                "  IIR stage: {} filters, score={:.6}",
                eq_filters.len(),
                post_iir_score
            );

            // 2. Compute IIR response
            let final_curve_iir = apply_filter_response(&curve, &eq_filters, sample_rate);

            // 3. Create residual curve (Measurement + IIR)
            // apply_filter_response already returns Curve with updated SPL/Phase
            let input_plus_iir = final_curve_iir.clone();

            // 4. Generate FIR for residual
            info!("  Generating FIR for residual...");
            let coeffs = fir_optim::generate_fir_correction(
                &input_plus_iir,
                &room_config.optimizer,
                room_config.target_curve.as_ref(),
                sample_rate,
            )
            .map_err(|e| anyhow!("FIR generation failed: {}", e))?;

            // 5. Save WAV
            let filename = format!("{}_residual_fir.wav", channel_name);
            let wav_path = output_dir.join(&filename);
            autoeq::fir::save_fir_to_wav(&coeffs, sample_rate as u32, &wav_path)
                .map_err(|e| anyhow!("Failed to save FIR WAV: {}", e))?;

            info!("  Saved FIR filter to {}", wav_path.display());

            // 6. Build chain (IIR + Convolution)
            let conv_plugin = output::create_convolution_plugin(&filename);
            let mut chain =
                output::build_channel_dsp_chain(channel_name, None, Vec::new(), &eq_filters);
            chain.plugins.push(conv_plugin);

            // Returning IIR corrected curve for now, as FIR response calculation is missing
            Ok((chain, pre_score, 0.0, final_curve_iir))
        }
        _ => {
            // Default IIR mode (existing logic)
            let (eq_filters, post_score) = eq_optim::optimize_channel_eq(
                &curve,
                &room_config.optimizer,
                room_config.target_curve.as_ref(),
                sample_rate,
            )
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("EQ optimization failed for channel {}", channel_name))?;

            info!("  Optimized {} EQ filters", eq_filters.len());
            info!(
                "  Pre-score: {:.6}, Post-score: {:.6}",
                pre_score, post_score
            );

            // Build DSP chain (no gain, no crossover for simple speaker)
            let chain =
                output::build_channel_dsp_chain(channel_name, None, Vec::new(), &eq_filters);

            let final_curve = apply_filter_response(&curve, &eq_filters, sample_rate);

            Ok((chain, pre_score, post_score, final_curve))
        }
    }
}

/// Process a speaker group with multiple drivers and crossovers
///
/// Returns: (DSP chain, pre_score, post_score)
fn process_speaker_group(
    channel_name: &str,
    group: &types::SpeakerGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &std::path::Path,
) -> Result<(ChannelDspChain, f64, f64, Curve)> {
    // Load all measurements in the group
    let mut driver_curves = Vec::new();
    for (i, source) in group.measurements.iter().enumerate() {
        let curve = load::load_source(source)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| {
                format!(
                    "Failed to load driver {} measurement for channel {}",
                    i, channel_name
                )
            })?;
        driver_curves.push(curve);
    }

    debug!("  Loaded {} driver measurements", driver_curves.len());

    // Step 1: Check driver measurements for level alignment issues
    // Per the algorithm: measurements should have the average over their passband close
    let min_freq = room_config.optimizer.min_freq;
    let max_freq = room_config.optimizer.max_freq;
    let max_db = room_config.optimizer.max_db;

    // Compute peak SPL for each driver in the optimization range
    let mut peaks = Vec::with_capacity(driver_curves.len());
    for driver in &driver_curves {
        let mut peak_spl = f64::NEG_INFINITY;
        for j in 0..driver.freq.len() {
            if driver.freq[j] >= min_freq && driver.freq[j] <= max_freq && driver.spl[j] > peak_spl
            {
                peak_spl = driver.spl[j];
            }
        }
        peaks.push(peak_spl);
    }

    // Check for large level differences
    let max_peak = peaks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_peak = peaks.iter().cloned().fold(f64::INFINITY, f64::min);
    let peak_spread = max_peak - min_peak;

    for (i, peak) in peaks.iter().enumerate() {
        debug!("    Driver {}: peak {:.1} dB", i, peak);
    }
    debug!(
        "    Level spread: {:.1} dB (max gain bounds: ±{:.1} dB)",
        peak_spread, max_db
    );

    // Warn if level spread exceeds the gain bounds (meaning optimizer might hit bounds)
    if peak_spread > max_db {
        warn!(
            "Driver levels differ by {:.1} dB, which exceeds the gain bounds of ±{:.1} dB.",
            peak_spread, max_db
        );
        warn!(
            "     This may indicate measurement issues (different distances, gains, or reference levels)."
        );
        warn!(
            "     Consider increasing max_db in the config to at least ±{:.1} dB or re-measuring.",
            peak_spread / 2.0 + 3.0
        );
    }

    // Get crossover configuration
    let crossover_config = if let Some(crossover_ref) = &group.crossover {
        room_config
            .crossovers
            .as_ref()
            .and_then(|xovers| xovers.get(crossover_ref))
            .ok_or_else(|| anyhow!("Crossover configuration '{}' not found", crossover_ref))?
    } else {
        return Err(anyhow!("Speaker group requires crossover configuration"));
    };

    let crossover_type = crossover_optim::parse_crossover_type(&crossover_config.crossover_type)
        .map_err(|e| anyhow!("{}", e))?;

    // Extract fixed crossover frequencies if specified
    let fixed_freqs: Option<Vec<f64>> = if let Some(ref freqs) = crossover_config.frequencies {
        // Multiple frequencies specified (for 3-way and above)
        Some(freqs.clone())
    } else if let Some(freq) = crossover_config.frequency {
        // Single frequency specified (for 2-way)
        Some(vec![freq])
    } else {
        // No fixed frequencies - will be optimized
        None
    };

    if let Some(ref freqs) = fixed_freqs {
        info!("  Using fixed crossover frequencies: {:?} Hz", freqs);
    }

    // Compute pre-score: use initial gains (0 dB) and geometric mean crossover frequencies
    let n_drivers = driver_curves.len();
    let initial_gains = vec![0.0; n_drivers];

    // Compute geometric mean crossover frequencies as initial guess
    let mut initial_xover_freqs = Vec::new();
    for i in 0..(n_drivers - 1) {
        // Geometric mean between adjacent driver frequency ranges
        let lower_mean =
            driver_curves[i].freq.iter().sum::<f64>() / driver_curves[i].freq.len() as f64;
        let upper_mean =
            driver_curves[i + 1].freq.iter().sum::<f64>() / driver_curves[i + 1].freq.len() as f64;
        let geom_mean = (lower_mean * upper_mean).sqrt();
        initial_xover_freqs.push(geom_mean);
    }

    // Convert curves to DriverMeasurement
    let driver_measurements: Vec<autoeq::loss::DriverMeasurement> = driver_curves
        .iter()
        .map(|curve| autoeq::loss::DriverMeasurement {
            freq: curve.freq.clone(),
            spl: curve.spl.clone(),
            phase: curve.phase.clone(),
        })
        .collect();

    // Initial delays
    let initial_delays = vec![0.0; n_drivers];

    let drivers_data = autoeq::loss::DriversLossData::new(driver_measurements, crossover_type);
    let pre_score = autoeq::loss::drivers_flat_loss(
        &drivers_data,
        &initial_gains,
        &initial_xover_freqs,
        Some(&initial_delays),
        sample_rate,
        room_config.optimizer.min_freq,
        room_config.optimizer.max_freq,
    );

    // Optimize crossover
    let (gains, delays, crossover_freqs, combined_curve) = crossover_optim::optimize_crossover(
        driver_curves,
        crossover_type,
        sample_rate,
        &room_config.optimizer,
        fixed_freqs,
    )
    .map_err(|e| anyhow!("Crossover optimization failed: {}", e))?;

    info!(
        "  Optimized crossover: freqs={:?}, gains={:?}, delays={:?}",
        crossover_freqs, gains, delays
    );

    // Optimize EQ on the combined response (returns filters and post_score)
    let (eq_filters, post_score) = eq_optim::optimize_channel_eq(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| anyhow!("{}", e))
    .with_context(|| format!("EQ optimization failed for channel {}", channel_name))?;

    info!("  Optimized {} EQ filters", eq_filters.len());
    info!(
        "  Pre-score: {:.6}, Post-score: {:.6}",
        pre_score, post_score
    );

    // Build multi-driver DSP chain with per-driver crossovers
    let chain = output::build_multidriver_dsp_chain(
        channel_name,
        &gains,
        &delays,
        &crossover_freqs,
        crossover_optim::crossover_type_to_string(&crossover_type),
        &eq_filters,
    );

    let final_curve = apply_filter_response(&combined_curve, &eq_filters, sample_rate);

    Ok((chain, pre_score, post_score, final_curve))
}

fn process_multisub_group(
    channel_name: &str,
    group: &types::MultiSubGroup,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &std::path::Path,
) -> Result<(ChannelDspChain, f64, f64, Curve)> {
    // 1. Optimize multisub integration (gain + delay)
    let (result, combined_curve) =
        multisub_optim::optimize_multisub(&group.subwoofers, &room_config.optimizer, sample_rate)
            .map_err(|e| anyhow!("Multi-sub optimization failed: {}", e))?;

    info!(
        "  Multi-sub optimization: gains={:?}, delays={:?} ms",
        result.gains, result.delays
    );

    // 2. Global EQ on the combined sum
    let (eq_filters, post_score) = eq_optim::optimize_channel_eq(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| anyhow!("EQ optimization failed for multi-sub sum: {}", e))?;

    info!(
        "  Global EQ: {} filters, score={:.6}",
        eq_filters.len(),
        post_score
    );

    let chain = output::build_multisub_dsp_chain(
        channel_name,
        &group.name,
        group.subwoofers.len(),
        &result.gains,
        &result.delays,
        &eq_filters,
    );

    let final_curve = apply_filter_response(&combined_curve, &eq_filters, sample_rate);

    Ok((chain, result.pre_objective, post_score, final_curve))
}

fn process_dba(
    channel_name: &str,
    dba_config: &types::DBAConfig,
    room_config: &RoomConfig,
    sample_rate: f64,
    _output_dir: &std::path::Path,
) -> Result<(ChannelDspChain, f64, f64, Curve)> {
    // 1. Optimize DBA
    let (result, combined_curve) =
        dba_optim::optimize_dba(dba_config, &room_config.optimizer, sample_rate)
            .map_err(|e| anyhow!("DBA optimization failed: {}", e))?;

    info!(
        "  DBA Optimization: Front Gain={:.2}dB, Rear Gain={:.2}dB, Rear Delay={:.2}ms",
        result.gains[0], result.gains[1], result.delays[1]
    );

    // 2. Global EQ
    let (eq_filters, post_score) = eq_optim::optimize_channel_eq(
        &combined_curve,
        &room_config.optimizer,
        room_config.target_curve.as_ref(),
        sample_rate,
    )
    .map_err(|e| anyhow!("EQ optimization failed for DBA sum: {}", e))?;

    info!(
        "  Global EQ: {} filters, score={:.6}",
        eq_filters.len(),
        post_score
    );

    // 3. Build Chain
    let chain =
        output::build_dba_dsp_chain(channel_name, &result.gains, &result.delays, &eq_filters);

    let final_curve = apply_filter_response(&combined_curve, &eq_filters, sample_rate);

    Ok((chain, result.pre_objective, post_score, final_curve))
}

/// Compute complex response of PEQ filters
fn compute_peq_complex_response(
    filters: &[math_audio_iir_fir::Biquad],
    freq: &ndarray::Array1<f64>,
    sample_rate: f64,
) -> Vec<Complex64> {
    freq.iter()
        .map(|&f| {
            let w = 2.0 * std::f64::consts::PI * f / sample_rate;
            let z_inv = Complex64::from_polar(1.0, -w);
            let z_inv_2 = z_inv * z_inv;

            let mut total_h = Complex64::new(1.0, 0.0);

            for b in filters {
                // Calculate coefficients based on parameters
                // Using standard RBJ formulas as we cannot access private fields
                let f0 = b.freq;
                let fs = b.srate;
                let q = b.q;
                let db = b.db_gain;

                let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
                let cos_w0 = w0.cos();
                let sin_w0 = w0.sin();
                let alpha = sin_w0 / (2.0 * q);
                let big_a = 10.0_f64.powf(db / 40.0);

                // Determine filter type from debug string (hacky but functional given private fields)
                let type_name = format!("{:?}", b.filter_type);

                let (b0, b1, b2, a0, a1, a2) =
                    if type_name.contains("Peaking") || type_name.contains("Pk") {
                        let b0 = 1.0 + alpha * big_a;
                        let b1 = -2.0 * cos_w0;
                        let b2 = 1.0 - alpha * big_a;
                        let a0 = 1.0 + alpha / big_a;
                        let a1 = -2.0 * cos_w0;
                        let a2 = 1.0 - alpha / big_a;
                        (b0, b1, b2, a0, a1, a2)
                    } else if type_name.contains("LowShelf") || type_name.contains("Ls") {
                        let sqrt_a = big_a.sqrt();
                        let b0 =
                            big_a * ((big_a + 1.0) - (big_a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                        let b1 = 2.0 * big_a * ((big_a - 1.0) - (big_a + 1.0) * cos_w0);
                        let b2 =
                            big_a * ((big_a + 1.0) - (big_a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                        let a0 = (big_a + 1.0) + (big_a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                        let a1 = -2.0 * ((big_a - 1.0) + (big_a + 1.0) * cos_w0);
                        let a2 = (big_a + 1.0) + (big_a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                        (b0, b1, b2, a0, a1, a2)
                    } else if type_name.contains("HighShelf") || type_name.contains("Hs") {
                        let sqrt_a = big_a.sqrt();
                        let b0 =
                            big_a * ((big_a + 1.0) + (big_a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                        let b1 = -2.0 * big_a * ((big_a - 1.0) + (big_a + 1.0) * cos_w0);
                        let b2 =
                            big_a * ((big_a + 1.0) + (big_a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                        let a0 = (big_a + 1.0) - (big_a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                        let a1 = 2.0 * ((big_a - 1.0) - (big_a + 1.0) * cos_w0);
                        let a2 = (big_a + 1.0) - (big_a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                        (b0, b1, b2, a0, a1, a2)
                    } else {
                        // Default / Identity / Unknown
                        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    };

                let num = Complex64::new(b0, 0.0)
                    + Complex64::new(b1, 0.0) * z_inv
                    + Complex64::new(b2, 0.0) * z_inv_2;
                let den = Complex64::new(a0, 0.0)
                    + Complex64::new(a1, 0.0) * z_inv
                    + Complex64::new(a2, 0.0) * z_inv_2;

                if den.norm_sqr() > 1e-12 {
                    total_h *= num / den;
                }
            }
            total_h
        })
        .collect()
}

/// Apply filter response to a curve
fn apply_filter_response(
    curve: &Curve,
    filters: &[math_audio_iir_fir::Biquad],
    sample_rate: f64,
) -> Curve {
    let complex_resp = compute_peq_complex_response(filters, &curve.freq, sample_rate);

    let mut new_spl = ndarray::Array1::zeros(curve.freq.len());
    let mut new_phase = ndarray::Array1::zeros(curve.freq.len());
    let old_phase = curve.phase.as_ref();

    for i in 0..curve.freq.len() {
        let h = complex_resp[i];
        let h_mag_db = 20.0 * h.norm().log10();
        let h_phase_deg = h.arg().to_degrees();

        new_spl[i] = curve.spl[i] + h_mag_db;
        let p_in = old_phase.map(|p| p[i]).unwrap_or(0.0);
        new_phase[i] = p_in + h_phase_deg;
    }

    Curve {
        freq: curve.freq.clone(),
        spl: new_spl,
        phase: Some(new_phase),
    }
}
