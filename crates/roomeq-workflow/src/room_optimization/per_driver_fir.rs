//! Artifact ownership for jointly designed physical-output FIRs.
use roomeq_model::{AutoeqError, ChannelDspChain, Curve, Result, RoomConfig};
use std::path::Path;

fn invalid(message: impl Into<String>) -> AutoeqError {
    AutoeqError::InvalidConfiguration {
        message: message.into(),
    }
}

pub(super) fn generate(
    chain: &mut ChannelDspChain,
    config: &RoomConfig,
    fs: f64,
    dir: Option<&Path>,
) -> Result<Curve> {
    let dir = dir.ok_or_else(|| {
        invalid("per_driver FIR placement requires an output directory for physical FIR sidecars")
    })?;
    let target: Curve = if let Some(target) = chain.target_curve.clone() {
        target.into()
    } else {
        // Full-band group EQ has no untouched upper reference. Use the same
        // existing full-band FIR target convention once, for the group only.
        let initial: Curve = chain
            .initial_curve
            .clone()
            .ok_or_else(|| invalid("per_driver FIR requires a calibrated group capture"))?
            .into();
        crate::fir::resolve_fir_target_curve(
            &initial,
            &config.optimizer,
            config.target_curve.as_ref(),
        )
        .map_err(|e| invalid(e.to_string()))?
    };
    let drivers = chain
        .drivers
        .as_ref()
        .ok_or_else(|| invalid("missing physical drivers"))?;
    let mut measurements = Vec::new();
    let mut branches = Vec::new();
    let mut optimizer = config.optimizer.clone();
    for driver in drivers {
        let raw: Curve = driver
            .initial_curve
            .clone()
            .ok_or_else(|| invalid(format!("missing calibrated capture for '{}'", driver.name)))?
            .into();
        raw.validate("per-driver capture")
            .map_err(|e| invalid(e.to_string()))?;
        if raw.phase.is_none() {
            return Err(invalid(
                "per_driver group FIR requires phase-referenced captures for every physical speaker",
            ));
        }
        optimizer.min_freq = optimizer.min_freq.max(raw.freq[0]);
        optimizer.max_freq = optimizer.max_freq.min(raw.freq[raw.freq.len() - 1]);
        // Interpolate BEFORE applying crossover, delay or FIR, preserving SPL
        // and unwrapping phase. Never normalize individual physical captures.
        let raw = autoeq_core::interpolate_log_space(&target.freq, &raw);
        let mut branch = chain.clone();
        branch.drivers = None;
        branch.plugins = driver.plugins.clone();
        branch.plugins.extend(chain.plugins.clone());
        branches.push(
            crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(&branch, &raw, fs, dir)?,
        );
        measurements.push(raw);
    }
    let designed = roomeq_engine::fir::generate_per_driver_firs(
        &branches,
        &measurements,
        &target,
        &optimizer,
        fs,
    )
    .map_err(invalid)?;
    // Stage all artifact writes before mutating the deployed chain.
    let mut plugins = Vec::new();
    for (driver, taps) in drivers.iter().zip(&designed.coefficients) {
        let name = format!("{}_driver_{}_{}", chain.channel, driver.index, driver.name);
        let (filename, path) = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
            dir,
            &name,
            autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
            fs,
        );
        math_audio_iir_fir::save_fir_to_wav(taps, fs as u32, &path).map_err(|e| {
            invalid(format!(
                "failed to write physical FIR '{}': {e}",
                path.display()
            ))
        })?;
        let mut plugin = roomeq_engine::output::create_convolution_plugin(&filename);
        plugin.parameters["room_eq_fir_placement"] = serde_json::json!("per_driver");
        plugin.parameters["latency_samples"] = serde_json::json!(designed.latency_samples);
        plugin.parameters["correction_design_delay_ms"] =
            serde_json::json!(designed.latency_samples as f64 * 1000.0 / fs);
        plugin.parameters["fir_taps"] = serde_json::json!(taps.len());
        plugin.parameters["phase_mode"] = serde_json::json!(config.optimizer.processing_mode);
        plugin.parameters["protected_null_bins"] = serde_json::json!(designed.protected_bins);
        plugin.parameters["joint_target_rms_before_db"] = serde_json::json!(designed.before_rms_db);
        plugin.parameters["joint_target_rms_after_db"] = serde_json::json!(designed.after_rms_db);
        plugins.push(plugin);
    }
    for (driver, plugin) in chain.drivers.as_mut().unwrap().iter_mut().zip(plugins) {
        driver.plugins.push(plugin);
    }
    chain.final_curve = Some((&designed.final_curve).into());
    chain.target_curve = Some((&target).into());
    Ok(designed.final_curve)
}

pub(super) fn has_physical_fir(chain: &ChannelDspChain) -> bool {
    chain.drivers.as_ref().is_some_and(|drivers| {
        drivers.iter().any(|d| {
            d.plugins.iter().any(|p| {
                p.parameters
                    .get("room_eq_fir_placement")
                    .and_then(|v| v.as_str())
                    == Some("per_driver")
            })
        })
    })
}

/// Replay the physical captures rather than multiplying their acoustic sum by
/// an electrical sum of filters. Those operations are not interchangeable.
pub(super) fn replay(chain: &ChannelDspChain, grid: &Curve, fs: f64, dir: &Path) -> Result<Curve> {
    use num_complex::Complex64;
    let mut sum = vec![Complex64::new(0.0, 0.0); grid.freq.len()];
    for driver in chain.drivers.as_deref().unwrap_or_default() {
        let raw: Curve = driver
            .initial_curve
            .clone()
            .ok_or_else(|| invalid("missing physical capture"))?
            .into();
        let raw = autoeq_core::interpolate_log_space(&grid.freq, &raw);
        let mut branch = chain.clone();
        branch.drivers = None;
        branch.plugins = driver.plugins.clone();
        branch.plugins.extend(chain.plugins.clone());
        let realized =
            crate::ctc::apply_channel_dsp_chain_to_curve_with_sidecar_dir(&branch, &raw, fs, dir)?;
        for (i, z) in sum.iter_mut().enumerate() {
            *z += Complex64::from_polar(
                10.0_f64.powf(realized.spl[i] / 20.0),
                realized.phase.as_ref().map_or(0.0, |p| p[i].to_radians()),
            );
        }
    }
    Ok(Curve {
        freq: grid.freq.clone(),
        spl: ndarray::Array1::from_iter(sum.iter().map(|z| 20.0 * z.norm().max(1e-15).log10())),
        phase: Some(ndarray::Array1::from_iter(
            sum.iter().map(|z| z.arg().to_degrees()),
        )),
        ..Curve::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{DriverDspChain, FirConfig, FirPlacement, ProcessingMode};
    fn setup() -> (ChannelDspChain, RoomConfig, Curve) {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let mut chain = result.channels.remove("L").unwrap();
        let freq = ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20000.0_f64.log10(), 160);
        let target = Curve {
            spl: ndarray::Array1::from_elem(freq.len(), 80.0),
            phase: Some(ndarray::Array1::zeros(freq.len())),
            freq,
            ..Curve::default()
        };
        let mut raw = target.clone();
        raw.spl -= 20.0 * 2.0_f64.log10();
        chain.plugins.clear();
        chain.target_curve = Some((&target).into());
        chain.drivers = Some(
            (0..2)
                .map(|index| DriverDspChain {
                    name: format!("speaker/{index}"),
                    index,
                    plugins: vec![],
                    initial_curve: Some((&raw).into()),
                })
                .collect(),
        );
        let mut config = RoomConfig::default();
        config.optimizer.processing_mode = ProcessingMode::PhaseLinear;
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 200.0;
        config.optimizer.fir = Some(FirConfig {
            placement: FirPlacement::PerDriver,
            phase: "linear".into(),
            ..Default::default()
        });
        (chain, config, target)
    }
    #[test]
    fn per_driver_sidecars_replay_the_physical_sum_and_have_unique_paths() {
        let (mut chain, config, target) = setup();
        let dir = tempfile::tempdir().unwrap();
        let rendered = generate(&mut chain, &config, 48000.0, Some(dir.path())).unwrap();
        assert!(chain.plugins.iter().all(|p| p.plugin_type != "convolution"));
        let paths: Vec<_> = chain
            .drivers
            .as_ref()
            .unwrap()
            .iter()
            .map(|driver| {
                let p = driver.plugins.last().unwrap();
                assert_eq!(p.plugin_type, "convolution");
                assert_eq!(p.parameters["latency_samples"], 2048);
                let path = p.parameters["ir_file"].as_str().unwrap().to_string();
                assert!(dir.path().join(&path).exists());
                path
            })
            .collect();
        assert_ne!(paths[0], paths[1]);
        let replayed = replay(&chain, &target, 48000.0, dir.path()).unwrap();
        for (a, b) in rendered.spl.iter().zip(&replayed.spl) {
            assert!((a - b).abs() < 1e-4);
        }
        // A physical FIR must not be silently replaced by a common filter.
        assert!(has_physical_fir(&chain));
    }
    #[test]
    fn per_driver_missing_artifact_directory_is_an_error_not_a_dangling_export() {
        let (mut chain, config, _) = setup();
        let before = serde_json::to_value(&chain).unwrap();
        assert!(generate(&mut chain, &config, 48000.0, None).is_err());
        assert_eq!(serde_json::to_value(&chain).unwrap(), before);
    }
    #[test]
    fn per_driver_missing_capture_fails_before_deploying_any_filter() {
        let (mut chain, config, _) = setup();
        chain.drivers.as_mut().unwrap()[1].initial_curve = None;
        let dir = tempfile::tempdir().unwrap();
        assert!(generate(&mut chain, &config, 48000.0, Some(dir.path())).is_err());
        assert!(chain.drivers.unwrap().iter().all(|d| d.plugins.is_empty()));
    }
}
