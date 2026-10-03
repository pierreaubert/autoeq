use super::apo_profile_verifier::verify_emitted_apo_text;
use autoeq::iir;
use autoeq::optim::pareto::ParetoFilter;
use autoeq_plot::param_utils::{PeqLayout, params_per_filter};
use std::{error::Error, path::Path};
use tokio::fs;
use tokio::io::AsyncReadExt;

/// Warn threshold for the APO round-trip objective gap (absolute, in the
/// scalar objective's units). Integer-Hz serialization of typical filters
/// shifts the objective by ~1e-9..1e-6; larger gaps (low-frequency,
/// high-Q filters near the rounding grid) deserve a warning so the
/// reported evidence is never silently detached from the shipped preset.
pub(super) const APO_ROUNDTRIP_WARN_THRESHOLD: f64 = 1e-6;

/// One Pareto candidate with labeled objectives for preset export.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct ParetoExportCandidate {
    pub(super) index: usize,
    pub(super) num_filters: usize,
    pub(super) converged: bool,
    /// Objective values positional with [`ParetoExport::objective_labels`];
    /// `None` when the candidate did not compute that objective.
    pub(super) objectives: Vec<Option<f64>>,
    /// Per-measurement losses as (measurement id, loss) pairs.
    pub(super) per_measurement_losses: Vec<(String, f64)>,
}

/// Serializable Pareto export metadata written alongside the preset.
///
/// Additive contract for Pareto runners (current CLI flows pass `None` and
/// write no sidecar): `objective_labels` names each position of every
/// candidate's `objectives`; `selection_policy` records in words how
/// `selected_index` was chosen (e.g. "fewest filters within 0.5 dB of best
/// flatness").
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct ParetoExport {
    pub(super) objective_labels: Vec<String>,
    pub(super) selection_policy: String,
    pub(super) selected_index: usize,
    pub(super) candidates: Vec<ParetoExportCandidate>,
}

pub(super) struct ProductExportContext<'a> {
    pub(super) request: &'a autoeq::workflow::ProductRequest,
    pub(super) prepared: &'a autoeq::workflow::PreparedProduct,
    pub(super) compatibility: &'a autoeq::workflow::TargetCompatibility,
    pub(super) source_parameters: &'a [f64],
    pub(super) effective_envelope: &'a super::runopt::EffectiveOptimizationEnvelope,
    pub(super) max_filter_transfer_delta_db: f64,
    pub(super) verification_frequencies_hz: &'a [f64],
}

/// Allow only floating-point round-trip noise when `log10` decodes a
/// serialized integer-Hz center back into the optimizer's logarithmic axis.
/// This is many orders below the APO frequency quantization step itself.
const FREQUENCY_BOUND_INVERSE_EPS: f64 = 16.0 * f64::EPSILON;

fn parameter_layout_names(model: autoeq::PeqModel) -> &'static [&'static str] {
    match model {
        autoeq::PeqModel::Free | autoeq::PeqModel::FreePkFree => {
            &["type_code", "log10_frequency_hz", "q", "gain_db"]
        }
        _ => &["log10_frequency_hz", "q", "gain_db"],
    }
}

fn validate_parameter_bounds(
    values: &[f64],
    lower: &[f64],
    upper: &[f64],
    model: autoeq::PeqModel,
    allow_frequency_inverse_eps: bool,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    let params_per_filter = params_per_filter(model);
    if values.is_empty()
        || values.len() != lower.len()
        || values.len() != upper.len()
        || !values.len().is_multiple_of(params_per_filter)
    {
        return Err(format!(
            "{label} parameter layout does not match the retained optimizer bounds: values={}, lower={}, upper={}, parameters_per_filter={params_per_filter}",
            values.len(),
            lower.len(),
            upper.len()
        )
        .into());
    }
    let frequency_index = model.layout().freq_idx;
    for (index, ((&value, &minimum), &maximum)) in values.iter().zip(lower).zip(upper).enumerate() {
        if !value.is_finite() || !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(format!(
                "{label} parameter {index} or its retained optimizer interval is non-finite or unordered"
            )
            .into());
        }
        let frequency_axis = index % params_per_filter == frequency_index;
        let allowance = if allow_frequency_inverse_eps && frequency_axis {
            FREQUENCY_BOUND_INVERSE_EPS * minimum.abs().max(maximum.abs()).max(value.abs()).max(1.0)
        } else {
            0.0
        };
        if value < minimum && minimum - value > allowance {
            return Err(format!(
                "{label} parameter {index}={value} falls below the retained optimizer bound {minimum}"
            )
            .into());
        }
        if value > maximum && value - maximum > allowance {
            return Err(format!(
                "{label} parameter {index}={value} exceeds the retained optimizer bound {maximum}"
            )
            .into());
        }
    }
    Ok(())
}

fn same_filter_values(left: &autoeq::iir::Biquad, right: &autoeq::iir::Biquad) -> bool {
    left.filter_type == right.filter_type
        && left.freq == right.freq
        && left.q == right.q
        && left.db_gain == right.db_gain
}

fn validate_candidate_objective_constraints(
    candidate: &[f64],
    envelope: &super::runopt::EffectiveOptimizationEnvelope,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if let Some(knots) = envelope.max_boost_envelope.as_deref() {
        autoeq::optim::validate_envelope_knots(knots, "objective_max_boost", false)
            .map_err(std::io::Error::other)?;
    }
    if let Some(knots) = envelope.min_cut_envelope.as_deref() {
        autoeq::optim::validate_envelope_knots(knots, "objective_min_cut", false)
            .map_err(std::io::Error::other)?;
    }
    let (gain_projected, gain_adjustments) = autoeq::optim::project_gains_onto_envelopes(
        candidate,
        envelope.peq_model,
        envelope.loss_type,
        envelope.max_boost_envelope.as_deref(),
        envelope.min_cut_envelope.as_deref(),
    );
    if !gain_adjustments.is_empty() || gain_projected != candidate {
        return Err(format!("{label} would require per-filter boost/cut envelope repair").into());
    }
    let (q_projected, q_adjustments) = autoeq::optim::enforce_local_q_at_centers(
        &gain_projected,
        envelope.peq_model,
        envelope.loss_type,
        envelope.constraints.global_max_q,
        envelope.constraints.local_q_knots.as_deref(),
    )
    .map_err(std::io::Error::other)?;
    if !q_adjustments.is_empty() || q_projected != candidate {
        return Err(format!("{label} would require global/local Q constraint repair").into());
    }
    if envelope.max_boost_envelope.is_some() || envelope.min_cut_envelope.is_some() {
        let breaches = autoeq::optim::check_composite_gain_envelope(
            &q_projected,
            &envelope.composite_frequencies_hz,
            envelope.sample_rate_hz,
            envelope.peq_model,
            envelope.loss_type,
            envelope.composite_band_hz[0],
            envelope.composite_band_hz[1],
            envelope.max_boost_envelope.as_deref(),
            envelope.min_cut_envelope.as_deref(),
            envelope.constraints.subdivisions_per_bin,
        )
        .map_err(std::io::Error::other)?;
        if !breaches.is_empty() {
            let first = &breaches[0];
            return Err(format!(
                "{label} breaches the retained composite gain envelope at {:.3} Hz ({:.6} dB vs {:.6} dB)",
                first.frequency_hz, first.observed_db, first.bound_db
            )
            .into());
        }
    }
    Ok(())
}

fn validate_profiled_output_envelope(
    args: &autoeq::cli::Args,
    loss_type: &autoeq::LossType,
    profile: &autoeq::workflow::DeviceProfile,
    source_parameters: &[f64],
    realized_filters: &[autoeq::iir::Biquad],
    envelope: &super::runopt::EffectiveOptimizationEnvelope,
) -> Result<(), Box<dyn Error>> {
    if envelope.peq_model != args.effective_peq_model()
        || envelope.loss_type != *loss_type
        || envelope.sample_rate_hz != args.sample_rate
    {
        return Err(
            "profiled APO output does not match the optimizer model, loss, or sample rate".into(),
        );
    }
    if !envelope.objective_max_db.is_finite() || !envelope.objective_min_db.is_finite() {
        return Err("retained objective gain bounds must be finite".into());
    }
    envelope
        .constraints
        .as_spec()
        .validate()
        .map_err(std::io::Error::other)?;
    validate_parameter_bounds(
        source_parameters,
        &envelope.lower_bounds,
        &envelope.upper_bounds,
        envelope.peq_model,
        false,
        "optimizer source",
    )?;
    if source_parameters.len()
        != envelope
            .num_filters
            .checked_mul(params_per_filter(envelope.peq_model))
            .ok_or_else(|| std::io::Error::other("optimizer parameter count overflowed"))?
        || realized_filters.len() != envelope.num_filters
    {
        return Err(format!(
            "profiled APO filter count/layout does not match the optimizer run (source params={}, filters={}, expected filters={})",
            source_parameters.len(),
            realized_filters.len(),
            envelope.num_filters
        )
        .into());
    }

    let source_filters = autoeq::x2peq::x2peq(
        source_parameters,
        envelope.sample_rate_hz,
        envelope.peq_model,
    );
    if source_filters.len() != realized_filters.len() {
        return Err("source model produced a different filter count than the APO output".into());
    }
    for (index, ((_, source), realized)) in source_filters.iter().zip(realized_filters).enumerate()
    {
        if source.filter_type != realized.filter_type {
            return Err(format!(
                "profiled APO filter {index} type/order does not match the source PEQ model"
            )
            .into());
        }
    }

    // The serializer is applied to both the optimization result and the
    // candidate handed to the saver. Compare every index, not only the filter
    // count or total response, so a reordered same-type pair cannot pass.
    let source_biquads = source_filters
        .iter()
        .map(|(_, filter)| filter.clone())
        .collect::<Vec<_>>();
    let canonical_realized = profile
        .apo_serialized_filters(envelope.sample_rate_hz, &source_biquads)
        .map_err(std::io::Error::other)?;
    if canonical_realized.len() != realized_filters.len()
        || canonical_realized
            .iter()
            .zip(realized_filters)
            .any(|(expected, actual)| !same_filter_values(expected, actual))
    {
        return Err(
            "profiled APO filter values or order differ from the source optimizer candidate".into(),
        );
    }

    let realized_peq = realized_filters
        .iter()
        .cloned()
        .map(|filter| (1.0, filter))
        .collect::<Vec<_>>();
    let mut realized_parameters = autoeq::x2peq::peq2x(&realized_peq, envelope.peq_model);
    let ppf = params_per_filter(envelope.peq_model);
    let q_index = envelope.peq_model.layout().q_idx;
    for (index, (_, source_filter)) in source_filters.iter().enumerate() {
        if matches!(
            source_filter.filter_type,
            autoeq::iir::BiquadFilterType::Lowshelf | autoeq::iir::BiquadFilterType::Highshelf
        ) {
            // APO represents these shelves with fixed S=1 and no Q token.
            // Retain the source-only Q solely for checking the optimizer box;
            // the realized artifact records Q as absent and slope as 12 dB/oct.
            realized_parameters[index * ppf + q_index] = source_parameters[index * ppf + q_index];
        }
    }
    validate_parameter_bounds(
        &realized_parameters,
        &envelope.lower_bounds,
        &envelope.upper_bounds,
        envelope.peq_model,
        true,
        "serialized APO",
    )?;
    validate_candidate_objective_constraints(
        source_parameters,
        envelope,
        "optimizer source candidate",
    )?;
    validate_candidate_objective_constraints(
        &realized_parameters,
        envelope,
        "serialized APO candidate",
    )?;
    Ok(())
}

async fn read_existing_product_file_bounded(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let file = match fs::File::open(path).await {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut bytes = Vec::new();
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(format!(
            "existing product output {} exceeds the {maximum_bytes}-byte rollback safety limit",
            path.display()
        )
        .into());
    }
    Ok(Some(bytes))
}

async fn rollback_profiled_preset_if_unchanged(
    preset_path: &Path,
    previous_preset: Option<&[u8]>,
    published_preset: &[u8],
) -> Result<(), Box<dyn Error>> {
    let current_preset = read_existing_product_file_bounded(preset_path, 16 * 1024 * 1024).await?;
    if current_preset.as_deref() != Some(published_preset) {
        return Err(
            "APO preset changed concurrently; rollback left the newer file untouched".into(),
        );
    }
    match previous_preset {
        Some(previous) => autoeq_artifacts::write_file_atomically(preset_path, previous)?,
        None => fs::remove_file(preset_path).await?,
    }
    Ok(())
}

async fn write_staged_file(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    fs::write(path, contents).await?;
    fs::File::open(path).await?.sync_all().await?;
    Ok(())
}

/// Stage and validate both outputs before publishing either one. The preset
/// is atomically replaced first; if sidecar publication then fails, restore the
/// previous preset only when the destination still contains our new bytes.
#[cfg(test)]
async fn publish_profiled_pair_with_hook<F>(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    before_sidecar_publish: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce() -> std::io::Result<()>,
{
    publish_profiled_pair_with_hooks(
        preset_path,
        preset_bytes,
        sidecar_path,
        sidecar_bytes,
        |_| Ok(()),
        before_sidecar_publish,
    )
    .await
}

async fn publish_profiled_pair_with_hooks<V, F>(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    verify_staged_preset: V,
    before_sidecar_publish: F,
) -> Result<(), Box<dyn Error>>
where
    V: FnOnce(&[u8]) -> std::io::Result<()>,
    F: FnOnce() -> std::io::Result<()>,
{
    let parent = preset_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let sidecar_parent = sidecar_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent != sidecar_parent {
        return Err("profiled APO preset and sidecar must share a directory".into());
    }

    let staging = autoeq_artifacts::ArtifactBundleStaging::new(parent)?;
    let staged_preset = staging.member_path(Path::new("preset.txt"))?;
    let staged_sidecar = staging.member_path(Path::new("provenance.json"))?;
    write_staged_file(&staged_preset, preset_bytes).await?;
    write_staged_file(&staged_sidecar, sidecar_bytes).await?;
    let staged_preset_bytes = fs::read(&staged_preset).await?;
    let staged_sidecar_bytes = fs::read(&staged_sidecar).await?;
    verify_staged_preset(&staged_preset_bytes)?;
    autoeq::workflow::verify_apo_preset_binding(&staged_preset_bytes, &staged_sidecar_bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    // Read the existing preset with a hard byte limit before publishing so a
    // failed second rename can restore the prior file without an unbounded read.
    let previous_preset = read_existing_product_file_bounded(preset_path, 16 * 1024 * 1024).await?;
    autoeq_artifacts::write_file_atomically(preset_path, &staged_preset_bytes)?;

    let sidecar_publish = before_sidecar_publish().and_then(|()| {
        autoeq_artifacts::write_file_atomically(sidecar_path, &staged_sidecar_bytes)
    });
    if let Err(publish_error) = sidecar_publish {
        if let Err(rollback_error) = rollback_profiled_preset_if_unchanged(
            preset_path,
            previous_preset.as_deref(),
            &staged_preset_bytes,
        )
        .await
        {
            return Err(format!(
                "failed to publish product provenance sidecar ({publish_error}); failed to restore the prior APO preset ({rollback_error})"
            )
            .into());
        }
        return Err(format!(
            "failed to publish product provenance sidecar; prior APO preset was restored: {publish_error}"
        )
        .into());
    }
    Ok(())
}

async fn publish_profiled_pair(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    verify_staged_preset: impl FnOnce(&[u8]) -> std::io::Result<()>,
) -> Result<(), Box<dyn Error>> {
    publish_profiled_pair_with_hooks(
        preset_path,
        preset_bytes,
        sidecar_path,
        sidecar_bytes,
        verify_staged_preset,
        || Ok(()),
    )
    .await
}

#[cfg(test)]
pub(super) async fn publish_profiled_pair_with_test_hook<F>(
    preset_path: &Path,
    preset_bytes: &[u8],
    sidecar_path: &Path,
    sidecar_bytes: &[u8],
    before_sidecar_publish: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce() -> std::io::Result<()>,
{
    publish_profiled_pair_with_hook(
        preset_path,
        preset_bytes,
        sidecar_path,
        sidecar_bytes,
        before_sidecar_publish,
    )
    .await
}

impl ParetoExport {
    /// Build export metadata from [`ParetoFilter`] results.
    ///
    /// Recognised labels are `flatness_loss` and `score_loss`; any other
    /// label maps to `None` for every candidate. `per_measurement_losses`
    /// must be positional with `filters` (short inputs are padded with
    /// empty entries, long inputs truncated).
    ///
    /// This is the entry point for Pareto runners (no such runner exists
    /// in the CLI yet, hence the allowance); covered by unit tests.
    #[allow(dead_code)]
    pub(super) fn from_pareto_filters(
        filters: &[ParetoFilter],
        objective_labels: Vec<String>,
        selection_policy: impl Into<String>,
        selected_index: usize,
        per_measurement_losses: Vec<Vec<(String, f64)>>,
    ) -> Self {
        let mut padded = per_measurement_losses;
        padded.resize_with(filters.len(), Vec::new);
        padded.truncate(filters.len());
        let candidates = filters
            .iter()
            .zip(padded)
            .enumerate()
            .map(|(index, (filter, per_measurement_losses))| {
                let objectives = objective_labels
                    .iter()
                    .map(|label| match label.as_str() {
                        "flatness_loss" => Some(filter.flatness_loss),
                        "score_loss" => filter.score_loss,
                        _ => None,
                    })
                    .collect();
                ParetoExportCandidate {
                    index,
                    num_filters: filter.num_filters,
                    converged: filter.converged,
                    objectives,
                    per_measurement_losses,
                }
            })
            .collect();
        Self {
            objective_labels,
            selection_policy: selection_policy.into(),
            selected_index,
            candidates,
        }
    }
}

/// Re-evaluate the scalar objective on the integer-Hz APO serialization of
/// `x` and return the absolute gap vs the optimizer response.
///
/// Mirrors the rounding in [`save_peq_to_file`] exactly (`x2peq` →
/// round frequencies → `peq2x`), so the reported optimization evidence
/// stays aligned with the shipped preset — most visible for
/// low-frequency/high-Q filters where a sub-Hz rounding step moves the
/// response most. Returns `None` when either evaluation is non-finite.
pub(super) fn apo_roundtrip_objective_gap(
    x: &[f64],
    sample_rate: f64,
    peq_model: autoeq::PeqModel,
    objective_data: &autoeq::optim::ObjectiveData,
) -> Option<f64> {
    let before = autoeq::optim::compute_fitness_penalties_ref(x, objective_data);
    if !before.is_finite() {
        return None;
    }
    let mut apo_peq = autoeq::x2peq::x2peq(x, sample_rate, peq_model);
    for (_, filter) in &mut apo_peq {
        filter.freq = filter.freq.round();
    }
    let reconstructed = autoeq::x2peq::peq2x(&apo_peq, peq_model);
    let after = autoeq::optim::compute_fitness_penalties_ref(&reconstructed, objective_data);
    if !after.is_finite() {
        return None;
    }
    Some((after - before).abs())
}

/// Save PEQ settings to APO format file
///
/// # Arguments
/// * `args` - Command line arguments
/// * `x` - Optimized filter parameters
/// * `output_path` - Base output path for files
/// * `loss_type` - Type of optimization performed
/// * `pareto` - Optional Pareto export metadata; when `Some`, objective
///   labels, the selected-point policy, per-candidate objectives and
///   per-measurement losses are written as JSON next to the preset
///
/// # Returns
/// * Result indicating success or error
pub(super) async fn save_peq_to_file(
    args: &autoeq::cli::Args,
    x: &[f64],
    output_path: &Path,
    loss_type: &autoeq::LossType,
    pareto: Option<&ParetoExport>,
) -> Result<(), Box<dyn Error>> {
    // Build the PEQ from the optimized parameters
    let peq_model = args.effective_peq_model();
    let peq = autoeq::x2peq::x2peq(x, args.sample_rate, peq_model);

    // Determine filename based on loss type
    let filename = match loss_type {
        autoeq::LossType::SpeakerFlat
        | autoeq::LossType::SpeakerFlatAsymmetric
        | autoeq::LossType::HeadphoneFlat => "iir-autoeq-flat.txt",
        autoeq::LossType::SpeakerScore
        | autoeq::LossType::HeadphoneScore
        | autoeq::LossType::Epa => "iir-autoeq-score.txt",
        autoeq::LossType::DriversFlat | autoeq::LossType::MultiSubFlat => {
            // Unreachable: DriversFlat mode uses a separate code path
            unreachable!("DriversFlat mode should not reach this point");
        }
    };

    // Create the full path (same directory as the plots)
    let parent_dir = output_path.parent().unwrap_or(output_path);
    let file_path = parent_dir.join(filename);

    // Generate comment string with optimization details
    let comment = format!(
        "# AutoEQ Parametric Equalizer Settings\n# Speaker: {}\n# Loss Type: {:?}\n# Filters: {}\n# Generated: {}",
        args.speaker.as_deref().unwrap_or("Unknown"),
        loss_type,
        args.num_filters,
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    );

    // Equalizer APO export uses integer-Hz center frequencies. The upstream
    // formatter casts frequencies to i32, so round first to avoid turning a
    // log-space round-trip such as 499.999... Hz into 499 Hz.
    let mut apo_peq = peq.clone();
    for (_, filter) in &mut apo_peq {
        filter.freq = filter.freq.round();
    }
    let apo_content = iir::peq_format_apo(&comment, &apo_peq);

    // Ensure parent directory exists
    if let Some(parent) = file_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    // Write the APO file
    fs::write(&file_path, apo_content).await?;
    crate::qa_println!(args, "🕶 PEQ settings saved to: {}", file_path.display());

    // Pareto runs additionally export objective labels, the
    // selected-point policy and per-measurement losses with the preset.
    if let Some(export) = pareto {
        let sidecar_name = format!(
            "{}-pareto.json",
            file_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "iir-autoeq".to_string())
        );
        let sidecar_path = parent_dir.join(&sidecar_name);
        let sidecar_content = serde_json::to_string_pretty(export)?;
        fs::write(&sidecar_path, sidecar_content).await?;
        crate::qa_println!(
            args,
            "📊 Pareto export saved to: {}",
            sidecar_path.display()
        );
    }

    // Save RME TotalMix format (.xml)
    let rme_filename = filename.replace(".txt", ".tmreq");
    let rme_path = parent_dir.join(&rme_filename);
    let rme_content = iir::peq_format_rme_room(&peq, &peq);
    fs::write(&rme_path, rme_content).await?;
    crate::qa_println!(
        args,
        "🎚  RME TotalMix RoomEQ preset saved to: {}",
        rme_path.display()
    );

    // Save Apple AUNBandEQ format (.aupreset)
    let aupreset_filename = filename.replace(".txt", ".aupreset");
    let aupreset_path = parent_dir.join(&aupreset_filename);
    let preset_name = format!("AutoEQ {}", args.speaker.as_deref().unwrap_or("Unknown"));
    let aupreset_content = iir::peq_format_aupreset(&peq, &preset_name);
    fs::write(&aupreset_path, aupreset_content).await?;
    crate::qa_println!(
        args,
        "🍎 Apple AUpreset saved to: {}",
        aupreset_path.display()
    );

    Ok(())
}

/// Save only the APO format whose current serialization behavior is checked
/// against an explicit product device profile. RME and AU remain available in
/// the legacy path but do not yet have equivalent profile validation.
pub(super) async fn save_profiled_apo_to_file(
    args: &autoeq::cli::Args,
    filters: &[autoeq::iir::Biquad],
    preamp_db: f64,
    output_path: &Path,
    loss_type: &autoeq::LossType,
    context: ProductExportContext<'_>,
) -> Result<(), Box<dyn Error>> {
    if !preamp_db.is_finite() {
        return Err("profiled APO preamp must be finite".into());
    }
    if preamp_db > 0.0 {
        return Err("profiled APO output supports non-positive preamp values only".into());
    }
    if !context.max_filter_transfer_delta_db.is_finite()
        || context.max_filter_transfer_delta_db < 0.0
    {
        return Err("profiled APO transfer delta must be finite and non-negative".into());
    }
    let profile = &context.request.device_profile;
    profile
        .validate_filters(args.sample_rate, filters)
        .map_err(std::io::Error::other)?;
    let expected_filters = profile
        .apo_serialized_filters(args.sample_rate, filters)
        .map_err(std::io::Error::other)?;
    validate_profiled_output_envelope(
        args,
        loss_type,
        profile,
        context.source_parameters,
        &expected_filters,
        context.effective_envelope,
    )?;
    let expected_preamp_db = profile
        .apo_serialized_preamp_db()
        .map_err(std::io::Error::other)?;
    if preamp_db != expected_preamp_db {
        return Err("profiled APO preamp does not match the serialized device profile".into());
    }
    let filename = match loss_type {
        autoeq::LossType::SpeakerFlat
        | autoeq::LossType::SpeakerFlatAsymmetric
        | autoeq::LossType::HeadphoneFlat => "iir-autoeq-flat.txt",
        autoeq::LossType::SpeakerScore
        | autoeq::LossType::HeadphoneScore
        | autoeq::LossType::Epa => "iir-autoeq-score.txt",
        autoeq::LossType::DriversFlat | autoeq::LossType::MultiSubFlat => {
            return Err("profiled APO export does not support multi-driver loss modes".into());
        }
    };
    let parent_dir = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_path = parent_dir.join(filename);
    let peq: Vec<(f64, autoeq::iir::Biquad)> = filters
        .iter()
        .cloned()
        .map(|filter| (1.0, filter))
        .collect();
    let comment = format!(
        "# AutoEQ Product Profile\n# Device profile: {}\n# Source: {}\n# Target: {}",
        context.request.device_profile.id,
        context.prepared.source_record.id,
        context.prepared.target_profile.record.id,
    );
    let serialized = autoeq::iir::peq_format_apo(&comment, &peq);
    let mut replaced_preamp = false;
    let mut lines = Vec::new();
    for line in serialized.lines() {
        if line.starts_with("Preamp:") {
            lines.push(format!("Preamp: {preamp_db:.1} dB"));
            replaced_preamp = true;
        } else if let Some(shelf_line) = profiled_apo_shelf_line(line)? {
            lines.push(shelf_line);
        } else {
            lines.push(line.to_owned());
        }
    }
    if !replaced_preamp {
        return Err("APO serializer did not emit a preamp line".into());
    }
    let preset_bytes = format!("{}\n", lines.join("\n")).into_bytes();
    let emitted_text = verify_emitted_apo_text(
        &preset_bytes,
        args.sample_rate,
        &expected_filters,
        expected_preamp_db,
        context.verification_frequencies_hz,
    )
    .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    let manifest_path = file_path.with_extension("product-provenance.json");
    let shelf_contract = autoeq::workflow::profiled_apo_shelf_contract();
    let mut composite_frequency_bytes = Vec::with_capacity(
        context.effective_envelope.composite_frequencies_hz.len() * std::mem::size_of::<u64>(),
    );
    for frequency_hz in &context.effective_envelope.composite_frequencies_hz {
        composite_frequency_bytes.extend_from_slice(&frequency_hz.to_bits().to_le_bytes());
    }
    let global_max_q = context.effective_envelope.constraints.global_max_q;
    let manifest = serde_json::json!({
        "schema": "autoeq.product-run-provenance",
        "schema_version": 4,
        "request": context.request,
        "source_record": &context.prepared.source_record,
        "target_record": &context.prepared.target_profile.record,
        "target_compatibility": context.compatibility,
        "device_profile": &context.request.device_profile,
        "effective_optimizer_envelope": {
            "model": context.effective_envelope.peq_model.to_string(),
            "filter_count": context.effective_envelope.num_filters,
            "parameter_names_per_filter": parameter_layout_names(context.effective_envelope.peq_model),
            "sample_rate_hz": context.effective_envelope.sample_rate_hz,
            "source_candidate_parameters": context.source_parameters,
            "lower_bounds": context.effective_envelope.lower_bounds,
            "upper_bounds": context.effective_envelope.upper_bounds,
            "bounds_units": "log10_frequency_hz, q, gain_db; free layouts prepend type_code",
            "global_max_q": if global_max_q.is_finite() { Some(global_max_q) } else { None },
            "global_max_q_unbounded": !global_max_q.is_finite(),
            "local_q_knots": context.effective_envelope.constraints.local_q_knots,
            "subdivisions_per_bin": context.effective_envelope.constraints.subdivisions_per_bin,
            "objective_gain_envelopes": {
                "max_boost_db": context.effective_envelope.max_boost_envelope,
                "min_cut_db": context.effective_envelope.min_cut_envelope,
                "interpolation": "linear_in_log_frequency_with_endpoint_hold",
                "objective_min_db": context.effective_envelope.objective_min_db,
                "objective_max_db": context.effective_envelope.objective_max_db,
                "composite_band_hz": context.effective_envelope.composite_band_hz,
                "composite_grid_point_count": context.effective_envelope.composite_frequencies_hz.len(),
                "composite_grid_sha256_le_f64": autoeq_artifacts::sha256_hex(&composite_frequency_bytes)
            },
            "source_and_serialized_candidates_checked": true,
            "frequency_inverse_transform_allowance": FREQUENCY_BOUND_INVERSE_EPS,
            "frequency_inverse_transform_allowance_units": "scaled_machine_epsilon_in_log10_frequency_axis",
            "shelf_q_semantics": "optimizer_source_q_is_checked_against_its_box; APO text emits no shelf Q and uses the verified 12 dB_per_octave S1 mapping"
        },
        "apo_serialization": {
            "sample_rate_hz": args.sample_rate,
            "frequency_quantization": "nearest_integer_hz",
            "q_decimal_places": 2,
            "gain_decimal_places": 2,
            "preamp_decimal_places": 1,
            "realized_preamp_db": emitted_text.preamp_db,
            "preset_sha256": autoeq_artifacts::sha256_hex(&preset_bytes),
            "max_filter_transfer_delta_db": context.max_filter_transfer_delta_db,
            "verified_output": "Equalizer APO text preset in the checked profiled subset",
            "emitted_text_verification": {
                "method": "strict_local_subset_parse_transfer_and_source_shelf_check_v2",
                "sample_rate_hz": emitted_text.sample_rate_hz,
                "max_transfer_delta_db": emitted_text.max_transfer_delta_db,
                "source_shelf_contract": shelf_contract,
                "max_shelf_scaled_coefficient_delta": emitted_text.max_shelf_scaled_coefficient_delta,
                "max_shelf_source_transfer_delta_db": emitted_text.max_shelf_source_transfer_delta_db,
                "channel_scope": "inherited_from_including_equalizer_apo_configuration",
                "playback_device_binding": "not_encoded_or_verified",
                "runtime_installation_checked": false,
                "consumer_parser_used": false
            }
        },
        "realized_filters": emitted_text.emitted_filters.iter().map(|filter| serde_json::json!({
            "type": filter.kind,
            "frequency_hz": filter.frequency_hz,
            "q": filter.q,
            "gain_db": filter.gain_db,
            "slope_db_per_octave": filter.slope_db_per_octave,
            "frequency_convention": filter.frequency_convention,
        })).collect::<Vec<_>>()
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    autoeq::workflow::verify_apo_preset_binding(&preset_bytes, &manifest_bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    publish_profiled_pair(
        &file_path,
        &preset_bytes,
        &manifest_path,
        &manifest_bytes,
        |staged_preset| {
            verify_emitted_apo_text(
                staged_preset,
                args.sample_rate,
                &expected_filters,
                expected_preamp_db,
                context.verification_frequencies_hz,
            )
            .map(|_| ())
            .map_err(std::io::Error::other)
        },
    )
    .await?;
    crate::qa_println!(
        args,
        "🕶 Profiled APO preset saved to: {}",
        file_path.display()
    );
    crate::qa_println!(
        args,
        "📋 Product provenance saved to: {}",
        manifest_path.display()
    );
    Ok(())
}

fn profiled_apo_shelf_line(line: &str) -> Result<Option<String>, std::io::Error> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 4 || fields[0] != "Filter" || fields[2] != "ON" {
        return Ok(None);
    }
    let token = match fields[3] {
        "LS" => "LSC",
        "HS" => "HSC",
        _ => return Ok(None),
    };
    let valid_index = fields[1]
        .strip_suffix(':')
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|value| value > 0);
    let valid_frequency = fields
        .get(5)
        .is_some_and(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()));
    if fields.len() != 12
        || !valid_index
        || fields[4] != "Fc"
        || fields[6] != "Hz"
        || fields[7] != "Gain"
        || fields[9] != "dB"
        || fields[10] != "Q"
        || !valid_frequency
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "legacy APO formatter emitted an unexpected shelf line",
        ));
    }
    Ok(Some(format!(
        "Filter {}: ON {token} 12 dB Fc {} Hz Gain {} dB",
        fields[1].trim_end_matches(':'),
        fields[5],
        fields[8]
    )))
}
