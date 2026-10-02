use autoeq::iir;
use autoeq::optim::pareto::ParetoFilter;
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
    pub(super) max_filter_transfer_delta_db: f64,
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
) -> Result<(), Box<dyn Error>> {
    publish_profiled_pair_with_hook(
        preset_path,
        preset_bytes,
        sidecar_path,
        sidecar_bytes,
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
    if !context.max_filter_transfer_delta_db.is_finite()
        || context.max_filter_transfer_delta_db < 0.0
    {
        return Err("profiled APO transfer delta must be finite and non-negative".into());
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
        } else {
            lines.push(line.to_owned());
        }
    }
    if !replaced_preamp {
        return Err("APO serializer did not emit a preamp line".into());
    }
    let preset_bytes = format!("{}\n", lines.join("\n")).into_bytes();

    let manifest_path = file_path.with_extension("product-provenance.json");
    let manifest = serde_json::json!({
        "schema": "autoeq.product-run-provenance",
        "schema_version": 1,
        "request": context.request,
        "source_record": &context.prepared.source_record,
        "target_record": &context.prepared.target_profile.record,
        "target_compatibility": context.compatibility,
        "device_profile": &context.request.device_profile,
        "apo_serialization": {
            "sample_rate_hz": args.sample_rate,
            "frequency_quantization": "nearest_integer_hz",
            "q_decimal_places": 2,
            "gain_decimal_places": 2,
            "preamp_decimal_places": 1,
            "realized_preamp_db": preamp_db,
            "preset_sha256": autoeq_artifacts::sha256_hex(&preset_bytes),
            "max_filter_transfer_delta_db": context.max_filter_transfer_delta_db,
            "verified_output": "Equalizer APO text preset only"
        },
        "realized_filters": filters.iter().map(|filter| serde_json::json!({
            "type": filter.filter_type.short_name(),
            "frequency_hz": filter.freq,
            "q": filter.q,
            "gain_db": filter.db_gain
        })).collect::<Vec<_>>()
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    autoeq::workflow::verify_apo_preset_binding(&preset_bytes, &manifest_bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    publish_profiled_pair(&file_path, &preset_bytes, &manifest_path, &manifest_bytes).await?;
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
