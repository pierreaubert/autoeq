//! Composes ordered per-channel report blocks from prepared evidence.
//!
//! This module selects already-prepared curves and delegates figures, tables,
//! and filter presentation to their existing owners. It performs no DSP, file
//! access, measurement inference, or report-shell assembly.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value, json};

use crate::PreparedReportData;
use crate::report_dataset::ReportDatasetOptions;

/// Builds ordered channel-tab blocks for [`crate::report_assembly`].
///
/// Native replay states are authoritative when present. Markerless legacy
/// data uses only the explicitly supplied `legacy_source_curves`; the saved
/// graph's optimizer/final curves are not treated as a replay substitute.
/// Driver identities are matched by their original graph index and name, not
/// by the filtered display-tab index.
///
/// # Errors
/// Returns an error for invalid prepared input or report options.
pub fn build_dataset_channel_blocks(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<Vec<Value>, String> {
    let mut blocks = build_document_channel_blocks(prepared, options)?;
    for block in &mut blocks {
        if let Some(groups) = block["groups"].as_object_mut() {
            for group in groups.values_mut().filter_map(Value::as_array_mut) {
                for section in group {
                    crate::report_document::project_section(section)?;
                }
            }
        }
    }
    Ok(blocks)
}

pub(crate) fn build_document_channel_blocks(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<Vec<Value>, String> {
    prepared.validate()?;
    options.validate()?;

    let Some(channels) = prepared.graph.get("channels").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let entries = crate::report_data::driver_display_entries(&prepared.graph);
    let entries = entries
        .as_array()
        .ok_or("driver display entries must be an array")?;
    let mut reference_curves = Map::new();
    for channel_name in channels.keys() {
        let channel = &channels[channel_name];
        let curve = select_channel_curve(prepared, options, channel_name)
            .or_else(|| select_channel_baseline(prepared, channel_name, channel, None));
        if let Some(curve) = curve {
            reference_curves.insert(channel_name.clone(), curve);
        }
    }
    let mut selected = Vec::with_capacity(entries.len());
    for entry in entries {
        let channel_name = entry["channel"]
            .as_str()
            .ok_or("channel tab entry requires a channel name")?;
        let channel = channels
            .get(channel_name)
            .ok_or("channel tab entry must reference a graph channel")?;
        let driver_position = entry["source_driver_position"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok());
        let driver = driver_position
            .and_then(|position| channel["drivers"].as_array()?.get(position))
            .filter(|driver| driver.is_object());
        let label = entry["label"].as_str().unwrap_or(channel_name);
        let baseline = select_channel_baseline(prepared, channel_name, channel, driver);
        let final_curve = match driver {
            Some(driver) => select_driver_curve(prepared, options, channel_name, entry, driver),
            None => select_channel_curve(prepared, options, channel_name),
        };
        selected.push((
            entry,
            channel_name.to_owned(),
            label.to_owned(),
            channel.clone(),
            driver.cloned(),
            baseline,
            final_curve,
        ));
    }

    let target = (prepared.target_snapshot["state"] == "available")
        .then(|| prepared.target_snapshot["target"].clone());
    let reference_values = Value::Object(reference_curves);
    let overlays = crate::target_overlay::build_target_overlay_curves_with_status(
        &prepared.graph,
        &reference_values,
        target.as_ref(),
        options.route_crossover_db.as_ref(),
        options.route_crossover_status.as_ref(),
    );
    let mut blocks = Vec::with_capacity(selected.len());
    for (entry, channel_name, label, channel, driver, baseline, final_curve) in selected {
        let mut block = build_channel_block(ChannelBlockInput {
            prepared,
            options,
            entry,
            channel_name: &channel_name,
            label: &label,
            channel: &channel,
            driver: driver.as_ref(),
            baseline,
            final_curve,
            target_curve: overlays.get(&channel_name),
        })?;
        if let Some(target) = target.as_ref()
            && let Some(reason) = crate::target_overlay::route_target_unavailable_reason(
                &prepared.graph,
                &channel_name,
                target,
                options.route_crossover_db.as_ref(),
                options.route_crossover_status.as_ref(),
                reference_values.get(&channel_name),
            )
        {
            block["preparation_status"] = json!([{
                "section":"target_overlay",
                "channel":channel_name,
                "status":"unavailable",
                "reason":reason
            }]);
        }
        blocks.push(block);
    }
    Ok(blocks)
}

struct ChannelBlockInput<'a> {
    prepared: &'a PreparedReportData,
    options: &'a ReportDatasetOptions,
    entry: &'a Value,
    channel_name: &'a str,
    label: &'a str,
    channel: &'a Value,
    driver: Option<&'a Value>,
    baseline: Option<Value>,
    final_curve: Option<Value>,
    target_curve: Option<&'a Value>,
}

fn build_channel_block(input: ChannelBlockInput<'_>) -> Result<Value, String> {
    let ChannelBlockInput {
        prepared,
        options,
        entry,
        channel_name,
        label,
        channel,
        driver,
        baseline,
        final_curve,
        target_curve,
    } = input;
    let mut speakers = Vec::new();
    let mut time_domain = Vec::new();
    let mut epa = Vec::new();
    let is_driver = driver.is_some();
    let acoustic = if let Some(driver) = driver {
        driver
            .get("measured_acoustics")
            .filter(|value| value.is_object())
            .unwrap_or(driver)
    } else {
        channel
    };
    let mut caption = String::new();
    if driver.is_some() && final_curve.is_some() && prepared.replay.is_some() {
        caption = format!(
            "Corrected physical-driver response for {} includes the prepared driver response. Total driver EQ preview unavailable; individual filter details remain below.",
            channel_name
        );
    } else if driver.is_some() && final_curve.is_some() {
        caption = format!(
            "Physical driver output of {}: its measurement through this driver's own DSP chain (shared EQ + driver gain/crossover/route). The EQ plot below shows this total driver shaping.",
            channel_name
        );
    }
    if let Some(replay) = &prepared.replay {
        if let Some(driver) = driver.filter(|_| final_curve.is_none()) {
            caption = driver_replay_reason(replay, channel_name, entry, driver);
        } else if driver.is_none() && final_curve.is_none() {
            caption = routed_source_reason(replay, channel_name);
        }
    } else if let Some(driver) = driver.filter(|_| final_curve.is_none()) {
        let driver_label = entry["label"].as_str().unwrap_or(channel_name);
        let reason = if driver.get("initial_curve").is_some_and(is_curve) {
            "no prepared driver response was supplied"
        } else {
            "no measured driver response is available"
        };
        caption = format!(
            "Raw measurement only; corrected response unavailable for {driver_label}: {reason}."
        );
    }

    speakers.push(crate::report_document::channel_heading(label, &caption));
    if driver.is_none()
        && let Some(method) = prepared
            .replay
            .as_ref()
            .and_then(|replay| replay.context.get("channel_baselines"))
            .and_then(|baselines| baselines.get(channel_name))
            .and_then(|state| state.get("summation"))
            .and_then(Value::as_str)
    {
        let note = match method {
            "single_driver" => "Baseline uses the single available measured driver.",
            "coherent" => "Baseline uses the coherent measured-driver sum with saved phase.",
            "incoherent_power_estimate" => {
                "Baseline is an incoherent power estimate because measured phase is unavailable."
            }
            _ => "Baseline summation method is unavailable.",
        };
        speakers.push(crate::report_document::caption_section(note, false));
    }

    let response = crate::response_figures::response_figure_report(&json!({
        "channel_name": label,
        "initial_curve": baseline,
        "final_curve": final_curve,
        "target_curve": if is_driver { Value::Null } else { target_curve.cloned().unwrap_or(Value::Null) },
        "tab": label,
        "title_suffix": " (Full Range)",
    }));
    let mut figure = response["channel_figure_with_overlays"].clone();
    if !figure.is_null() {
        if let Some(eq_figure) =
            selected_eq_figure(prepared, options, entry, channel_name, channel, driver)?
        {
            let traces = eq_figure["figure"]["series"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            if !traces.is_empty() {
                figure["figure"]["y2"] =
                    json!({"label":"EQ gain (dB)","scale":"linear","min":-25.0,"max":25.0});
                if let Some(series) = figure["figure"]["series"].as_array_mut() {
                    series.extend(traces.into_iter().map(|mut trace| {
                        trace["y_axis"] = json!(1);
                        trace
                    }));
                }
            }
        }
        push_figure(&mut speakers, figure);
    }
    let mut landmark_channel = channel.clone();
    if let Some(curve) = &baseline {
        landmark_channel["initial_curve"] = curve.clone();
    }
    let landmarks = crate::acoustic_curves::response_landmarks(&landmark_channel);
    if let Some(table) = crate::acoustic_curves::landmarks_table([(label.into(), landmarks)]) {
        speakers.push(crate::report_document::diagnostic_section(vec![table]));
    }

    if is_driver {
        let rate = acoustic.get("sample_rate_hz").and_then(Value::as_f64);
        if let Some(rate) = rate {
            time_domain.push(crate::report_document::paragraph_section(&[format!(
                "Independent measured driver IR: {label}; native sample rate {rate:.1} Hz. This is not the summed parent-speaker response. No measured post-DSP capture is implied."
            )]));
            if rate <= 16_000.0 {
                time_domain.push(crate::report_document::paragraph_section(&[
                    "1–8 kHz reflection analysis is unavailable at this native sample rate. T60 bands beyond Nyquist are marked unavailable."
                ]));
            }
        }
    }
    let ir_pre = if is_driver {
        acoustic
            .get("pre_ir")
            .or_else(|| driver.and_then(|value| value.get("pre_ir")))
    } else {
        channel.get("pre_ir")
    };
    let ir_post = if is_driver {
        driver.and_then(|value| value.get("post_ir"))
    } else {
        channel.get("post_ir")
    };
    let ir = crate::temporal_figures::build_temporal_figure(
        "create_ir_figure",
        &json!({
            "channel_name":label,"pre_ir":ir_pre.unwrap_or(&Value::Null),"post_ir":ir_post.unwrap_or(&Value::Null),"tab":label
        }),
    )?;
    push_figure(&mut time_domain, ir);

    let has_acoustic_evidence = !is_driver
        || driver
            .and_then(|value| value.get("measured_acoustics"))
            .and_then(Value::as_object)
            .is_some_and(|value| !value.is_empty());
    if has_acoustic_evidence {
        time_domain.push(crate::report_document::diagnostic_section(
            crate::channel_evidence::speech_transmission(acoustic, label),
        ));
        let (reflection, figures) =
            crate::acoustic_reflections::reflection_document(acoustic, label, Some(label));
        time_domain.push(crate::report_document::diagnostic_section(vec![reflection]));
        for figure in figures {
            push_figure(&mut time_domain, figure);
        }
        let early_late = acoustic
            .get("early_late_curves")
            .filter(|value| !value.is_null());
        if let Some(curves) = early_late {
            let figure = crate::temporal_figures::build_temporal_figure(
                "create_early_late_figure",
                &json!({
                    "label":label,"report":curves,"tab":label
                }),
            )?;
            let caption = if figure.is_null() {
                "Early vs late sound figure is unavailable for the saved evidence."
            } else {
                push_figure(&mut time_domain, figure);
                "Third-octave early and late energy contributions share the full curve's peak-band reference. Full is an incoherent energy sum, not a coherent pressure response."
            };
            time_domain.push(crate::report_document::caption_section(caption, false));
        } else {
            time_domain.push(crate::report_document::caption_section(
                "Early vs late sound: pending roomeq field early_late_curves with a shared level reference.",
                true,
            ));
        }

        if is_driver {
            let reference =
                crate::acoustic_t60::itu_reference(&json!({"channels":{label:acoustic}}));
            let rows = crate::acoustic_t60::t60_rows(acoustic);
            time_domain.push(crate::report_document::diagnostic_section(vec![
                crate::acoustic_t60::t60_band_table(rows.as_deref()),
            ]));
            let t60 = crate::temporal_figures::build_temporal_figure(
                "create_t60_octaves_figure",
                &json!({
                    "label":label,"rows":rows,"tab":label,"itu_reference":reference
                }),
            )?;
            let has_t60_figure = !t60.is_null();
            push_figure(&mut time_domain, t60);
            if has_t60_figure {
                time_domain.push(crate::report_document::diagnostic_section(vec![
                    crate::acoustic_t60::t60_reference(reference.as_ref().unwrap_or(&Value::Null)),
                ]));
            }
        }
        if let Some(waterfall) = crate::channel_evidence::optimization_waterfall(
            &acoustic["waterfall"],
            &acoustic["resonance_decays"],
        ) {
            time_domain.push(crate::report_document::diagnostic_section(waterfall));
        } else {
            time_domain.push(crate::report_document::caption_section(
                "Waterfall and resonance decay: pending roomeq fields waterfall and resonance_decays from a measured room impulse response.", true,
            ));
        }
        if let Some(wavelet) = crate::channel_evidence::optimization_wavelet(&acoustic["wavelet"]) {
            time_domain.push(crate::report_document::diagnostic_section(wavelet));
        } else {
            time_domain.push(crate::report_document::caption_section(
                "Three-cycle wavelet: pending roomeq field wavelet from a measured room impulse response.", true,
            ));
        }
    }

    let epa_input = &prepared.graph["metadata"]["epa_per_channel"][channel_name];
    if let Some(scores) = crate::evidence_sections::epa_channel_block(epa_input) {
        epa.push(crate::report_document::diagnostic_section(vec![scores]));
    }

    let passes = if let Some(driver) = driver {
        split_driver_passes(channel, driver)
    } else {
        eq_passes(channel)
    };
    if let Some(driver) = driver {
        let driver_passes = passes.0;
        let shared_passes = passes.1;
        if let Some(listing) = crate::eq_sections::driver_filter_listing(&json!({
            "driver_name":label,"channel_name":channel_name,"id_prefix":format!("eq_{}",entry["driver"].as_u64().unwrap_or(0)),
            "driver_passes":driver_passes,"shared_passes":shared_passes
        })) {
            speakers.push(crate::report_document::eq_filter_section(listing));
        }
        speakers.push(crate::report_document::filter_summary(
            "Sub DSP Chain",
            driver_shaping_summary(driver),
        ));
    } else {
        let filters = passes.0;
        let has_labeled = filters.as_array().into_iter().flatten().any(|pass| {
            pass["label"]
                .as_str()
                .is_some_and(|value| !value.is_empty())
        });
        let filter_count = filters.as_array().into_iter().flatten().any(|pass| {
            pass["filters"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        });
        if filter_count {
            let listing = crate::eq_sections::channel_filter_listing(&json!({
                "title":if has_labeled {"EQ Filters (3-Pass Pipeline)"} else {"EQ Filters"},"labeled":has_labeled,"passes":filters
            }));
            speakers.push(crate::report_document::eq_filter_section(listing));
        }
    }

    for group in [&mut speakers, &mut time_domain, &mut epa] {
        for section in group {
            section["tab"] = json!(label);
        }
    }
    Ok(json!({"label":label,"groups":{"speakers":speakers,"time_domain":time_domain,"epa":epa}}))
}

fn select_channel_baseline(
    prepared: &PreparedReportData,
    channel_name: &str,
    channel: &Value,
    driver: Option<&Value>,
) -> Option<Value> {
    if let Some(driver) = driver {
        let curve = driver.get("initial_curve")?;
        return Some(crate::report_data::clip_curve_to_measured_band(
            curve, driver,
        ));
    }
    if prepared.replay.is_some()
        && channel["drivers"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    {
        return tagged_channel_curve(
            prepared
                .replay
                .as_ref()?
                .context
                .get("channel_baselines")?
                .get(channel_name)?,
        );
    }
    channel
        .get("initial_curve")
        .filter(|value| !value.is_null())
        .cloned()
}

fn select_channel_curve(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
    channel_name: &str,
) -> Option<Value> {
    if let Some(replay) = &prepared.replay {
        return replay
            .context
            .get("routed_sources")
            .and_then(|items| items.get(channel_name))
            .and_then(tagged_channel_curve);
    }
    options
        .legacy_source_curves
        .as_ref()?
        .get(channel_name)
        .and_then(tagged_channel_curve)
        .or_else(|| {
            options
                .legacy_source_curves
                .as_ref()?
                .get(channel_name)
                .filter(|value| is_curve(value))
                .cloned()
        })
}

fn select_driver_curve(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
    channel_name: &str,
    entry: &Value,
    driver: &Value,
) -> Option<Value> {
    if let Some(replay) = &prepared.replay {
        let candidates = replay
            .context
            .get("drivers")?
            .get(channel_name)?
            .as_array()?;
        let state = match_identity(candidates, entry, driver)?;
        let curve = tagged_channel_curve(state.get("state")?)?;
        return Some(crate::report_data::clip_curve_to_measured_band(
            &curve, driver,
        ));
    }
    let candidates = options
        .legacy_source_curves
        .as_ref()?
        .get("drivers")?
        .get(channel_name)?
        .as_array()?;
    match_identity(candidates, entry, driver)
        .and_then(|candidate| tagged_channel_curve(candidate.get("state")?))
        .map(|curve| crate::report_data::clip_curve_to_measured_band(&curve, driver))
}

fn select_eq_curve(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
    entry: &Value,
    channel_name: &str,
    driver: Option<&Value>,
    channel: &Value,
) -> Option<Value> {
    if let Some(driver) = driver {
        let curves = options
            .driver_eq_curves
            .as_ref()?
            .get(channel_name)?
            .as_array()?;
        return match_identity(curves, entry, driver)
            .and_then(|item| tagged_channel_curve(item.get("state")?));
    }
    if let Some(replay) = &prepared.replay {
        return replay
            .context
            .get("channels")
            .and_then(|channels| channels.get(channel_name))
            .filter(|state| state["status"] == "available")
            .and_then(|state| state.get("transfer_curve"))
            .filter(|curve| is_curve(curve))
            .cloned();
    }
    channel
        .get("eq_response")
        .filter(|value| is_curve(value))
        .cloned()
}

fn selected_eq_figure(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
    entry: &Value,
    channel_name: &str,
    channel: &Value,
    driver: Option<&Value>,
) -> Result<Option<Value>, String> {
    let Some(curve) = select_eq_curve(prepared, options, entry, channel_name, driver, channel)
    else {
        return Ok(None);
    };
    let Some(freq) = curve.get("freq").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(spl) = curve.get("spl").and_then(Value::as_array) else {
        return Ok(None);
    };
    if freq.len() != spl.len() {
        return Ok(None);
    }
    let request = json!({"channel_name":entry["label"],"tab":entry["label"],"freq_points":freq,"combined_response":spl,"filters":[]});
    crate::eq_figures::build_eq_figure("single", &request).map(Some)
}

fn match_identity<'a>(candidates: &'a [Value], entry: &Value, driver: &Value) -> Option<&'a Value> {
    let source_index = entry
        .get("source_driver_index")
        .filter(|value| !value.is_null());
    let source_name = entry
        .get("source_driver_name")
        .filter(|value| !value.is_null());
    if source_index.is_none() && source_name.is_none() {
        return None;
    }
    if source_index.is_some_and(|index| driver.get("index") != Some(index))
        || source_name.is_some_and(|name| driver.get("name") != Some(name))
    {
        return None;
    }
    let matches: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            let index_matches =
                source_index.is_none_or(|index| candidate.get("index") == Some(index));
            let name_matches = source_name.is_none_or(|name| candidate.get("name") == Some(name));
            index_matches && name_matches
        })
        .collect();
    if matches.len() == 1 {
        return matches.first().copied();
    }
    let _ = driver;
    None
}

fn tagged_channel_curve(state: &Value) -> Option<Value> {
    match state.get("status").and_then(Value::as_str)? {
        "available" => state.get("curve").filter(|curve| is_curve(curve)).cloned(),
        _ => None,
    }
}

fn is_curve(value: &Value) -> bool {
    value["freq"].as_array().is_some() && value["spl"].as_array().is_some()
}

fn driver_replay_reason(
    replay: &crate::PreparedReplay,
    channel_name: &str,
    entry: &Value,
    driver: &Value,
) -> String {
    let Some(candidates) = replay
        .context
        .get("drivers")
        .and_then(|value| value.get(channel_name))
        .and_then(Value::as_array)
    else {
        return format!(
            "Corrected response unavailable for {}: no prepared driver evidence was supplied.",
            entry["label"].as_str().unwrap_or(channel_name)
        );
    };
    let Some(item) = match_identity(candidates, entry, driver) else {
        return format!(
            "Corrected response unavailable for {}: source driver identity is missing or ambiguous.",
            entry["label"].as_str().unwrap_or(channel_name)
        );
    };
    let reason = item["state"]["reason"]
        .as_str()
        .unwrap_or("prepared response is unavailable");
    format!(
        "Raw measurement only; corrected response unavailable for {}: {}.",
        entry["label"].as_str().unwrap_or(channel_name),
        reason
    )
}

fn routed_source_reason(replay: &crate::PreparedReplay, channel_name: &str) -> String {
    let state = replay
        .context
        .get("routed_sources")
        .and_then(|items| items.get(channel_name));
    let reason = state
        .and_then(|value| value["reason"].as_str())
        .unwrap_or("no routed-source response was prepared");
    format!(
        "Corrected routed response unavailable for {}: {}.",
        channel_name, reason
    )
}

fn eq_passes(channel: &Value) -> (Value, Value) {
    (passes_for_plugins(&channel["plugins"]), json!([]))
}

fn split_driver_passes(channel: &Value, driver: &Value) -> (Value, Value) {
    let driver_plugins = filter_plugins(&driver["plugins"], true);
    let shared_plugins = filter_plugins(&channel["plugins"], true);
    (
        passes_for_plugins(&driver_plugins),
        passes_for_plugins(&shared_plugins),
    )
}

fn filter_plugins(plugins: &Value, skip_route_owned: bool) -> Value {
    let values: Vec<_> = plugins
        .as_array()
        .into_iter()
        .flatten()
        .filter(|plugin| {
            plugin["plugin_type"] == "eq"
                && plugin["parameters"]["filters"]
                    .as_array()
                    .is_some_and(|filters| !filters.is_empty())
                && (!skip_route_owned || plugin["parameters"]["room_eq_stage"] != "route_owned")
        })
        .cloned()
        .collect();
    Value::Array(values)
}

fn passes_for_plugins(plugins: &Value) -> Value {
    let display = [
        (
            "cea2034_speaker_correction",
            "Pass 1: Speaker Correction (CEA2034)",
            "rgba(255, 165, 0, 0.9)",
        ),
        (
            "room_eq_correction",
            "Pass 2: Room EQ",
            "rgba(100, 100, 255, 0.9)",
        ),
        (
            "user_preference",
            "Pass 3: User Preference",
            "rgba(180, 100, 255, 0.9)",
        ),
    ];
    let mut passes = Vec::new();
    for plugin in plugins.as_array().into_iter().flatten() {
        let parameters = &plugin["parameters"];
        let label = parameters["label"].as_str().unwrap_or("");
        let filters = parameters["filters"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if filters.is_empty() {
            continue;
        }
        let (display_name, color) = display
            .iter()
            .find(|(known, _, _)| *known == label)
            .map(|(_, name, color)| (*name, *color))
            .unwrap_or(("Room EQ", "rgba(100, 100, 255, 0.9)"));
        let descriptors: Vec<_> = filters.iter().map(filter_descriptor).collect();
        passes.push(
            json!({"label":label,"display_name":display_name,"color":color,"filters":descriptors}),
        );
    }
    Value::Array(passes)
}

fn filter_descriptor(filter: &Value) -> Value {
    let mut output = json!({
        "topology":filter.get("topology").cloned().unwrap_or(Value::Null),
        "filter_type":filter["filter_type"].as_str().unwrap_or("peak"),
        "freq":filter.get("freq").cloned().unwrap_or(json!(0)),
        "q":filter.get("q").cloned().unwrap_or(json!(1)),
        "db_gain":filter.get("db_gain").cloned().unwrap_or(json!(0)),
    });
    let sections = filter
        .get("kautz_sections")
        .or_else(|| filter.get("sections"))
        .and_then(Value::as_array);
    if filter["topology"] == "kautz_filter" {
        let converted: Vec<_> = sections.into_iter().flatten().map(|section| json!({
            "pole":section.get("pole_freq").or_else(||section.get("freq")).or_else(||section.get("frequency")).or_else(||section.get("pole_freq_hz")).cloned().unwrap_or(Value::Null),
            "q":section.get("q").cloned().unwrap_or(Value::Null),
            "weight":section.get("gain").cloned().unwrap_or(json!(0)),
        })).collect();
        output["sections"] = json!(converted);
    }
    output
}

fn driver_shaping_summary(driver: &Value) -> String {
    let mut gain_db = 0.0;
    let mut delay_ms = 0.0;
    let mut inverted = false;
    let mut low_pass = None;
    for plugin in driver["plugins"].as_array().into_iter().flatten() {
        let params = &plugin["parameters"];
        match plugin["plugin_type"].as_str().unwrap_or_default() {
            "gain" => {
                gain_db += params["gain_db"].as_f64().unwrap_or(0.0);
                inverted |= params["invert"].as_bool().unwrap_or(false);
            }
            "delay" => delay_ms += params["delay_ms"].as_f64().unwrap_or(0.0),
            "crossover"
                if params["output"]
                    .as_str()
                    .unwrap_or_default()
                    .to_lowercase()
                    .starts_with("low") =>
            {
                if let Some(frequency) = params["frequency"].as_f64().filter(|value| *value > 0.0) {
                    low_pass = Some((frequency, params["type"].as_str().map(str::to_owned)));
                }
            }
            _ => {}
        }
    }
    let mut parts = vec![format!("driver gain {gain_db:+.1} dB")];
    if inverted {
        parts.push("polarity inverted".into());
    }
    if delay_ms.abs() > 1.0e-9 {
        parts.push(format!("delay {delay_ms:.3} ms"));
    }
    if let Some((hz, kind)) = low_pass {
        parts.push(format!(
            "low-pass{} @ {hz:.1} Hz",
            kind.map(|value| format!(" {value}")).unwrap_or_default()
        ));
    }
    parts.join("; ")
}

fn push_figure(target: &mut Vec<Value>, figure: Value) {
    if !figure.is_null() {
        target.push(figure);
    }
}

#[cfg(test)]
mod tests {
    use super::{build_dataset_channel_blocks, is_curve};
    use crate::report_dataset::ReportDatasetOptions;
    use crate::{PreparedReplay, PreparedReportData};
    use serde_json::{Map, Value, json};

    fn prepared(graph: Value, replay: Value) -> PreparedReportData {
        let replay: PreparedReplay = serde_json::from_value(replay).unwrap();
        PreparedReportData::new(
            graph,
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            Some(replay),
        )
        .unwrap()
    }

    fn graph() -> Value {
        json!({"channels":{"Sub":{
            "initial_curve":{"freq":[50.0,100.0,200.0],"spl":[70.0,71.0,72.0]},
            "plugins":[],
            "drivers":[
                {"index":2,"name":"unplottable","initial_curve":null},
                {"index":7,"name":"Driver Seven","initial_curve":{"freq":[20.0,50.0,100.0,200.0,500.0],"spl":[65.0,70.0,71.0,72.0,60.0]},"measured_band_hz":[50.0,200.0],"plugins":[]}
            ]
        }},"metadata":{"epa_per_channel":{}}})
    }

    fn figure_series(block: &Value) -> Vec<Value> {
        block["groups"]["speakers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|section| section["kind"] == "figure")
            .and_then(|section| section["figure"]["series"].as_array())
            .unwrap()
            .clone()
    }

    #[test]
    fn driver_tab_uses_original_identity_not_filtered_display_index() {
        let data = prepared(
            graph(),
            json!({
                "sample_rate_hz":48000,
                "drivers":{"Sub":[
                    {"index":7,"name":"Driver Seven","state":{"status":"available","curve":{"freq":[50.0,100.0,200.0],"spl":[68.0,69.0,70.0]}}}
                ]},
                "routed_sources":{}
            }),
        );
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["label"], "Driver Seven");
        let series = figure_series(&blocks[0]);
        assert!(series.iter().any(
            |series| series["name"] == "Before EQ" && series["y"] == json!([70.0, 71.0, 72.0])
        ));
        assert!(
            series
                .iter()
                .any(|series| series["name"] == "After EQ"
                    && series["y"] == json!([68.0, 69.0, 70.0]))
        );
    }

    #[test]
    fn serialized_early_late_evidence_reaches_temporal_figure_builder() {
        let curves = json!({
            "method":"incoherent_band_energy",
            "reference":"full_peak_band",
            "smoothing":"third_octave",
            "split_ms":20.0,
            "full":{"freq":[100.0,125.0],"spl":[0.0,-1.0]},
            "early":{"freq":[100.0,125.0],"spl":[-1.0,-2.0]},
            "late":{"freq":[100.0,125.0],"spl":[-7.0,-8.0]}
        });
        let prepared = PreparedReportData::new(
            json!({"channels":{"L":{"initial_curve":{"freq":[100.0,125.0],"spl":[0.0,-1.0]},"early_late_curves":curves}},"metadata":{}}),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        ).unwrap();

        let blocks =
            build_dataset_channel_blocks(&prepared, &ReportDatasetOptions::default()).unwrap();
        let time_domain = &blocks[0]["groups"]["time_domain"];
        assert!(time_domain.as_array().unwrap().iter().any(|section| {
            section["kind"] == "figure"
                && section["figure"]["title"] == "L: early vs late band energy (20 ms split)"
        }));
    }

    #[test]
    fn native_unavailable_driver_does_not_fall_back_to_saved_final_curve() {
        let mut source = graph();
        source["channels"]["Sub"]["drivers"][1]["final_curve"] =
            json!({"freq":[50.0,100.0,200.0],"spl":[99.0,99.0,99.0]});
        let data = prepared(
            source,
            json!({
                "sample_rate_hz":48000,
                "drivers":{"Sub":[
                    {"index":7,"name":"Driver Seven","state":{"status":"unavailable","reason":"required route resource is absent"}}
                ]},
                "routed_sources":{}
            }),
        );
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        assert!(
            blocks[0]["groups"]["speakers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|section| {
                    section["html"]
                        .as_str()
                        .is_some_and(|html| html.contains("required route resource is absent"))
                })
        );
        assert!(
            !figure_series(&blocks[0])
                .iter()
                .any(|series| series["name"] == "After EQ")
        );
    }

    #[test]
    fn channel_block_exposes_unavailable_lfe_target_shaping() {
        let mut source = graph();
        source["metadata"]["bass_management"]["routing_graph"]["routes"] = json!([{
            "source_channel":"Sub","route_kind":"lfe_lowpass_to_sub","low_pass_hz":80.0
        }]);
        let replay: PreparedReplay = serde_json::from_value(json!({
            "sample_rate_hz":null,
            "drivers":{"Sub":[
                {"index":7,"name":"Driver Seven","state":{"status":"unavailable","reason":"driver replay unavailable"}}
            ]},
            "routed_sources":{"Sub":{"status":"available","curve":{"freq":[50.0,100.0,200.0],"spl":[70.0,71.0,72.0]}}}
        }))
        .unwrap();
        let prepared = PreparedReportData::new(
            source,
            None,
            None,
            None,
            json!({"schema_version":1,"state":"available","target":{"freq":[20.0,20000.0],"spl":[75.0,75.0]}}),
            Some(replay),
        )
        .unwrap();
        let options = ReportDatasetOptions {
            route_crossover_db: Some(json!({})),
            route_crossover_status: Some(json!({})),
            ..ReportDatasetOptions::default()
        };
        let blocks = build_dataset_channel_blocks(&prepared, &options).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0]["preparation_status"][0]["section"],
            "target_overlay"
        );
        assert_eq!(blocks[0]["preparation_status"][0]["channel"], "Sub");
        assert_eq!(blocks[0]["preparation_status"][0]["status"], "unavailable");
        assert!(
            blocks[0]["groups"]["speakers"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|section| section["figure"]["series"].as_array())
                .flatten()
                .all(|series| series["name"] != "Sub target")
        );
    }

    #[test]
    fn ambiguous_driver_identity_and_missing_identity_fail_closed() {
        let mut replay = Map::new();
        replay.insert("sample_rate_hz".into(), json!(48000));
        replay.insert("drivers".into(), json!({"Sub":[
            {"index":7,"name":"Driver Seven","state":{"status":"available","curve":{"freq":[50.0,100.0],"spl":[90.0,90.0]}}},
            {"index":7,"name":"Driver Seven","state":{"status":"available","curve":{"freq":[50.0,100.0],"spl":[80.0,80.0]}}}
        ]}));
        replay.insert("routed_sources".into(), json!({}));
        let data = prepared(graph(), Value::Object(replay));
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        assert!(
            !figure_series(&blocks[0])
                .iter()
                .any(|series| series["name"] == "After EQ")
        );

        let mut source = graph();
        source["channels"]["Sub"]["drivers"][1]
            .as_object_mut()
            .unwrap()
            .remove("index");
        source["channels"]["Sub"]["drivers"][1]
            .as_object_mut()
            .unwrap()
            .remove("name");
        let data = prepared(
            source,
            json!({"sample_rate_hz":48000,"drivers":{"Sub":[
                {"index":7,"name":"Driver Seven","state":{"status":"available","curve":{"freq":[50.0,100.0],"spl":[90.0,90.0]}}}
            ]},"routed_sources":{}}),
        );
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        assert!(
            !figure_series(&blocks[0])
                .iter()
                .any(|series| series["name"] == "After EQ")
        );
    }

    #[test]
    fn channel_baseline_and_routed_response_are_taken_from_native_preparation() {
        // Driver tabs remain authoritative for multi-driver channels; channel-level
        // baseline routing is exercised by a channel without plottable drivers.
        let mut source = graph();
        source["channels"]["Sub"]["drivers"] =
            json!([{"index":3,"name":"unmeasured","initial_curve":null}]);
        let data = prepared(
            source,
            json!({
                "sample_rate_hz":48000,
                "channel_baselines":{"Sub":{"status":"available","summation":"single_driver","curve":{"freq":[50.0,100.0,200.0],"spl":[60.0,61.0,62.0]}}},
                "routed_sources":{"Sub":{"status":"available","curve":{"freq":[50.0,100.0,200.0],"spl":[55.0,56.0,57.0]}}}
            }),
        );
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        let series = figure_series(&blocks[0]);
        assert!(series.iter().any(
            |series| series["name"] == "Before EQ" && series["y"] == json!([60.0, 61.0, 62.0])
        ));
        assert!(
            series
                .iter()
                .any(|series| series["name"] == "After EQ"
                    && series["y"] == json!([55.0, 56.0, 57.0]))
        );
        assert!(
            blocks[0]["groups"]["speakers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|section| {
                    section["html"].as_str().is_some_and(|html| {
                        html.contains("Baseline uses the single available measured driver.")
                    })
                })
        );
    }

    #[test]
    fn available_file_target_uses_measured_baseline_when_corrected_curve_is_missing() {
        let source = json!({"channels":{"L":{
            "initial_curve":{"freq":[50.0,100.0,200.0],"spl":[70.0,71.0,72.0]},
            "plugins":[]
        }},"metadata":{"effective_config":{"optimizer":{"min_freq":50.0,"max_freq":200.0}}}});
        let mut data = prepared(
            source,
            json!({
                "sample_rate_hz":48000,
                "channels":{"L":{"status":"unavailable","reason":"no prepared corrected response"}},
                "routed_sources":{"L":{"status":"unavailable","reason":"no prepared corrected response"}}
            }),
        );
        data.target_snapshot = json!({
            "schema_version":1,
            "state":"available",
            "target":{"freq":[50.0,100.0,200.0],"spl":[75.0,76.0,77.0]}
        });

        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        let series = figure_series(&blocks[0]);
        assert!(
            series.iter().any(|item| {
                item["name"] == "Before EQ" && item["y"] == json!([70.0, 71.0, 72.0])
            })
        );
        assert!(series.iter().any(|item| item["name"] == "Target"));
        assert!(!series.iter().any(|item| item["name"] == "After EQ"));
        assert!(
            blocks[0]["groups"]["speakers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|section| section["html"]
                    .as_str()
                    .is_some_and(|html| { html.contains("no prepared corrected response") }))
        );
    }

    #[test]
    fn native_channel_eq_preview_uses_prepared_transfer_not_saved_curve() {
        let source = json!({"channels":{"L":{
            "initial_curve":{"freq":[50.0,100.0,200.0],"spl":[70.0,71.0,72.0]},
            "eq_response":{"freq":[50.0,100.0,200.0],"spl":[99.0,99.0,99.0]},
            "plugins":[]
        }}});
        let available = prepared(
            source.clone(),
            json!({
                "sample_rate_hz":48000,
                "channels":{"L":{"status":"available","transfer_curve":{"freq":[50.0,100.0,200.0],"spl":[-5.0,-4.0,-3.0]}}},
                "routed_sources":{"L":{"status":"available","curve":{"freq":[50.0,100.0,200.0],"spl":[65.0,67.0,69.0]}}}
            }),
        );
        let blocks =
            build_dataset_channel_blocks(&available, &ReportDatasetOptions::default()).unwrap();
        let series = figure_series(&blocks[0]);
        assert!(
            series
                .iter()
                .any(|item| item["y"] == json!([-5.0, -4.0, -3.0]))
        );
        assert!(
            !series
                .iter()
                .any(|item| item["y"] == json!([99.0, 99.0, 99.0]))
        );

        let unavailable = prepared(
            source,
            json!({
                "sample_rate_hz":48000,
                "channels":{"L":{"status":"unavailable","reason":"no canonical transfer was prepared"}},
                "routed_sources":{"L":{"status":"unavailable","reason":"no routed source was prepared"}}
            }),
        );
        let blocks =
            build_dataset_channel_blocks(&unavailable, &ReportDatasetOptions::default()).unwrap();
        assert!(
            !figure_series(&blocks[0])
                .iter()
                .any(|item| item["y"] == json!([99.0, 99.0, 99.0]))
        );
    }

    #[test]
    fn full_python_channel_blocks_fixture_preserves_source_order_and_section_content() {
        let cases: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/report_dataset_full_python_goldens.json"
        ))
        .unwrap();
        let fixture = &cases[0];
        let graph = fixture["source_data"].clone();
        let mut legacy = Map::new();
        let channels = graph["channels"].as_object().unwrap();
        for (name, channel) in channels {
            if let Some(curve) = channel.get("final_curve").filter(|curve| is_curve(curve)) {
                legacy.insert(name.clone(), json!({"status":"available","curve":curve}));
            }
        }
        let mut driver_sources = Map::new();
        for (channel_name, channel) in channels {
            if let Some(drivers) = channel["drivers"].as_array() {
                let sources = drivers
                    .iter()
                    .map(|driver| {
                        let mut entry = Map::new();
                        if let Some(name) = driver.get("name") {
                            entry.insert("name".into(), name.clone());
                        }
                        if let Some(index) = driver.get("index") {
                            entry.insert("index".into(), index.clone());
                        }
                        // Graph final curves are not prepared replay output.
                        let state = json!({"status":"unavailable","reason":"no frozen driver replay in fixture"});
                        entry.insert("state".into(), state);
                        Value::Object(entry)
                    })
                    .collect::<Vec<_>>();
                driver_sources.insert(channel_name.clone(), Value::Array(sources));
            }
        }
        legacy.insert("drivers".into(), Value::Object(driver_sources));
        let options = ReportDatasetOptions {
            legacy_source_curves: Some(Value::Object(legacy)),
            ..ReportDatasetOptions::default()
        };
        let prepared = PreparedReportData::new(
            graph,
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();
        let actual = build_dataset_channel_blocks(&prepared, &options).unwrap();
        let expected = fixture["request"]["channel_blocks"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual
                .iter()
                .map(|block| &block["label"])
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|block| &block["label"])
                .collect::<Vec<_>>()
        );
        for (block_index, (actual_block, expected_block)) in actual.iter().zip(expected).enumerate()
        {
            for group in ["speakers", "time_domain", "epa"] {
                let actual_sections = actual_block["groups"][group].as_array().unwrap();
                let expected_sections = expected_block["groups"][group].as_array().unwrap();
                assert_eq!(
                    actual_sections.len(),
                    expected_sections.len(),
                    "block {block_index} group {group}"
                );
                for (section_index, (actual_section, expected_section)) in
                    actual_sections.iter().zip(expected_sections).enumerate()
                {
                    let mut actual_section = actual_section.clone();
                    let mut expected_section = expected_section.clone();
                    if group == "speakers"
                        && section_index == 0
                        && ["Sub A", "Sub B"].contains(&actual_block["label"].as_str().unwrap())
                    {
                        let label = actual_block["label"].as_str().unwrap();
                        let actual_html = actual_section["html"].as_str().unwrap();
                        let legacy_html = expected_section["html"].as_str().unwrap();
                        assert!(actual_html.starts_with(&format!("<h2>Channel: {label}</h2>\n")));
                        assert!(
                            actual_html
                                .contains("Raw measurement only; corrected response unavailable")
                        );
                        assert!(legacy_html.contains("Physical driver output"));
                        actual_section["html"] = json!(format!("<h2>Channel: {label}</h2>\n"));
                        expected_section["html"] = json!(format!("<h2>Channel: {label}</h2>\n"));
                    }
                    if let Some(path) =
                        first_difference(&actual_section, &expected_section, String::new())
                    {
                        panic!(
                            "Python channel golden mismatch at block {block_index} group {group} section {section_index}: {path}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn native_available_driver_replay_uses_prepared_curve_and_keeps_measured_ir() {
        let mut source = graph();
        let driver = &mut source["channels"]["Sub"]["drivers"][1];
        driver["final_curve"] = json!({"freq":[50.0,100.0,200.0],"spl":[99.0,99.0,99.0]});
        driver["sample_rate_hz"] = json!(48000);
        driver["pre_ir"] = json!({"time_ms":[0.0,1.0,2.0,3.0],"amplitude":[0.0,1.0,0.5,0.0]});
        driver["post_ir"] = json!({"time_ms":[0.0,1.0,2.0,3.0],"amplitude":[0.0,0.8,0.4,0.0]});
        let data = prepared(
            source,
            json!({
                "sample_rate_hz":48000,
                "drivers":{"Sub":[
                    {"index":7,"name":"Driver Seven","state":{"status":"available","curve":{"freq":[50.0,100.0,200.0],"spl":[62.0,63.0,64.0]}}}
                ]},
                "routed_sources":{}
            }),
        );
        let blocks = build_dataset_channel_blocks(&data, &ReportDatasetOptions::default()).unwrap();
        assert_eq!(blocks.len(), 1);
        let block = &blocks[0];
        assert_eq!(block["label"], "Driver Seven");
        let series = figure_series(block);
        assert!(
            series.iter().any(|item| {
                item["name"] == "After EQ" && item["y"] == json!([62.0, 63.0, 64.0])
            })
        );
        assert!(
            !series
                .iter()
                .any(|item| item["y"] == json!([99.0, 99.0, 99.0]))
        );
        let time_sections = block["groups"]["time_domain"].as_array().unwrap();
        assert!(time_sections.iter().any(|section| {
            section["kind"] == "html"
                && section["html"].as_str().is_some_and(|html| {
                    html.contains("Independent measured driver IR") && html.contains("48000.0 Hz")
                })
        }));
        let ir = time_sections
            .iter()
            .find(|section| section["kind"] == "figure")
            .expect("driver metadata IR should produce a temporal figure");
        let ir_series = ir["figure"]["series"].as_array().unwrap();
        assert!(ir_series.iter().any(|item| item["name"] == "Before EQ"));
        assert!(ir_series.iter().any(|item| item["name"] == "After EQ"));
    }

    fn first_difference(left: &Value, right: &Value, path: String) -> Option<String> {
        match (left, right) {
            (Value::Object(left), Value::Object(right)) => {
                for key in left.keys().chain(right.keys()) {
                    let nested = format!("{path}.{key}");
                    match (left.get(key), right.get(key)) {
                        (Some(left), Some(right)) => {
                            if let Some(diff) = first_difference(left, right, nested) {
                                return Some(diff);
                            }
                        }
                        _ => return Some(nested),
                    }
                }
                None
            }
            (Value::Array(left), Value::Array(right)) => {
                for index in 0..left.len().max(right.len()) {
                    let nested = format!("{path}[{index}]");
                    match (left.get(index), right.get(index)) {
                        (Some(left), Some(right)) => {
                            if let Some(diff) = first_difference(left, right, nested) {
                                return Some(diff);
                            }
                        }
                        _ => return Some(nested),
                    }
                }
                None
            }
            _ if left == right => None,
            _ => Some(format!("{path}: actual={left}, expected={right}")),
        }
    }
}
