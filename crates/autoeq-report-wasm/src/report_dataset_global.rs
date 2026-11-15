//! Composes global dashboard groups from one frozen report dataset.
//!
//! The composer delegates response, acoustic, target, temporal, and evidence
//! calculations to their existing owners. It performs no file access or DSP.

// Rust guideline compliant 2026-02-21

use serde_json::{Map, Value, json};

use crate::PreparedReportData;
use crate::report_dataset::ReportDatasetOptions;

/// Builds global report sections from prepared graph and host evidence.
///
/// The result has `status_sections`, `global_groups`, `relative_levels_html`,
/// `modes_html`, and structured `preparation_status` fields.
///
/// # Errors
/// Returns an error for invalid prepared data, options, or required evidence shapes.
pub fn build_global_report_sections(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<Value, String> {
    let mut global = build_global_document_sections(prepared, options)?;
    if let Some(groups) = global["global_groups"].as_object_mut() {
        for group in groups.values_mut().filter_map(Value::as_array_mut) {
            for section in group {
                crate::report_document::project_section(section)?;
            }
        }
    }
    if let Some(sections) = global["status_sections"].as_array_mut() {
        for section in sections {
            crate::report_document::project_section(section)?;
        }
    }
    if let Some(shared) = global["shared_sections"].as_object_mut() {
        crate::report_sections::project_shared_sections(shared)?;
    }
    let object = global.as_object_mut().expect("global document object");
    for (key, html_key) in [
        ("relative_levels", "relative_levels_html"),
        ("modes", "modes_html"),
    ] {
        let section = object.shift_remove(key).unwrap_or_default();
        object.insert(
            html_key.into(),
            json!(crate::report_document::section_fragment(section)?),
        );
    }
    Ok(global)
}

pub(crate) fn build_global_document_sections(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
) -> Result<Value, String> {
    prepared.validate()?;
    options.validate()?;
    let graph = &prepared.graph;
    let mut groups = Map::new();
    for role in [
        "why",
        "overview",
        "speakers",
        "timing",
        "symmetric",
        "time_domain",
        "epa",
    ] {
        groups.insert(role.to_string(), Value::Array(Vec::new()));
    }
    let mut preparation_status = Vec::new();
    let shared = crate::report_sections::build_shared_document_sections(graph);
    let mut status_sections = Vec::new();

    let binding = crate::report_dataset::binding_evidence(prepared, options)?;
    let playback = crate::evidence_sections::playback_notices(&json!({
        "metadata": graph.get("metadata").unwrap_or(&Value::Null),
        "binding_verified": binding["verified"],
        "binding_reason": binding["reason"],
    }));
    status_sections.push(json!({"kind":"playback_notices","notices":playback}));
    if options.payload_binding.is_none() {
        preparation_status.push(json!({
            "section":"playback_status",
            "status":"unavailable",
            "reason":"canonical payload-binding evidence was not supplied"
        }));
    }
    if prepared.target_snapshot["state"] != "available" {
        preparation_status.push(json!({
            "section":"target_overlay",
            "status":prepared.target_snapshot["state"],
            "reason":prepared.target_snapshot["error"].as_str().unwrap_or("no available target snapshot was supplied")
        }));
    }

    let global_evidence = crate::evidence_sections::global_document_evidence(graph, "", &binding);
    push_figure(
        &mut groups,
        "why",
        global_evidence["correction_explanation_html"].clone(),
    );
    push_figure(
        &mut groups,
        "why",
        global_evidence["acceptance_views_html"].clone(),
    );
    push_figure(
        &mut groups,
        "why",
        global_evidence["waveform_status_html"].clone(),
    );
    push_figure(
        &mut groups,
        "time_domain",
        global_evidence["capture_reflections_html"].clone(),
    );
    push_figure(
        &mut groups,
        "time_domain",
        global_evidence["capture_clock_qa_html"].clone(),
    );

    if let Some(mixed_phase) =
        crate::evidence_sections::mixed_phase_block(&graph["metadata"], &graph["channels"])
    {
        push_figure(
            &mut groups,
            "why",
            crate::report_document::diagnostic_section(vec![mixed_phase]),
        );
    }
    push_figure(
        &mut groups,
        "overview",
        crate::report_document::diagnostic_section(vec![crate::acoustic_summary::results_summary(
            graph,
        )]),
    );
    let combined = combined_overview(prepared, options, &mut preparation_status)?;
    for figure in combined.as_array().into_iter().flatten() {
        push_figure(&mut groups, "overview", figure.clone());
    }
    push_figure(&mut groups, "overview", shared["gain_summary_html"].clone());
    push_figure(
        &mut groups,
        "overview",
        shared["crossover_config_html"].clone(),
    );
    let bass = graph["metadata"]["bass_management"]
        .as_object()
        .is_some_and(|value| !value.is_empty());
    if bass {
        for key in [
            "bass_management_summary_html",
            "bass_management_groups_table_html",
            "bass_management_sub_outputs_table_html",
        ] {
            if let Some(section) = shared.get(key) {
                push_figure(&mut groups, "overview", section.clone());
            }
        }
        for builder in ["routing", "headroom"] {
            let figure = match builder {
                "routing" => crate::bass_figures::routing_figure(graph, None),
                _ => crate::bass_figures::headroom_figure(graph, None),
            };
            if let Some(figure) = figure {
                push_figure(&mut groups, "overview", figure);
            }
        }
    }

    let levels = crate::acoustic_curves::level_compensation(&graph["channels"]);
    let relative_levels = crate::acoustic_curves::level_table(&levels)
        .map(|table| crate::report_document::diagnostic_section(vec![table]));
    let (t60, room_table) = crate::acoustic_t60::t60_document_data(graph);
    if let Some(rows) = t60
        .get("room_rows")
        .filter(|value| value.is_array() && !value.as_array().unwrap().is_empty())
    {
        if let Some(table) = room_table {
            push_figure(
                &mut groups,
                "time_domain",
                crate::report_document::diagnostic_section(vec![table]),
            );
        }
        let figure = crate::temporal_figures::build_temporal_figure(
            "create_t60_octaves_figure",
            &json!({
                "label":"Room mean",
                "rows": rows,
                "itu_reference": t60.get("reference").unwrap_or(&Value::Null),
            }),
        )?;
        push_figure(&mut groups, "time_domain", figure);
        let counts = rows
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|row| {
                Some(format!(
                    "{} Hz: {}",
                    row["centre_hz"].as_u64()?,
                    row["speaker_count"].as_u64()?
                ))
            })
            .collect::<Vec<_>>()
            .join(", ");
        push_figure(
            &mut groups,
            "time_domain",
            crate::report_document::caption_section(
                &format!("Room T60 contributing speakers by octave: {counts}"),
                false,
            ),
        );
        push_figure(
            &mut groups,
            "time_domain",
            crate::report_document::diagnostic_section(vec![crate::acoustic_t60::t60_reference(
                &t60["reference"],
            )]),
        );
    }

    let tof = shared
        .get("time_of_flight_html")
        .cloned()
        .unwrap_or(Value::Null);
    push_figure(&mut groups, "timing", tof);
    // Reuse the report's stable channel-tab order so global and channel views
    // traverse the same prepared entries. This deliberately avoids relying on
    // JSON object insertion order, which is not retained by PreparedReportData.
    let mut channel_names = Vec::new();
    for entry in crate::report_data::display_channel_entries(graph)
        .as_array()
        .into_iter()
        .flatten()
    {
        let Some(name) = entry["channel"].as_str() else {
            continue;
        };
        if !channel_names.iter().any(|existing| existing == name) {
            channel_names.push(name.to_owned());
        }
    }
    let pair_groups = crate::symmetric_pairs::symmetric_groups(&channel_names);
    let pairs = &pair_groups[0];
    let unpaired = &pair_groups[1];
    if !pairs.as_array().is_none_or(Vec::is_empty) || !unpaired.as_array().is_none_or(Vec::is_empty)
    {
        let mut summary = vec!["Rust aligns final curves within their shared frequency range. The absolute sum assumes equal phase; it is an upper bound, not a prediction of interference from simultaneous playback. The complex sum is the coherent pressure sum from measured phase; the sum difference between them is the cancellation loss.".to_owned()];
        if let Some(unpaired) = unpaired.as_array().filter(|channels| !channels.is_empty()) {
            let names = unpaired
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            summary.push(format!("Unpaired channels (not summed): {names}"));
        }
        push_figure(
            &mut groups,
            "symmetric",
            crate::report_document::paragraph_section(&summary),
        );
        for pair in pairs.as_array().into_iter().flatten() {
            let Some(label) = pair[0].as_str() else {
                continue;
            };
            let Some(combo) = prepared
                .symmetric_pairs
                .as_ref()
                .and_then(|value| value.get(label))
            else {
                let mut note = crate::report_document::paragraph_section(&[format!(
                    "Symmetric pair {label}: no Rust pair export in this output bundle. Regenerate with the updated RoomEQ exporter."
                )]);
                note["tab"] = json!(label);
                push_figure(&mut groups, "symmetric", note);
                continue;
            };
            let members = pair[1]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            let complex_sum = combo
                .get("complex_spl")
                .filter(|value| !value.is_null())
                .cloned();
            let figure = crate::temporal_figures::build_temporal_figure(
                "create_symmetric_pair_figure",
                &json!({
                    "label":label,
                    "freq":combo["freq"],
                    "abs_spl":combo["sum_spl"],
                    "complex_spl":complex_sum,
                    "tab":label,
                }),
            )?;
            if figure.is_null() {
                preparation_status.push(json!({
                    "section":"symmetric_pair",
                    "label":label,
                    "status":"unavailable",
                    "reason":"the saved pair export lacks a finite aligned magnitude sum"
                }));
            } else {
                push_figure(&mut groups, "symmetric", figure);
            }
            if complex_sum.is_none() {
                let names = if members.len() == 2 {
                    format!("{} and {}", members[0], members[1])
                } else {
                    "both paired channels".to_owned()
                };
                let mut note = crate::report_document::caption_section(
                    format!(
                        "Symmetric pair {label}: complex sum unavailable without finite measured phase on both {names} final curves; showing the absolute (magnitude) sum only."
                    ),
                    false,
                );
                note["tab"] = json!(label);
                push_figure(&mut groups, "symmetric", note);
                preparation_status.push(json!({
                    "section":"symmetric_complex_sum",
                    "label":label,
                    "status":"unavailable",
                    "reason":format!("prepared measured-phase sum is unavailable for {names}")
                }));
            }
        }
    }

    let flow = shared
        .get("signal_flow_sections")
        .cloned()
        .unwrap_or_else(|| json!([]));
    groups.insert("flow".to_string(), flow);

    let mut capture_view = Value::Null;
    if let Some(capture) = &options.capture_verification {
        if binding["verified"] != true {
            let reason = binding["reason"]
                .as_str()
                .unwrap_or("payload binding is unavailable");
            capture_view = unavailable_capture_view(format!(
                "saved optimization graph binding is invalid: {reason}"
            ));
            preparation_status.push(json!({
                "section":"capture_views",
                "status":"unavailable",
                "reason":"capture verification cannot be displayed without a verified frozen payload binding"
            }));
        } else if capture["graph_id"] != binding["graph_identity"] {
            capture_view = unavailable_capture_view(
                "verification candidate graph does not match the saved optimization graph".into(),
            );
            preparation_status.push(json!({
                "section":"capture_views",
                "status":"unavailable",
                "reason":"verification candidate graph does not match the verified saved optimization graph"
            }));
        } else if let Some(view_bindings) = &options.capture_view_bindings {
            let (section, rendered_available) =
                crate::evidence_sections::capture_views_section(capture, view_bindings);
            let bindings_verified = capture_view_bindings_verified(capture, view_bindings);
            capture_view = section;
            preparation_status.push(json!({
                "section":"capture_views",
                "status":if bindings_verified && rendered_available { "available" } else { "unavailable" },
                "reason":if bindings_verified && rendered_available {
                    "capture report and per-view payload bindings match the saved optimization graph"
                } else {
                    "one or more capture views lack valid matching payload bindings or renderable evidence"
                }
            }));
        } else {
            capture_view = unavailable_capture_view(
                "capture-view payload verification is unavailable.".into(),
            );
            preparation_status.push(json!({
                "section":"capture_views",
                "status":"unavailable",
                "reason":"capture-view payload verification is unavailable"
            }));
        }
    } else {
        preparation_status.push(json!({
            "section":"capture_acceptance",
            "status":"unavailable",
            "reason":"capture verification evidence was not supplied"
        }));
    }
    push_figure(&mut groups, "why", capture_view);
    let modes = global_evidence
        .get("modes_summary_html")
        .cloned()
        .unwrap_or_default();
    if graph["channels"].as_object().is_some_and(|channels| {
        channels.values().any(|channel| {
            channel.get("waterfall").is_some()
                || channel.get("resonance_decays").is_some()
                || channel["drivers"].as_array().is_some_and(|drivers| {
                    drivers.iter().any(|driver| {
                        let acoustic = &driver["measured_acoustics"];
                        acoustic.get("waterfall").is_some()
                            || acoustic.get("resonance_decays").is_some()
                    })
                })
        })
    }) {
        let rendered = !modes.is_null();
        preparation_status.push(json!({
            "section":"modes_summary",
            "status":if rendered { "available" } else { "unavailable" },
            "reason":if rendered {
                "saved waterfall/resonance data was rendered by the shared report owner"
            } else {
                "saved waterfall/resonance fields did not produce a valid resonance summary"
            }
        }));
    }
    let captured_arrival_evidence = graph["metadata"]["effective_config"]["speakers"]
        .as_object()
        .is_some_and(|speakers| {
            speakers
                .values()
                .any(|speaker| speaker["provenance"]["capture"]["reflection_report"].is_object())
        });
    if captured_arrival_evidence {
        let rendered = !global_evidence["capture_reflections_html"].is_null();
        preparation_status.push(json!({
            "section":"capture_reflections",
            "status":if rendered { "available" } else { "unavailable" },
            "reason":if rendered {
                "serialized captured-arrival evidence was rendered by the shared report owner"
            } else {
                "captured-arrival evidence did not produce a report section"
            }
        }));
    }

    Ok(json!({
        "title":"Room EQ",
        "status_sections":status_sections,
        "global_groups":groups,
        "relative_levels":relative_levels,
        "modes":modes,
        "preparation_status":preparation_status,
        "shared_sections":shared,
        "provenance":crate::report_provenance::report_provenance(
            options.header_provenance.as_ref().or(prepared.provenance.as_ref()).unwrap_or(&Value::Null)
        ),
    }))
}

fn capture_view_bindings_verified(report: &Value, verification: &Value) -> bool {
    let (Some(graph), Some(comparisons), Some(bindings)) = (
        report["graph_id"].as_str(),
        report["comparisons"].as_array(),
        verification["comparisons"].as_array(),
    ) else {
        return false;
    };
    if graph.is_empty()
        || verification["graph_id"] != graph
        || comparisons.is_empty()
        || bindings.len() != comparisons.len()
    {
        return false;
    }
    comparisons
        .iter()
        .zip(bindings)
        .enumerate()
        .all(|(index, (comparison, binding))| {
            let payload_sha256 = binding["payload_sha256"].as_str();
            binding["index"].as_u64() == Some(index as u64)
                && binding["verified"] == true
                && binding["source"] == comparison["source"]
                && binding["seat"] == comparison["seat"]
                && payload_sha256.is_some_and(|digest| {
                    !digest.is_empty() && comparison["capture_views_binding"]["sha256"] == digest
                })
                && comparison["capture_views_binding"]["algorithm"] == "sha256-json-typed-v1"
                && comparison["capture_views_binding"]["graph_identity"] == graph
                && comparison["capture_views"]["candidate_graph_id"] == graph
                && comparison["capture_views"]["source"] == comparison["source"]
                && comparison["capture_views"]["seat"] == comparison["seat"]
        })
}

fn unavailable_capture_view(reason: String) -> Value {
    use crate::report_document::{
        DiagnosticBlock, DiagnosticIntroduction, DiagnosticSectionStyle, diagnostic_section,
    };
    let mut section = diagnostic_section(vec![DiagnosticBlock::Paragraph(format!(
        "Unavailable: {reason}"
    ))]);
    section["introduction"] = json!(DiagnosticIntroduction {
        title: "Capture acceptance diagnostics".into(),
        text: String::new(),
        style: DiagnosticSectionStyle::CaptureViewsUnavailable,
    });
    section
}

fn push_figure(groups: &mut Map<String, Value>, role: &str, value: Value) {
    if !value.is_null() {
        groups
            .entry(role.to_string())
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("global group is initialized as an array")
            .push(value);
    }
}

fn combined_overview(
    prepared: &PreparedReportData,
    options: &ReportDatasetOptions,
    status: &mut Vec<Value>,
) -> Result<Value, String> {
    let graph = &prepared.graph;
    let channels = graph.get("channels").and_then(Value::as_object);
    let Some(channels) = channels else {
        return Ok(json!([]));
    };
    let entries = crate::report_data::driver_display_entries(graph);
    let mut names: Vec<_> = channels.keys().collect();
    names.sort_by(|left, right| {
        crate::report_helpers::channel_sort_key(left)
            .cmp(&crate::report_helpers::channel_sort_key(right))
    });
    let mut before_series = Vec::new();
    let mut before_curves = Vec::new();
    for (color_index, channel_name) in names.iter().enumerate() {
        let channel_entries = entries
            .as_array()
            .into_iter()
            .flatten()
            .filter(|entry| entry["channel"] == **channel_name && !entry["driver"].is_null());
        let mut expanded = false;
        for entry in channel_entries {
            let Some(position) = entry["source_driver_position"].as_u64().map(|v| v as usize)
            else {
                continue;
            };
            let driver = &channels[*channel_name]["drivers"][position];
            let curve =
                crate::report_data::clip_curve_to_measured_band(&driver["initial_curve"], driver);
            if valid_curve(&curve) {
                expanded = true;
                let curve = smooth_display_curve(&curve, options.smoothed_octaves);
                before_curves.push(curve.clone());
                before_series.push(series(
                    &format!(
                        "{}/{} (before)",
                        channel_name,
                        entry["source_driver_name"]
                            .as_str()
                            .or_else(|| entry["label"].as_str())
                            .unwrap_or_default()
                    ),
                    &curve,
                    color_index,
                ));
            }
        }
        if !expanded {
            let curve = &channels[*channel_name]["initial_curve"];
            if valid_curve(curve) {
                let curve = smooth_display_curve(curve, options.smoothed_octaves);
                before_curves.push(curve.clone());
                before_series.push(series(
                    &format!("{channel_name} (before)"),
                    &curve,
                    color_index,
                ));
            }
        }
    }

    let mut routed = Map::new();
    if let Some(replay) = &prepared.replay {
        if let Some(sources) = replay
            .context
            .get("routed_sources")
            .and_then(Value::as_object)
        {
            for (name, state) in sources {
                if state["status"] == "available" && valid_curve(&state["curve"]) {
                    routed.insert(name.clone(), state["curve"].clone());
                } else {
                    status.push(json!({"section":"corrected_routed_response","channel":name,"status":"unavailable","reason":state["reason"].as_str().unwrap_or("routed replay is unavailable")}));
                }
            }
        } else {
            status.push(json!({"section":"corrected_routed_response","status":"unavailable","reason":"native replay contains no routed-source states"}));
        }
    } else if let Some(curves) = &options.legacy_source_curves {
        if let Some(object) = curves.as_object() {
            routed.extend(
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }
    } else {
        status.push(json!({"section":"corrected_routed_response","status":"unavailable","reason":"neither native routed replay nor explicitly prepared legacy source curves are available"}));
    }

    let target = (prepared.target_snapshot["state"] == "available")
        .then_some(&prepared.target_snapshot["target"]);
    status.extend(crate::target_overlay::unavailable_route_target_statuses(
        graph,
        target,
        options.route_crossover_db.as_ref(),
        options.route_crossover_status.as_ref(),
        Some(&Value::Object(routed.clone())),
    ));
    let target_curves = crate::target_overlay::build_target_overlay_curves_with_status(
        graph,
        &Value::Object(routed.clone()),
        target,
        options.route_crossover_db.as_ref(),
        options.route_crossover_status.as_ref(),
    );
    let mut corrected_series = Vec::new();
    let mut corrected_curves = Vec::new();
    for (color_index, channel_name) in names.iter().enumerate() {
        if let Some(curve) = routed.get(*channel_name).filter(|curve| valid_curve(curve)) {
            let curve = smooth_display_curve(curve, options.smoothed_octaves);
            corrected_curves.push(curve.clone());
            corrected_series.push(series(
                &format!("{channel_name} (corrected)"),
                &curve,
                color_index,
            ));
        }
        if let Some(curve) = target_curves
            .get(*channel_name)
            .filter(|curve| valid_curve(curve))
        {
            corrected_curves.push(curve.clone());
            corrected_series.push(json!({"name":format!("{channel_name} target"),"x":curve["freq"],"y":curve["spl"],"color_index":color_index,"dash":"dot","width":2.0}));
        }
    }

    let mut eq_series = Vec::new();
    let mut eq_levels = Vec::new();
    for (color_index, channel_name) in names.iter().enumerate() {
        let channel = &channels[*channel_name];
        // A native replay is authoritative for the realized unity-input
        // transfer. Saved graph EQ is only a legacy display source and can be
        // stale relative to the frozen plugin state.
        let eq = prepared.replay.as_ref().and_then(|replay| {
            replay
                .context
                .get("channels")?
                .get(*channel_name)
                .filter(|state| state["status"] == "available")
                .map(|state| &state["transfer_curve"])
        });
        let eq = if prepared.replay.is_some() {
            eq
        } else {
            Some(&channel["eq_response"])
        };
        if let Some(eq) = eq.filter(|curve| valid_curve(curve)) {
            add_eq_series(
                &mut eq_series,
                &mut eq_levels,
                &format!("EQ: {channel_name}"),
                eq,
                color_index,
            );
        }
        if let Some(prepared_eq) = options
            .driver_eq_curves
            .as_ref()
            .and_then(|curves| curves.get(*channel_name))
        {
            let rows: Vec<&Value> = prepared_eq
                .as_array()
                .map(|values| values.iter().collect())
                .unwrap_or_else(|| {
                    prepared_eq
                        .as_object()
                        .map(|values| values.values().collect())
                        .unwrap_or_default()
                });
            for row in rows {
                let curve = row.get("curve").unwrap_or(row);
                if valid_curve(curve) {
                    let label = row["name"].as_str().unwrap_or(channel_name);
                    add_eq_series(
                        &mut eq_series,
                        &mut eq_levels,
                        &format!("EQ: {label}"),
                        curve,
                        color_index,
                    );
                }
            }
        }
    }
    let (spl_min, spl_max) = crate::report_data::compute_y_range(&Value::Array(
        before_curves.into_iter().chain(corrected_curves).collect(),
    ));
    let (eq_min, eq_max) = if eq_levels.is_empty() {
        (-15.0, 15.0)
    } else {
        let max = eq_levels.iter().copied().reduce(f64::max).unwrap_or(0.0);
        let min = eq_levels.iter().copied().reduce(f64::min).unwrap_or(0.0);
        (
            ((min / 5.0).floor() * 5.0 - 5.0).max(-20.0),
            ((max / 5.0).ceil() * 5.0 + 5.0).min(20.0),
        )
    };
    let mut crossovers = names
        .iter()
        .flat_map(|name| crate::report_data::crossover_frequencies(&channels[*name]))
        .collect::<Vec<_>>();
    crossovers.sort_by(f64::total_cmp);
    crossovers.dedup_by(|left, right| left == right);
    let has_channels = !channels.is_empty();
    Ok(json!(crate::combined_figures::combined_figures(
        &json!({
            "has_channels":has_channels,
            "before_series":before_series,
            "eq_series":eq_series,
            "corrected_series":corrected_series,
            "eq_frequency_extent":[20.000000000000004_f64,20_000.000000000004_f64],
            "crossovers":crossovers,
            "spl_y_min":spl_min,
            "spl_y_max":spl_max,
            "eq_y_lower":eq_min,
            "eq_y_upper":eq_max,
        }),
        None
    )))
}

fn valid_curve(curve: &Value) -> bool {
    curve["freq"]
        .as_array()
        .is_some_and(|freq| !freq.is_empty())
        && curve["spl"].as_array().is_some_and(|spl| !spl.is_empty())
        && curve["freq"]
            .as_array()
            .zip(curve["spl"].as_array())
            .is_some_and(|(freq, spl)| freq.len() == spl.len())
}

fn smooth_display_curve(curve: &Value, octaves: f64) -> Value {
    crate::acoustic_curves::smooth_curve(curve, octaves).unwrap_or_else(|| curve.clone())
}

fn series(name: &str, curve: &Value, color_index: usize) -> Value {
    json!({"name":name,"x":curve["freq"],"y":curve["spl"],"color_index":color_index})
}

fn add_eq_series(
    traces: &mut Vec<Value>,
    levels: &mut Vec<f64>,
    name: &str,
    curve: &Value,
    color: usize,
) {
    levels.extend(
        curve["spl"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_f64)
            .filter(|value| value.is_finite()),
    );
    traces.push(series(name, curve, color));
}

#[cfg(test)]
mod tests {
    use super::{build_global_report_sections, capture_view_bindings_verified};
    use crate::PreparedReportData;
    use crate::report_dataset::ReportDatasetOptions;
    use serde_json::{Value, json};

    #[test]
    fn global_available_sections_match_pre_port_python_capture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/report_dataset_global_python.json"
        ))
        .expect("captured Python fixture parses");
        let case = &fixture[0];
        let prepared = PreparedReportData::new(
            case["input"].clone(),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .expect("fixture becomes a prepared report");
        let legacy_options: ReportDatasetOptions =
            serde_json::from_value(case["options"].clone()).expect("fixture options parse");
        let result = build_global_report_sections(&prepared, &legacy_options)
            .expect("global report composition succeeds");
        let mut mismatches = Vec::new();
        for key in [
            "title",
            "status_sections",
            "relative_levels_html",
            "modes_html",
            "provenance",
        ] {
            if result[key] != case["expected"][key] {
                mismatches.push(format!("field:{key}"));
            }
        }
        for role in [
            "why",
            "overview",
            "speakers",
            "timing",
            "symmetric",
            "time_domain",
            "epa",
            "flow",
        ] {
            if normalize_saved_blocks(&result["global_groups"][role])
                != normalize_saved_blocks(&case["expected"]["global_groups"][role])
            {
                mismatches.push(format!(
                    "group:{role} actual={:?} expected={:?} diffs={:?}",
                    section_signatures(&result["global_groups"][role]),
                    section_signatures(&case["expected"]["global_groups"][role]),
                    value_differences(
                        &normalize_saved_blocks(&result["global_groups"][role]),
                        &normalize_saved_blocks(&case["expected"]["global_groups"][role])
                    )
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "full global report mismatches: {mismatches:?}"
        );
        assert!(
            result["preparation_status"]
                .as_array()
                .is_some_and(|items| !items.is_empty())
        );
    }

    #[test]
    fn saved_absolute_pair_sum_reaches_symmetric_figure_builder() {
        let prepared = PreparedReportData::new(
            json!({"channels":{"L":{},"R":{}},"metadata":{}}),
            None,
            None,
            Some(
                json!({"L+R":{"freq":[100.0,1000.0],"sum_spl":[86.0,87.0],"diff_spl":[null,null]}}),
            ),
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();

        let result =
            build_global_report_sections(&prepared, &ReportDatasetOptions::default()).unwrap();
        let sections = result["global_groups"]["symmetric"].as_array().unwrap();
        assert!(sections.iter().any(|section| {
            section["kind"] == "figure"
                && section["figure"]["title"].as_str().is_some_and(|title| {
                    title.starts_with("Symmetric pair: L+R (absolute sum only")
                })
                && section["figure"]["series"][0]["y"] == json!([86.0, 87.0])
        }));
        assert!(sections.iter().any(|section| {
            section["kind"] == "html"
                && section["html"].as_str().is_some_and(|html| {
                    html.contains("complex sum unavailable without finite measured phase")
                })
        }));
    }

    #[test]
    fn prepared_measured_phase_sum_reaches_the_symmetric_figure() {
        let prepared = PreparedReportData::new(
            json!({
                "channels":{
                    "L":{"final_curve":{"freq":[100.0,1000.0],"spl":[80.0,80.0],"phase":[0.0,0.0]}},
                    "R":{"final_curve":{"freq":[100.0,1000.0],"spl":[80.0,80.0],"phase":[0.0,0.0]}}
                },
                "metadata":{}
            }),
            None,
            None,
            Some(json!({"L+R":{"freq":[100.0,1000.0],"sum_spl":[86.0,86.0],"diff_spl":[null,null],"complex_spl":[86.02059991327963,86.02059991327963]}})),
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();

        let result =
            build_global_report_sections(&prepared, &ReportDatasetOptions::default()).unwrap();
        let figure = result["global_groups"]["symmetric"]
            .as_array()
            .unwrap()
            .iter()
            .find(|section| section["kind"] == "figure")
            .unwrap();
        let series = figure["figure"]["series"].as_array().unwrap();
        assert_eq!(series.len(), 3);
        assert!(series[1]["name"] == "L+R complex sum");
        assert_eq!(
            series[1]["y"],
            json!([86.02059991327963, 86.02059991327963])
        );
        assert!(
            !result["preparation_status"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| { entry["section"] == "symmetric_complex_sum" })
        );
    }

    #[test]
    fn native_replay_is_authoritative_and_binding_options_fail_closed() {
        let prepared = PreparedReportData::new(
            json!({"channels":{"L":{"initial_curve":{"freq":[20.0],"spl":[70.0]},"final_curve":{"freq":[20.0],"spl":[99.0]}}}}),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            Some(crate::PreparedReplay {
                sample_rate_hz: Some(48_000),
                context: serde_json::Map::from_iter([("routed_sources".into(), json!({
                    "L":{"status":"unavailable","reason":"required phase missing"}
                }))]),
            }),
        ).unwrap();
        let options = ReportDatasetOptions {
            payload_binding: Some(json!({"verified":false,"reason":"hash mismatch"})),
            ..ReportDatasetOptions::default()
        };
        let output = build_global_report_sections(&prepared, &options).unwrap();
        assert_eq!(output["status_sections"][0]["kind"], "html");
        assert!(
            output["preparation_status"]
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| {
                    entry["section"] == "corrected_routed_response"
                        && entry["reason"] == "required phase missing"
                })
        );
        assert!(
            output["global_groups"]["overview"]
                .as_array()
                .unwrap()
                .iter()
                .all(|section| {
                    !section["figure"]["series"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|series| series["y"] == json!([99.0]))
                })
        );
    }

    #[test]
    fn loaded_target_reports_unavailable_for_unshaped_lfe_route() {
        let graph = json!({
            "channels":{"LFE":{"initial_curve":{"freq":[40.0,80.0,160.0],"spl":[70.0,70.0,70.0]}}},
            "metadata":{"bass_management":{"routing_graph":{"routes":[{
                "source_channel":"LFE","route_kind":"lfe_lowpass_to_sub","low_pass_hz":80.0
            }]}}}
        });
        let prepared = PreparedReportData::new(
            graph,
            None,
            None,
            None,
            json!({"schema_version":1,"state":"available","target":{"freq":[20.0,20000.0],"spl":[75.0,75.0]}}),
            None,
        )
        .unwrap();
        let options = ReportDatasetOptions {
            route_crossover_db: Some(json!({})),
            route_crossover_status: Some(json!({
                "LFE":{"status":"unavailable","reason":"an explicit playback sample rate is required"}
            })),
            legacy_source_curves: Some(json!({
                "LFE":{"freq":[40.0,80.0,160.0],"spl":[70.0,70.0,70.0]}
            })),
            ..ReportDatasetOptions::default()
        };
        let result = build_global_report_sections(&prepared, &options).unwrap();
        assert!(
            result["preparation_status"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| {
                    item["section"] == "target_overlay"
                        && item["channel"] == "LFE"
                        && item["status"] == "unavailable"
                        && item["reason"]
                            .as_str()
                            .unwrap()
                            .contains("playback sample rate")
                })
        );
        assert!(
            result["global_groups"]["overview"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|section| section["figure"]["series"].as_array().into_iter().flatten())
                .all(|series| series["name"] != "LFE target")
        );
    }

    #[test]
    fn combined_overview_uses_native_eq_transfer_instead_of_stale_saved_curve() {
        let prepared = PreparedReportData::new(
            json!({"channels":{"L":{
                "initial_curve":{"freq":[20.0, 100.0],"spl":[70.0, 70.0]},
                "eq_response":{"freq":[20.0, 100.0],"spl":[99.0, 99.0]}
            }}}),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            Some(crate::PreparedReplay {
                sample_rate_hz: Some(48_000),
                context: serde_json::Map::from_iter([(
                    "channels".into(),
                    json!({
                        "L":{"status":"available","transfer_curve":{
                            "freq":[20.0, 100.0],"spl":[-5.0, -5.0]
                        }}
                    }),
                )]),
            }),
        )
        .unwrap();
        let output =
            build_global_report_sections(&prepared, &ReportDatasetOptions::default()).unwrap();
        let eq_series = output["global_groups"]["overview"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|section| section["figure"]["series"].as_array().into_iter().flatten())
            .filter(|series| {
                series["name"]
                    .as_str()
                    .is_some_and(|name| name.starts_with("EQ:"))
            })
            .collect::<Vec<_>>();
        assert_eq!(eq_series.len(), 1);
        assert_eq!(eq_series[0]["y"], json!([-5.0, -5.0]));
        assert!(
            !eq_series
                .iter()
                .any(|series| series["y"] == json!([99.0, 99.0]))
        );
    }

    #[test]
    fn bound_capture_comparisons_are_composed_as_native_sections() {
        use crate::report_document::{DiagnosticBlock, ReportSection};

        let fixture: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/capture_views_nested_python.json"
        ))
        .unwrap();
        let prepared = PreparedReportData::new(
            json!({"channels":{}}),
            Some(json!({"slim_graph_sha256":"loaded-digest"})),
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();
        let options = ReportDatasetOptions {
            payload_binding: Some(json!({
                "verified":true,
                "reason":"hash verified",
                "slim_graph_sha256":"loaded-digest",
                "graph_identity":fixture["report"]["graph_id"],
                "canonical_payload_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            })),
            capture_verification: Some(fixture["report"].clone()),
            capture_view_bindings: Some(fixture["verification"].clone()),
            ..ReportDatasetOptions::default()
        };
        let document = crate::report_dataset_payload::build_report_document(&prepared, &options)
            .expect("bound capture fixture composes");
        assert!(document.sections.iter().all(|section| {
            !matches!(section, ReportSection::Legacy(value) if value["kind"] == "html")
        }));
        assert!(document.sections.iter().any(|section| {
            matches!(section, ReportSection::Diagnostics(diagnostics)
                if diagnostics.blocks.iter().any(|block| matches!(block, DiagnosticBlock::CaptureChart(_))))
        }));
        assert!(
            document
                .preparation_status
                .as_array()
                .unwrap()
                .iter()
                .any(|entry| {
                    entry["section"] == "capture_views" && entry["status"] == "available"
                })
        );
    }

    #[test]
    fn verified_payload_binding_requires_matching_graph_and_payload_digests() {
        let prepared = PreparedReportData::new(
            json!({"channels":{}}),
            Some(json!({"slim_graph_sha256":"loaded-digest"})),
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();
        let options = |graph_digest: &str, payload_digest: &str| ReportDatasetOptions {
            payload_binding: Some(json!({
                "verified":true,
                "reason":"hash verified",
                "slim_graph_sha256":graph_digest,
                "graph_identity":"stable-graph-id",
                "canonical_payload_sha256":payload_digest
            })),
            ..ReportDatasetOptions::default()
        };
        let valid_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let matching =
            build_global_report_sections(&prepared, &options("loaded-digest", valid_sha)).unwrap();
        assert!(
            matching["status_sections"][0]["html"]
                .as_str()
                .unwrap()
                .contains("Delivered payload: verified")
        );
        let wrong_graph =
            build_global_report_sections(&prepared, &options("other-digest", valid_sha)).unwrap();
        assert!(
            wrong_graph["status_sections"][0]["html"]
                .as_str()
                .unwrap()
                .contains("Delivered payload: not verified")
        );
        let bad_payload =
            build_global_report_sections(&prepared, &options("loaded-digest", "not-a-sha256"))
                .unwrap();
        assert!(
            bad_payload["status_sections"][0]["html"]
                .as_str()
                .unwrap()
                .contains("Delivered payload: not verified")
        );
    }

    #[test]
    fn capture_view_status_requires_every_matching_verified_entry() {
        let report = json!({
            "graph_id":"graph-1",
            "comparisons":[{
                "source":"L", "seat":"seat-1",
                "capture_views_binding":{"algorithm":"sha256-json-typed-v1","graph_identity":"graph-1","sha256":"abc"},
                "capture_views":{"candidate_graph_id":"graph-1","source":"L","seat":"seat-1"}
            }]
        });
        let valid = json!({"graph_id":"graph-1","comparisons":[{
            "index":0,"verified":true,"source":"L","seat":"seat-1","payload_sha256":"abc"
        }]});
        assert!(capture_view_bindings_verified(&report, &valid));

        let unverified = json!({"graph_id":"graph-1","comparisons":[{
            "index":0,"verified":false,"source":"L","seat":"seat-1","payload_sha256":"abc"
        }]});
        assert!(!capture_view_bindings_verified(&report, &unverified));
        let wrong_digest = json!({"graph_id":"graph-1","comparisons":[{
            "index":0,"verified":true,"source":"L","seat":"seat-1","payload_sha256":"changed"
        }]});
        assert!(!capture_view_bindings_verified(&report, &wrong_digest));
    }

    fn normalize_saved_blocks(value: &Value) -> Value {
        let Some(sections) = value.as_array() else {
            return value.clone();
        };
        Value::Array(
            sections
                .iter()
                .map(|section| {
                    let Some(html) = section["html"].as_str() else {
                        return section.clone();
                    };
                    let mut rest = html.to_string();
                    let mut params = Vec::new();
                    while let Some(start) = rest.find("<pre>") {
                        let content_start = start + "<pre>".len();
                        let Some(relative_end) = rest[content_start..].find("</pre>") else {
                            break;
                        };
                        let end = content_start + relative_end;
                        let decoded = rest[content_start..end]
                            .replace("&quot;", "\"")
                            .replace("&lt;", "<")
                            .replace("&gt;", ">")
                            .replace("&amp;", "&");
                        params.push(
                            serde_json::from_str::<Value>(&decoded)
                                .unwrap_or(Value::String(decoded)),
                        );
                        rest.replace_range(start..end + "</pre>".len(), "<saved-parameters/>");
                    }
                    let rest = rest.replace("<saved-parameters/>", "<pre></pre>");
                    let mut normalized = section.clone();
                    normalized["html"] = json!(rest);
                    normalized["saved_parameter_blocks"] = json!(params);
                    normalized
                })
                .collect(),
        )
    }

    fn section_signatures(value: &Value) -> Vec<String> {
        value
            .as_array()
            .into_iter()
            .flatten()
            .map(|section| {
                if let Some(title) = section["figure"]["title"].as_str() {
                    return format!("figure:{title}");
                }
                if let Some(title) = section["chart"]["title"].as_str() {
                    return format!("chart:{title}");
                }
                if let Some(html) = section["html"].as_str() {
                    let snippet = html.chars().take(96).collect::<String>();
                    return format!("html:{snippet}");
                }
                format!("kind:{}", section["kind"])
            })
            .collect()
    }

    fn value_differences(actual: &Value, expected: &Value) -> Vec<String> {
        fn visit(actual: &Value, expected: &Value, path: &str, output: &mut Vec<String>) {
            if output.len() >= 12 || actual == expected {
                return;
            }
            match (actual, expected) {
                (Value::Object(actual), Value::Object(expected)) => {
                    let keys: std::collections::BTreeSet<_> = actual
                        .keys()
                        .map(String::as_str)
                        .chain(expected.keys().map(String::as_str))
                        .collect();
                    for key in keys {
                        let child = format!("{path}/{key}");
                        match (actual.get(key), expected.get(key)) {
                            (Some(actual), Some(expected)) => {
                                visit(actual, expected, &child, output)
                            }
                            _ => output.push(format!("{child}: missing field")),
                        }
                    }
                }
                (Value::Array(actual), Value::Array(expected)) => {
                    if actual.len() != expected.len() {
                        output.push(format!(
                            "{path}: lengths {} != {}",
                            actual.len(),
                            expected.len()
                        ));
                    }
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        visit(actual, expected, &format!("{path}/{index}"), output);
                    }
                }
                (Value::String(actual), Value::String(expected)) => {
                    let actual_chars: Vec<_> = actual.chars().collect();
                    let expected_chars: Vec<_> = expected.chars().collect();
                    let offset = actual_chars
                        .iter()
                        .zip(&expected_chars)
                        .take_while(|(left, right)| left == right)
                        .count();
                    output.push(format!(
                        "{path}: strings differ at char {offset} (lengths {} != {}): {:?} != {:?}",
                        actual_chars.len(),
                        expected_chars.len(),
                        actual_chars
                            .iter()
                            .skip(offset)
                            .take(24)
                            .copied()
                            .collect::<String>(),
                        expected_chars
                            .iter()
                            .skip(offset)
                            .take(24)
                            .copied()
                            .collect::<String>(),
                    ));
                }
                _ => output.push(format!("{path}: {actual} != {expected}")),
            }
        }

        let mut output = Vec::new();
        visit(actual, expected, "$", &mut output);
        output
    }
}
