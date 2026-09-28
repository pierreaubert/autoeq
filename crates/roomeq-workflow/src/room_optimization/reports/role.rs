#![allow(dead_code)]
use super::super::*;
use super::misc::bass_consistency_rms_db;
use super::misc::dialog_band_roughness_rms_db;
use super::misc::group_mean_deviation_rms_db;
use super::misc::headroom_peak_boost_db;
use super::misc::max_optional;
use super::misc::mean;
use super::types::RoleChannelMatchingGroup;
use super::types::channel_matching_role_key;

pub(in super::super) fn update_perceptual_metrics(
    metadata: &mut OptimizationMetadata,
    channels: Option<&HashMap<String, ChannelDspChain>>,
    config: Option<&RoomConfig>,
) {
    let Some(epa_per_channel) = metadata.epa_per_channel.as_ref() else {
        metadata.perceptual_metrics = None;
        return;
    };
    if epa_per_channel.is_empty() {
        metadata.perceptual_metrics = None;
        return;
    }

    // Canonical order: HashMap iteration is per-process random, and even
    // summation order changes the last ulp. Sort once for every reduction
    // below so reports and hashes are deterministic across runs.
    let mut epa_names: Vec<&String> = epa_per_channel.keys().collect();
    epa_names.sort();
    let count = epa_per_channel.len() as f64;
    let epa_preference_pre = epa_names
        .iter()
        .map(|name| epa_per_channel[*name].pre.preference)
        .sum::<f64>()
        / count;
    let epa_preference_post = epa_names
        .iter()
        .map(|name| epa_per_channel[*name].post.preference)
        .sum::<f64>()
        / count;
    let channel_matching_midrange_rms_db = metadata
        .inter_channel_deviation
        .as_ref()
        .map(|icd| icd.midrange_rms_db);
    let role_channel_matching_rms_db = channels.and_then(role_channel_matching_rms_db);
    let bass_consistency_rms_db = channels.and_then(bass_consistency_rms_db);
    let dialog_band_roughness_rms_db = channels.and_then(dialog_band_roughness_rms_db);
    let headroom_peak_boost_db = channels.and_then(headroom_peak_boost_db);
    let sorted_chains: Option<Vec<&ChannelDspChain>> = channels.map(|channels| {
        let mut names: Vec<&String> = channels.keys().collect();
        names.sort();
        names.into_iter().map(|name| &channels[name]).collect()
    });
    let fir_pre_ringing_audible_db = sorted_chains.as_deref().and_then(|chains| {
        max_optional(chains.iter().filter_map(|chain| {
            chain
                .fir_temporal_masking
                .as_ref()
                .map(|m| m.pre_ringing_audible_db)
        }))
    });
    let fir_post_ringing_audible_db = sorted_chains.as_deref().and_then(|chains| {
        max_optional(chains.iter().filter_map(|chain| {
            chain
                .fir_temporal_masking
                .as_ref()
                .map(|m| m.post_ringing_audible_db)
        }))
    });
    let fir_temporal_masking_penalty = sorted_chains.as_deref().and_then(|chains| {
        max_optional(
            chains
                .iter()
                .filter_map(|chain| chain.fir_temporal_masking.as_ref().map(|m| m.penalty)),
        )
    });
    let direct_plus_early_correction_energy_db = sorted_chains.as_deref().and_then(|chains| {
        max_optional(chains.iter().filter_map(|chain| {
            chain
                .direct_early_late_correction
                .as_ref()
                .map(|m| m.direct_plus_early_energy_db)
        }))
    });
    let early_cue_advisory = sorted_chains.as_deref().and_then(|chains| {
        chains
            .iter()
            .filter_map(|chain| chain.direct_early_late_correction.as_ref())
            .find(|metrics| metrics.advisory != "ok")
            .map(|metrics| metrics.advisory.clone())
    });
    let headroom_risk = headroom_peak_boost_db.map(|peak_boost| {
        let margin_db = config
            .and_then(|cfg| cfg.system.as_ref())
            .and_then(|system| system.bass_management.as_ref())
            .map(|bm| bm.headroom_margin_db)
            .unwrap_or(6.0);
        if peak_boost > margin_db {
            "high_boost_exceeds_headroom_margin".to_string()
        } else if peak_boost > margin_db * 0.5 {
            "moderate_boost_uses_headroom".to_string()
        } else {
            "ok".to_string()
        }
    });
    let timing_confidence = metadata.group_delay.as_ref().map(|gd| {
        if gd.applied {
            "gd_applied".to_string()
        } else if gd.advisory == "success" {
            "gd_success_not_applied".to_string()
        } else {
            format!("gd_{}", gd.advisory)
        }
    });

    metadata.perceptual_metrics = Some(PerceptualMetrics {
        epa_preference_pre,
        epa_preference_post,
        epa_preference_delta: epa_preference_post - epa_preference_pre,
        channel_matching_midrange_rms_db,
        role_channel_matching_rms_db,
        bass_consistency_rms_db,
        dialog_band_roughness_rms_db,
        headroom_peak_boost_db,
        headroom_risk,
        timing_confidence,
        fir_pre_ringing_audible_db,
        fir_post_ringing_audible_db,
        fir_temporal_masking_penalty,
        direct_plus_early_correction_energy_db,
        early_cue_advisory,
    });
}

pub(in super::super) fn role_channel_matching_rms_db(
    channels: &HashMap<String, ChannelDspChain>,
) -> Option<f64> {
    // Canonical order: both the channel iteration (group membership
    // order feeds RMS summation) and the group iteration (feeds the
    // final mean) must be deterministic across runs.
    let mut names: Vec<&String> = channels.keys().collect();
    names.sort();
    let mut grouped: HashMap<&'static str, Vec<&ChannelDspChain>> = HashMap::new();
    for name in names {
        if let Some(key) = channel_matching_role_key(name) {
            grouped.entry(key).or_default().push(&channels[name]);
        }
    }

    let mut group_keys: Vec<&&'static str> = grouped.keys().collect();
    group_keys.sort();
    let mut group_rms = Vec::new();
    for key in group_keys {
        let group = &grouped[key];
        if group.len() < 2 {
            continue;
        }
        if let Some(rms) = group_mean_deviation_rms_db(group, (300.0, 4_000.0)) {
            group_rms.push(rms);
        }
    }
    mean(&group_rms)
}

#[cfg(test)]
pub(in super::super) fn role_aware_channel_matching_groups(
    final_curves: &HashMap<String, roomeq_model::Curve>,
) -> Vec<HashMap<String, roomeq_model::Curve>> {
    role_aware_channel_matching_groups_with_keys(final_curves)
        .into_iter()
        .map(|group| group.curves)
        .collect()
}

pub(in super::super) fn role_aware_channel_matching_groups_with_keys(
    final_curves: &HashMap<String, roomeq_model::Curve>,
) -> Vec<RoleChannelMatchingGroup> {
    let mut grouped: HashMap<&'static str, HashMap<String, roomeq_model::Curve>> = HashMap::new();

    for (name, curve) in final_curves {
        if let Some(key) = channel_matching_role_key(name) {
            grouped
                .entry(key)
                .or_default()
                .insert(name.clone(), curve.clone());
        }
    }

    let order = [
        "front_lr",
        "side_surrounds",
        "rear_surrounds",
        "wides",
        "top_front",
        "top_middle",
        "top_rear",
        "generic",
    ];

    order
        .iter()
        .filter_map(|key| {
            grouped.remove(key).map(|curves| RoleChannelMatchingGroup {
                role_key: key,
                curves,
            })
        })
        .filter(|group| group.curves.len() > 1)
        .collect()
}
