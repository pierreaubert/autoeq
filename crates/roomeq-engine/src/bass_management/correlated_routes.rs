//! Select coherent combinations of route candidates already scored by the search.

// Rust guideline compliant 2026-02-21
use crate::Curve;
use roomeq_model::{BassManagementGroupReport, BassManagementSourceReport};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub(super) struct GroupCandidate {
    pub group: BassManagementGroupReport,
    pub sources: Vec<BassManagementSourceReport>,
    pub responses: Vec<(String, Curve)>,
    pub score: f64,
}

#[derive(Debug)]
pub(super) struct GroupAlternatives {
    pub reference: Vec<(String, Curve)>,
    pub current: GroupCandidate,
    pub common_polarity: [Option<GroupCandidate>; 2],
}

fn passes(
    reference: &BTreeMap<String, Curve>,
    candidates: &[&GroupCandidate],
    limit_db: f64,
) -> bool {
    let delivered: BTreeMap<_, _> = candidates
        .iter()
        .flat_map(|candidate| candidate.responses.iter().cloned())
        .collect();
    if delivered.keys().ne(reference.keys()) {
        return false;
    }
    let mut groups = BTreeSet::new();
    groups.insert(reference.keys().cloned().collect::<Vec<_>>());
    for candidate in candidates {
        groups.insert(
            candidate
                .responses
                .iter()
                .map(|(name, _)| name.clone())
                .collect(),
        );
    }
    use roomeq_model::HomeCinemaRole::{Center, FrontLeft, FrontRight};
    for roles in [
        &[FrontLeft, FrontRight][..],
        &[FrontLeft, Center, FrontRight][..],
    ] {
        let selected: Vec<_> = reference
            .keys()
            .filter(|name| roles.contains(&roomeq_model::home_cinema::role_for_channel(name)))
            .cloned()
            .collect();
        if selected.len() == roles.len() {
            groups.insert(selected);
        }
    }
    groups
        .into_iter()
        .filter(|group| group.len() >= 2)
        .all(|group| {
            let before: Vec<_> = group.iter().map(|name| reference[name].clone()).collect();
            let after: Vec<_> = group.iter().map(|name| delivered[name].clone()).collect();
            roomeq_quality::correlated_bass_efficiency_regression(
                &before,
                &after,
                super::optimize::CORRELATED_BASS_MAX_HZ,
            )
            .is_ok_and(|(loss, _)| loss <= limit_db + 1e-6)
        })
}

/// Keep independent winners when coherent; otherwise test two cached combinations.
///
/// Every group alternative has already passed its unchanged source/crossover
/// acceptance. This step adds no objective calls and does not replace final
/// serialized replay at training and held-out seats.
pub(super) fn select(
    groups: &[GroupAlternatives],
    roles: &[String],
    limit_db: f64,
) -> Option<Vec<GroupCandidate>> {
    let reference: BTreeMap<_, _> = groups
        .iter()
        .flat_map(|group| group.reference.iter().cloned())
        .collect();
    if reference.len() != roles.len() || roles.iter().any(|role| !reference.contains_key(role)) {
        return None;
    }
    let current: Vec<_> = groups.iter().map(|group| &group.current).collect();
    if passes(&reference, &current, limit_db) {
        return None;
    }
    let mut best: Option<(f64, Vec<&GroupCandidate>)> = None;
    for polarity in 0..2 {
        let candidates: Option<Vec<_>> = groups
            .iter()
            .map(|group| {
                group.common_polarity[polarity].as_ref().or_else(|| {
                    group
                        .current
                        .sources
                        .iter()
                        .all(|source| source.polarity_inverted == (polarity == 1))
                        .then_some(&group.current)
                })
            })
            .collect();
        let Some(candidates) = candidates else {
            continue;
        };
        let score: f64 = candidates.iter().map(|candidate| candidate.score).sum();
        if score.is_finite()
            && passes(&reference, &candidates, limit_db)
            && best.as_ref().is_none_or(|(previous, _)| score < *previous)
        {
            best = Some((score, candidates));
        }
    }
    best.map(|(_, candidates)| candidates.into_iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(group_id: &str, names: &[&str], inverted: bool) -> GroupCandidate {
        GroupCandidate {
            group: BassManagementGroupReport {
                group_id: group_id.into(),
                roles: names.iter().map(|name| (*name).into()).collect(),
                crossover_type: "LR24".into(),
                selected_crossover_hz: Some(80.0),
                configured_crossover_hz: Some(80.0),
                main_delay_ms: 0.0,
                bass_route_delay_ms: 0.0,
                polarity_inverted: inverted,
                trim_db: 0.0,
                objective_before: Some(1.0),
                objective_after: Some(0.1),
                selected_sub_low_pass_hz: vec![],
                advisories: vec![],
            },
            sources: names
                .iter()
                .map(|name| BassManagementSourceReport {
                    source_channel: (*name).into(),
                    group_id: group_id.into(),
                    main_delay_ms: 0.0,
                    bass_route_delay_ms: 0.0,
                    polarity_inverted: inverted,
                    trim_db: 0.0,
                    objective_before: Some(1.0),
                    objective_after: Some(0.1),
                    accepted: true,
                    safety_restored: false,
                    advisories: vec![],
                })
                .collect(),
            responses: names
                .iter()
                .map(|name| {
                    (
                        (*name).into(),
                        Curve {
                            freq: ndarray::array![20.0, 50.0, 80.0, 150.0, 300.0],
                            spl: ndarray::Array1::from_elem(5, 80.0),
                            phase: Some(ndarray::Array1::from_elem(
                                5,
                                if inverted { 180.0 } else { 0.0 },
                            )),
                            ..Default::default()
                        },
                    )
                })
                .collect(),
            score: 0.1,
        }
    }

    fn groups() -> Vec<GroupAlternatives> {
        let front = candidate("fronts", &["L", "R"], false);
        let center = candidate("center", &["C"], true);
        let corrected = candidate("center", &["C"], false);
        vec![
            GroupAlternatives {
                reference: front.responses.clone(),
                current: front,
                common_polarity: [None, None],
            },
            GroupAlternatives {
                reference: corrected.responses.clone(),
                current: center,
                common_polarity: [Some(corrected), None],
            },
        ]
    }

    #[test]
    fn cross_group_polarity_cancellation_is_repaired_using_cached_validated_routes() {
        let roles = vec!["L".into(), "R".into(), "C".into()];
        let groups = groups();
        let selected = select(&groups, &roles, 3.0).expect("coherent cached alternative");
        assert_eq!(selected.len(), 2);
        assert!(
            selected
                .iter()
                .flat_map(|candidate| &candidate.sources)
                .all(|source| !source.polarity_inverted)
        );
        let reference = groups
            .iter()
            .flat_map(|group| group.reference.iter().cloned())
            .collect();
        assert!(passes(
            &reference,
            &selected.iter().collect::<Vec<_>>(),
            3.0
        ));
    }

    #[test]
    fn missing_cached_routes_or_measurement_roles_cannot_create_a_candidate() {
        let roles = vec!["L".into(), "R".into(), "C".into()];
        let mut groups = groups();
        groups[1].common_polarity[0] = None;
        assert!(select(&groups, &roles, 3.0).is_none());
        let roles = vec!["L".into(), "R".into(), "C".into(), "SL".into()];
        assert!(select(&groups, &roles, 3.0).is_none());
    }

    #[test]
    fn coherent_independent_winners_are_preserved() {
        let roles = vec!["L".into(), "R".into(), "C".into()];
        let mut groups = groups();
        groups[1].current = groups[1].common_polarity[0].clone().unwrap();
        assert!(select(&groups, &roles, 3.0).is_none());
    }
}
