use crate::Curve;
use roomeq_model::{
    CrossoverCancellationBaseline, CrossoverCancellationContext, CrossoverCancellationEvidence,
};

pub fn assess_configured_crossover_cancellation(
    config: &roomeq_model::RoomConfig,
    source: &str,
    main: &Curve,
    bass: &Curve,
    combined: &Curve,
    crossover_hz: f64,
) -> Option<CrossoverCancellationEvidence> {
    let empty = CrossoverCancellationContext {
        limit_db: config.optimizer.max_crossover_cancellation_db,
        sources: Default::default(),
    };
    assess_crossover_cancellation(
        source,
        main,
        bass,
        combined,
        crossover_hz,
        Some(
            config
                .optimizer
                .crossover_cancellation_baseline
                .as_ref()
                .unwrap_or(&empty),
        ),
    )
}

pub fn cancellation_baseline(
    main: &Curve,
    bass: &Curve,
    crossover_hz: f64,
) -> Option<CrossoverCancellationBaseline> {
    if !super::same_frequency_grid(&main.freq, &bass.freq)
        || !super::curve_has_usable_phase(main)
        || !super::curve_has_usable_phase(bass)
        || main.spl.len() != main.freq.len()
        || bass.spl.len() != main.freq.len()
        || main.spl.iter().chain(&bass.spl).any(|v| !v.is_finite())
    {
        return None;
    }
    let sum = super::complex_sum_mains(&[main, bass]);
    Some(CrossoverCancellationBaseline {
        crossover_hz,
        frequencies_hz: main.freq.to_vec(),
        cancellation_db: main
            .spl
            .iter()
            .zip(&bass.spl)
            .zip(&sum.spl)
            .map(|((m, b), s)| (m.max(*b) - s).max(0.0))
            .collect(),
    })
}

/// Compare both spectra on the candidate grid, never extrapolating a baseline.
pub fn assess_crossover_cancellation(
    source: &str,
    main: &Curve,
    bass: &Curve,
    combined: &Curve,
    crossover_hz: f64,
    context: Option<&CrossoverCancellationContext>,
) -> Option<CrossoverCancellationEvidence> {
    if !super::same_frequency_grid(&main.freq, &bass.freq)
        || !super::same_frequency_grid(&main.freq, &combined.freq)
        || main.spl.len() != main.freq.len()
        || bass.spl.len() != main.freq.len()
        || combined.spl.len() != main.freq.len()
    {
        return None;
    }
    let baseline = context.and_then(|c| c.sources.get(source));
    let old_xo = baseline.map_or(crossover_hz, |b| b.crossover_hz);
    let lo = (crossover_hz.min(old_xo) / 2.0)
        .max(20.0)
        .max(*main.freq.first()?);
    let hi = (crossover_hz.max(old_xo) * 2.0)
        .min(2000.0)
        .min(*main.freq.last()?);
    let in_band = |f: f64| {
        f >= lo
            && f <= hi
            && ((f >= crossover_hz / 2.0 && f <= crossover_hz * 2.0)
                || (f >= old_xo / 2.0 && f <= old_xo * 2.0))
    };
    let indices: Vec<_> = main
        .freq
        .iter()
        .enumerate()
        .filter(|(_, f)| in_band(**f))
        .map(|(i, _)| i)
        .collect();
    if indices.len() < 2 {
        return None;
    }
    let mut worst = (0.0_f64, main.freq[indices[0]]);
    for &i in &indices {
        if !main.spl[i].is_finite() || !bass.spl[i].is_finite() || !combined.spl[i].is_finite() {
            return None;
        }
        let deficit = (main.spl[i].max(bass.spl[i]) - combined.spl[i]).max(0.0);
        if deficit > worst.0 {
            worst = (deficit, main.freq[i]);
        }
    }
    let before = baseline.and_then(|b| {
        if b.frequencies_hz.len() < 2
            || b.frequencies_hz.len() != b.cancellation_db.len()
            || b.frequencies_hz[0] > lo
            || *b.frequencies_hz.last()? < hi
            || b.frequencies_hz.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || b.frequencies_hz.windows(2).any(|w| w[0] >= w[1])
            || b.cancellation_db.iter().any(|d| !d.is_finite() || *d < 0.0)
        {
            return None;
        }
        let curve = Curve {
            freq: b.frequencies_hz.clone().into(),
            spl: b.cancellation_db.clone().into(),
            ..Default::default()
        };
        let aligned = autoeq_core::curve_transforms::interpolate_log_space(&main.freq, &curve);
        indices
            .iter()
            .map(|&i| (aligned.spl[i], main.freq[i]))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    });
    let limit = context.map_or(roomeq_model::DEFAULT_MAX_CROSSOVER_CANCELLATION_DB, |c| {
        c.limit_db
    });
    let accepted =
        roomeq_model::crossover_cancellation_accepted(worst.0, before.map(|b| b.0), limit);
    let reason = if worst.0 <= limit + roomeq_model::CROSSOVER_CANCELLATION_TOLERANCE_DB {
        "within_limit"
    } else if accepted {
        "improved_residual_cancellation"
    } else if before.is_none() {
        "baseline_evidence_unavailable"
    } else {
        "insufficient_cancellation_improvement"
    };
    Some(CrossoverCancellationEvidence {
        source_channel: source.into(),
        baseline_db: before.map(|b| b.0),
        baseline_worst_frequency_hz: before.map(|b| b.1),
        final_db: worst.0,
        final_worst_frequency_hz: worst.1,
        comparison_band_hz: [lo, hi],
        limit_db: limit,
        improvement_db: before.map(|b| b.0 - worst.0),
        accepted,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn branches(deficit: f64) -> (Curve, Curve) {
        let main = Curve {
            freq: ndarray::array![20., 40., 80., 160., 320., 640., 2000.],
            spl: ndarray::Array1::zeros(7),
            phase: Some(ndarray::Array1::zeros(7)),
            ..Default::default()
        };
        let mut bass = main.clone();
        let phase = (10_f64.powf(-deficit / 20.0) / 2.0).acos().to_degrees() * 2.0;
        bass.phase = Some(ndarray::Array1::from_elem(7, phase));
        (main, bass)
    }
    fn context() -> CrossoverCancellationContext {
        let (main, bass) = branches(10.0);
        CrossoverCancellationContext {
            limit_db: 3.0,
            sources: [(
                "L".into(),
                cancellation_baseline(&main, &bass, 80.).unwrap(),
            )]
            .into(),
        }
    }
    #[test]
    fn cancellation_replay_credits_improvement_against_frozen_baseline() {
        let context = context();
        for (deficit, accepted) in [(4.0, true), (10.0, false), (11.0, false)] {
            let (main, bass) = branches(deficit);
            let sum = super::super::complex_sum_mains(&[&main, &bass]);
            let result =
                assess_crossover_cancellation("L", &main, &bass, &sum, 80., Some(&context))
                    .unwrap();
            assert_eq!(result.accepted, accepted);
            assert!((result.final_db - deficit).abs() < 1e-10);
            assert!((result.baseline_db.unwrap() - 10.0).abs() < 1e-10);
            let other = assess_crossover_cancellation("R", &main, &bass, &sum, 80., Some(&context))
                .unwrap();
            assert!(!other.accepted, "L evidence must not authorize R");
        }
    }
    #[test]
    fn cancellation_compares_both_crossover_windows_and_rejects_invalid_support() {
        let mut context = context();
        let (main, mut bass) = branches(2.0);
        bass.phase.as_mut().unwrap()[1] =
            (10_f64.powf(-12.0 / 20.0) / 2.0).acos().to_degrees() * 2.0;
        let sum = super::super::complex_sum_mains(&[&main, &bass]);
        let result =
            assess_crossover_cancellation("L", &main, &bass, &sum, 320., Some(&context)).unwrap();
        assert!(!result.accepted);
        assert_eq!(result.final_worst_frequency_hz, 40.0);
        assert_eq!(result.comparison_band_hz, [40., 640.]);
        context.sources.get_mut("L").unwrap().cancellation_db[0] = f64::NAN;
        let result =
            assess_crossover_cancellation("L", &main, &bass, &sum, 320., Some(&context)).unwrap();
        assert_eq!(result.baseline_db, None);
        assert!(!result.accepted);
    }
    #[test]
    fn cancellation_is_invariant_to_common_safety_attenuation() {
        let (mut main, mut bass) = branches(4.0);
        main.spl.mapv_inplace(|v| v - 20.0);
        bass.spl.mapv_inplace(|v| v - 20.0);
        let sum = super::super::complex_sum_mains(&[&main, &bass]);
        let result =
            assess_crossover_cancellation("L", &main, &bass, &sum, 80., Some(&context())).unwrap();
        assert!(result.accepted);
        assert!((result.improvement_db.unwrap() - 6.0).abs() < 1e-10);
    }
}
