use super::super::types::ChannelOptimizationResult;
use super::super::*;
use super::misc::apply_phase_only_adjustment_to_reported_curve;

pub(in super::super) fn sync_reported_phase_adjustment(
    channel_name: &str,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    delay_ms: f64,
    invert_polarity: bool,
) {
    if let Some(ch_result) = channel_results.get_mut(channel_name) {
        apply_phase_only_adjustment_to_reported_curve(
            &mut ch_result.final_curve,
            delay_ms,
            invert_polarity,
        );

        if let Some(chain) = channel_chains.get_mut(channel_name) {
            chain.final_curve = Some((&ch_result.final_curve).into());
        }
    } else if let Some(chain) = channel_chains.get_mut(channel_name)
        && let Some(final_curve) = chain.final_curve.clone()
    {
        let mut curve: Curve = final_curve.into();
        apply_phase_only_adjustment_to_reported_curve(&mut curve, delay_ms, invert_polarity);
        chain.final_curve = Some((&curve).into());
    }
}

pub(in super::super) fn sync_reported_gain_adjustment(
    channel_name: &str,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    gain_db: f64,
    invert_polarity: bool,
) {
    if let Some(ch_result) = channel_results.get_mut(channel_name) {
        ch_result.final_curve.spl = &ch_result.final_curve.spl + gain_db;
        apply_phase_only_adjustment_to_reported_curve(
            &mut ch_result.final_curve,
            0.0,
            invert_polarity,
        );

        if let Some(chain) = channel_chains.get_mut(channel_name) {
            chain.final_curve = Some((&ch_result.final_curve).into());
        }
    } else if let Some(chain) = channel_chains.get_mut(channel_name)
        && let Some(final_curve) = chain.final_curve.clone()
    {
        let mut curve: Curve = final_curve.into();
        curve.spl = &curve.spl + gain_db;
        apply_phase_only_adjustment_to_reported_curve(&mut curve, 0.0, invert_polarity);
        chain.final_curve = Some((&curve).into());
    }
}

pub(in super::super) fn sync_reported_biquad_adjustment(
    channel_name: &str,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    filters: &[Biquad],
    sample_rate: f64,
) {
    if filters.is_empty() {
        return;
    }

    if let Some(ch_result) = channel_results.get_mut(channel_name) {
        let response = crate::response::compute_peq_complex_response(
            filters,
            &ch_result.final_curve.freq,
            sample_rate,
        );
        ch_result.final_curve =
            crate::response::apply_complex_response(&ch_result.final_curve, &response);

        if let Some(chain) = channel_chains.get_mut(channel_name) {
            chain.final_curve = Some((&ch_result.final_curve).into());
        }
    } else if let Some(chain) = channel_chains.get_mut(channel_name)
        && let Some(final_curve) = chain.final_curve.clone()
    {
        let curve: Curve = final_curve.into();
        let response =
            crate::response::compute_peq_complex_response(filters, &curve.freq, sample_rate);
        let corrected = crate::response::apply_complex_response(&curve, &response);
        chain.final_curve = Some((&corrected).into());
    }
}

pub(in super::super) fn sync_reported_fir_adjustment(
    channel_name: &str,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
    coeffs: &[f64],
    sample_rate: f64,
) {
    if coeffs.is_empty() {
        return;
    }

    if let Some(ch_result) = channel_results.get_mut(channel_name) {
        let response = crate::response::compute_fir_complex_response(
            coeffs,
            &ch_result.final_curve.freq,
            sample_rate,
        );
        ch_result.final_curve =
            crate::response::apply_complex_response(&ch_result.final_curve, &response);

        if let Some(chain) = channel_chains.get_mut(channel_name) {
            chain.final_curve = Some((&ch_result.final_curve).into());
        }
    } else if let Some(chain) = channel_chains.get_mut(channel_name)
        && let Some(final_curve) = chain.final_curve.clone()
    {
        let curve: Curve = final_curve.into();
        let response =
            crate::response::compute_fir_complex_response(coeffs, &curve.freq, sample_rate);
        let corrected = crate::response::apply_complex_response(&curve, &response);
        chain.final_curve = Some((&corrected).into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::ChannelOptimizationResult;
    use crate::roomeq::types::{ChannelDspChain, CurveData};
    use math_audio_iir_fir::{Biquad, BiquadFilterType};
    use ndarray::Array1;
    use std::collections::HashMap;

    fn small_curve() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: Some(Array1::from_elem(16, 0.0)),
            ..Default::default()
        }
    }

    fn curve_data(curve: &crate::Curve) -> CurveData {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.as_ref().map(|p| p.to_vec()),
            norm_range: None,
        }
    }

    fn make_channel(name: &str) -> (ChannelOptimizationResult, ChannelDspChain) {
        let curve = small_curve();
        let ch = ChannelOptimizationResult {
            name: name.to_string(),
            pre_score: 0.0,
            post_score: 0.0,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
        };
        let chain = ChannelDspChain {
            channel: name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(curve_data(&curve)),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        (ch, chain)
    }

    #[test]
    fn sync_phase_adjustment_updates_result_and_chain() {
        let (ch, chain) = make_channel("left");
        let mut results = HashMap::from([("left".to_string(), ch)]);
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        sync_reported_phase_adjustment("left", &mut results, &mut chains, 1.0, true);
        let curve = &results["left"].final_curve;
        let freq = curve.freq[0];
        let phase = curve.phase.as_ref().unwrap()[0];
        let expected = 180.0 - 360.0 * freq * 1e-3;
        assert!((phase - expected).abs() < 1.0);
        assert!(chains["left"].final_curve.is_some());
    }

    #[test]
    fn sync_gain_adjustment_adds_db_and_inverts() {
        let (ch, chain) = make_channel("left");
        let mut results = HashMap::from([("left".to_string(), ch)]);
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let before = results["left"].final_curve.spl[0];
        sync_reported_gain_adjustment("left", &mut results, &mut chains, 3.0, true);
        let after = results["left"].final_curve.spl[0];
        assert!((after - before - 3.0).abs() < 1e-9);
        let phase = results["left"].final_curve.phase.as_ref().unwrap();
        assert!((phase[0] - 180.0).abs() < 1e-9);
    }

    #[test]
    fn sync_biquad_adjustment_applies_filter() {
        let (ch, chain) = make_channel("left");
        let mut results = HashMap::from([("left".to_string(), ch)]);
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let before = results["left"].final_curve.spl[5];
        let filters = vec![Biquad::new(
            BiquadFilterType::Peak,
            1000.0,
            48_000.0,
            1.0,
            3.0,
        )];
        sync_reported_biquad_adjustment("left", &mut results, &mut chains, &filters, 48_000.0);
        let after = results["left"].final_curve.spl[5];
        assert!((after - before).abs() > 1e-6);
    }

    #[test]
    fn sync_fir_adjustment_applies_coefficients() {
        let (ch, chain) = make_channel("left");
        let mut results = HashMap::from([("left".to_string(), ch)]);
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let before = results["left"].final_curve.spl[5];
        let coeffs = vec![0.5, 0.5];
        sync_reported_fir_adjustment("left", &mut results, &mut chains, &coeffs, 48_000.0);
        let after = results["left"].final_curve.spl[5];
        assert!((after - before).abs() > 1e-6);
    }

    #[test]
    fn sync_biquad_empty_filters_noop() {
        let (ch, chain) = make_channel("left");
        let mut results = HashMap::from([("left".to_string(), ch)]);
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        let before = results["left"].final_curve.spl.clone();
        sync_reported_biquad_adjustment("left", &mut results, &mut chains, &[], 48_000.0);
        assert_eq!(results["left"].final_curve.spl, before);
    }

    #[test]
    fn sync_phase_adjustment_updates_chain_when_result_missing() {
        let (_, chain) = make_channel("left");
        let mut results = HashMap::<String, ChannelOptimizationResult>::new();
        let mut chains = HashMap::from([("left".to_string(), chain)]);
        sync_reported_phase_adjustment("left", &mut results, &mut chains, 0.5, false);
        let curve = chains["left"].final_curve.as_ref().unwrap();
        let freq = curve.freq[0];
        let chain_phase = curve.phase.as_ref().unwrap()[0];
        let expected = -360.0 * freq * 0.5e-3;
        assert!((chain_phase - expected).abs() < 1.0);
    }
}
