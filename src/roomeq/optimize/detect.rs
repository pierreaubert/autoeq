use crate::Curve;

pub(in super::super) fn detect_passband_and_mean(curve: &Curve) -> (Option<(f64, f64)>, f64) {
    roomeq_analysis::response_metrics::detect_passband_and_mean(curve)
}
