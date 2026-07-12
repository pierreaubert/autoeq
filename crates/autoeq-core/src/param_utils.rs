//! Parameter vector utilities for handling different PEQ models
//!
//! This module provides utilities for working with parameter vectors that may have
//! different layouts depending on the PEQ model being used.

use crate::PeqModel;
use crate::iir::BiquadFilterType;

/// Position of each parameter inside a filter's parameter group.
#[derive(Debug, Clone, Copy)]
pub struct ParamLayout {
    /// Index of the encoded filter type, if present.
    pub type_idx: Option<usize>,
    /// Index of the log10 frequency parameter.
    pub freq_idx: usize,
    /// Index of the Q parameter.
    pub q_idx: usize,
    /// Index of the gain parameter.
    pub gain_idx: usize,
}

/// Strategy trait that encapsulates the parameter-vector layout for a PEQ model.
///
/// Implementations describe how many parameters each filter consumes, where each
/// semantic parameter lives, and how to convert between the flat vector and filter
/// descriptors.
pub trait PeqLayout {
    /// Number of scalar parameters stored for each filter.
    fn params_per_filter(&self) -> usize;

    /// Position of each semantic parameter within a filter group.
    fn layout(&self) -> ParamLayout;

    /// Number of filters represented by a parameter vector.
    fn num_filters(&self, x: &[f64]) -> usize {
        x.len() / self.params_per_filter()
    }

    /// Extract the parameters for the `i`-th filter.
    fn get_filter_params(&self, x: &[f64], i: usize) -> FilterParams;

    /// Write the parameters for the `i`-th filter into an existing vector.
    fn set_filter_params(&self, x: &mut [f64], i: usize, params: &FilterParams);

    /// Append a filter's parameters to a vector (used when building a vector from
    /// scratch, e.g. `peq2x`).
    fn append_filter_params(&self, x: &mut Vec<f64>, params: &FilterParams) {
        let l = self.layout();
        if let Some(type_idx) = l.type_idx {
            x.push(params.filter_type.unwrap_or(0.0));
            // Only fixed/free layout is supported; assert indices line up.
            assert_eq!(l.freq_idx, type_idx + 1);
        }
        x.push(params.freq);
        x.push(params.q);
        x.push(params.gain);
    }

    /// Determine the biquad type for filter `i` given the model and optional type
    /// parameter.
    fn determine_filter_type(
        &self,
        i: usize,
        num_filters: usize,
        type_param: Option<f64>,
    ) -> BiquadFilterType;

    /// Build an initial-guess parameter group for filter `i`.
    fn initial_guess_filter(
        &self,
        i: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        min_db: f64,
        max_freq: f64,
    ) -> Vec<f64>;
}

impl PeqLayout for PeqModel {
    fn params_per_filter(&self) -> usize {
        match self {
            PeqModel::Pk
            | PeqModel::HpPk
            | PeqModel::HpPkLp
            | PeqModel::LsPk
            | PeqModel::LsPkHs => 3,
            PeqModel::FreePkFree | PeqModel::Free => 4,
        }
    }

    fn layout(&self) -> ParamLayout {
        match self {
            PeqModel::Pk
            | PeqModel::HpPk
            | PeqModel::HpPkLp
            | PeqModel::LsPk
            | PeqModel::LsPkHs => ParamLayout {
                type_idx: None,
                freq_idx: 0,
                q_idx: 1,
                gain_idx: 2,
            },
            PeqModel::FreePkFree | PeqModel::Free => ParamLayout {
                type_idx: Some(0),
                freq_idx: 1,
                q_idx: 2,
                gain_idx: 3,
            },
        }
    }

    fn get_filter_params(&self, x: &[f64], i: usize) -> FilterParams {
        let l = self.layout();
        let offset = i * self.params_per_filter();
        FilterParams {
            filter_type: l.type_idx.map(|idx| x[offset + idx]),
            freq: x[offset + l.freq_idx],
            q: x[offset + l.q_idx],
            gain: x[offset + l.gain_idx],
        }
    }

    fn set_filter_params(&self, x: &mut [f64], i: usize, params: &FilterParams) {
        let l = self.layout();
        let offset = i * self.params_per_filter();
        if let Some(type_idx) = l.type_idx {
            x[offset + type_idx] = params.filter_type.unwrap_or(0.0);
        }
        x[offset + l.freq_idx] = params.freq;
        x[offset + l.q_idx] = params.q;
        x[offset + l.gain_idx] = params.gain;
    }

    fn determine_filter_type(
        &self,
        i: usize,
        num_filters: usize,
        type_param: Option<f64>,
    ) -> BiquadFilterType {
        match self {
            PeqModel::Pk => BiquadFilterType::Peak,
            PeqModel::HpPk => {
                if i == 0 {
                    BiquadFilterType::HighpassVariableQ
                } else {
                    BiquadFilterType::Peak
                }
            }
            PeqModel::HpPkLp => {
                if i == 0 {
                    BiquadFilterType::HighpassVariableQ
                } else if i == num_filters - 1 {
                    BiquadFilterType::Lowpass
                } else {
                    BiquadFilterType::Peak
                }
            }
            PeqModel::LsPk => {
                if i == 0 {
                    BiquadFilterType::Lowshelf
                } else {
                    BiquadFilterType::Peak
                }
            }
            PeqModel::LsPkHs => {
                if i == 0 {
                    BiquadFilterType::Lowshelf
                } else if i == num_filters - 1 {
                    BiquadFilterType::Highshelf
                } else {
                    BiquadFilterType::Peak
                }
            }
            PeqModel::FreePkFree => {
                if i == 0 || i == num_filters - 1 {
                    decode_filter_type(type_param.unwrap_or(0.0))
                } else {
                    BiquadFilterType::Peak
                }
            }
            PeqModel::Free => decode_filter_type(type_param.unwrap_or(0.0)),
        }
    }

    fn initial_guess_filter(
        &self,
        i: usize,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        min_db: f64,
        max_freq: f64,
    ) -> Vec<f64> {
        let l = self.layout();
        let mut group = Vec::with_capacity(self.params_per_filter());
        let sign = if i.is_multiple_of(2) { 0.5 } else { -0.5 };

        if let Some(type_idx) = l.type_idx {
            group.push(0.0_f64.clamp(lower_bounds[type_idx], upper_bounds[type_idx]));
        }

        let freq = lower_bounds[l.freq_idx]
            .min(max_freq.log10())
            .clamp(lower_bounds[l.freq_idx], upper_bounds[l.freq_idx]);
        group.push(freq);

        let q = (upper_bounds[l.q_idx] * lower_bounds[l.q_idx])
            .sqrt()
            .clamp(lower_bounds[l.q_idx], upper_bounds[l.q_idx]);
        group.push(q);

        let gain = (sign * upper_bounds[l.gain_idx].max(min_db))
            .clamp(lower_bounds[l.gain_idx], upper_bounds[l.gain_idx]);
        group.push(gain);

        group
    }
}

/// Get the number of parameters per filter for a given PEQ model
pub fn params_per_filter(peq_model: PeqModel) -> usize {
    peq_model.params_per_filter()
}

/// Get the number of filters from a parameter vector
pub fn num_filters(x: &[f64], peq_model: PeqModel) -> usize {
    peq_model.num_filters(x)
}

/// Extract filter parameters for the i-th filter
pub fn get_filter_params(x: &[f64], i: usize, peq_model: PeqModel) -> FilterParams {
    peq_model.get_filter_params(x, i)
}

/// Set filter parameters for the i-th filter
pub fn set_filter_params(x: &mut [f64], i: usize, params: &FilterParams, peq_model: PeqModel) {
    peq_model.set_filter_params(x, i, params);
}

/// Container for filter parameters
#[derive(Debug, Clone)]
pub struct FilterParams {
    /// Filter type (encoded as f64 for free filter models)
    pub filter_type: Option<f64>,
    /// Frequency (as log10 for optimization)
    pub freq: f64,
    /// Q factor
    pub q: f64,
    /// Gain in dB
    pub gain: f64,
}

/// Determine the filter type based on model and position
pub fn determine_filter_type(
    i: usize,
    num_filters: usize,
    peq_model: PeqModel,
    type_param: Option<f64>,
) -> BiquadFilterType {
    peq_model.determine_filter_type(i, num_filters, type_param)
}

/// Decode filter type from parameter value
/// Maps continuous parameter to discrete filter types for optimization
pub fn decode_filter_type(type_value: f64) -> BiquadFilterType {
    // Map [0, 1) -> Peak
    // Map [1, 2) -> Lowpass
    // Map [2, 3) -> Highpass
    // Map [3, 4) -> Lowshelf
    // Map [4, 5) -> Highshelf
    // Map [5, 6) -> HighpassVariableQ
    // Map [6, 7) -> Bandpass
    // Map [7, 8) -> Notch
    // Map [8, 9) -> AllPass
    // Map [9, 10) -> Orfanidis low shelf
    // Map [10, 11) -> Orfanidis high shelf
    // Map [11, 12) -> matched peak

    let idx = type_value.floor() as i32;
    match idx {
        0 => BiquadFilterType::Peak,
        1 => BiquadFilterType::Lowpass,
        2 => BiquadFilterType::Highpass,
        3 => BiquadFilterType::Lowshelf,
        4 => BiquadFilterType::Highshelf,
        5 => BiquadFilterType::HighpassVariableQ,
        6 => BiquadFilterType::Bandpass,
        7 => BiquadFilterType::Notch,
        8 => BiquadFilterType::AllPass,
        9 => BiquadFilterType::LowshelfOrf,
        10 => BiquadFilterType::HighshelfOrf,
        11 => BiquadFilterType::PeakMatched,
        _ => BiquadFilterType::Peak, // Default
    }
}

/// Encode filter type to parameter value
pub fn encode_filter_type(filter_type: BiquadFilterType) -> f64 {
    match filter_type {
        BiquadFilterType::Peak => 0.0,
        BiquadFilterType::Lowpass => 1.0,
        BiquadFilterType::Highpass => 2.0,
        BiquadFilterType::Lowshelf => 3.0,
        BiquadFilterType::Highshelf => 4.0,
        BiquadFilterType::HighpassVariableQ => 5.0,
        BiquadFilterType::Bandpass => 6.0,
        BiquadFilterType::Notch => 7.0,
        BiquadFilterType::AllPass => 8.0,
        BiquadFilterType::LowshelfOrf => 9.0,
        BiquadFilterType::HighshelfOrf => 10.0,
        BiquadFilterType::PeakMatched => 11.0,
    }
}

/// Get bounds for filter type parameter
pub fn filter_type_bounds() -> (f64, f64) {
    (0.0, 11.999)
}

/// Convert log10 frequency parameter to Hz
///
/// Filter frequencies are stored as log10(Hz) in the optimization parameter vector.
/// This function converts back to linear Hz.
#[inline]
pub fn freq_from_log10(log_freq: f64) -> f64 {
    10f64.powf(log_freq)
}

/// Convert log10 frequency parameter to Hz with minimum value clamping
///
/// Like `freq_from_log10` but clamps the result to avoid numerical issues
/// with very small or negative frequencies.
#[inline]
pub fn freq_from_log10_clamped(log_freq: f64, min_freq: f64) -> f64 {
    10f64.powf(log_freq).max(min_freq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use crate::iir::BiquadFilterType;

    #[test]
    fn get_filter_params_fixed() {
        let x = vec![2.0, 1.5, 3.0];
        let p = get_filter_params(&x, 0, PeqModel::Pk);
        assert_eq!(p.freq, 2.0);
        assert_eq!(p.q, 1.5);
        assert_eq!(p.gain, 3.0);
        assert!(p.filter_type.is_none());
    }

    #[test]
    fn get_filter_params_free() {
        let x = vec![1.0, 2.0, 1.5, 3.0];
        let p = get_filter_params(&x, 0, PeqModel::Free);
        assert_eq!(p.filter_type, Some(1.0));
        assert_eq!(p.freq, 2.0);
        assert_eq!(p.q, 1.5);
        assert_eq!(p.gain, 3.0);
    }

    #[test]
    fn set_filter_params_round_trip() {
        let mut x = vec![0.0; 3];
        let params = FilterParams {
            filter_type: None,
            freq: 2.0,
            q: 1.5,
            gain: 3.0,
        };
        set_filter_params(&mut x, 0, &params, PeqModel::Pk);
        assert_eq!(x, vec![2.0, 1.5, 3.0]);
    }

    #[test]
    fn determine_filter_type_pk() {
        assert_eq!(
            determine_filter_type(0, 3, PeqModel::Pk, None),
            BiquadFilterType::Peak
        );
    }

    #[test]
    fn determine_filter_type_hp_pk() {
        assert_eq!(
            determine_filter_type(0, 3, PeqModel::HpPk, None),
            BiquadFilterType::HighpassVariableQ
        );
        assert_eq!(
            determine_filter_type(1, 3, PeqModel::HpPk, None),
            BiquadFilterType::Peak
        );
    }

    #[test]
    fn determine_filter_type_ls_pk_hs() {
        assert_eq!(
            determine_filter_type(0, 3, PeqModel::LsPkHs, None),
            BiquadFilterType::Lowshelf
        );
        assert_eq!(
            determine_filter_type(1, 3, PeqModel::LsPkHs, None),
            BiquadFilterType::Peak
        );
        assert_eq!(
            determine_filter_type(2, 3, PeqModel::LsPkHs, None),
            BiquadFilterType::Highshelf
        );
    }

    #[test]
    fn filter_type_bounds_range() {
        let (min, max) = filter_type_bounds();
        assert_eq!(min, 0.0);
        assert_eq!(max, 11.999);
    }

    #[test]
    fn audit_every_encoded_filter_type_is_decodable_and_in_bounds() {
        let filter_types = [
            BiquadFilterType::Peak,
            BiquadFilterType::Lowpass,
            BiquadFilterType::Highpass,
            BiquadFilterType::Lowshelf,
            BiquadFilterType::Highshelf,
            BiquadFilterType::HighpassVariableQ,
            BiquadFilterType::Bandpass,
            BiquadFilterType::Notch,
            BiquadFilterType::AllPass,
            BiquadFilterType::LowshelfOrf,
            BiquadFilterType::HighshelfOrf,
            BiquadFilterType::PeakMatched,
        ];
        let (lower, upper) = filter_type_bounds();

        for filter_type in filter_types {
            let encoded = encode_filter_type(filter_type);
            assert!(encoded >= lower && encoded <= upper);
            assert_eq!(decode_filter_type(encoded), filter_type);
        }
    }

    #[test]
    fn freq_from_log10_basic() {
        assert!((freq_from_log10(2.0) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn freq_from_log10_clamped_respects_min() {
        assert_eq!(freq_from_log10_clamped(-10.0, 20.0), 20.0);
    }
}
