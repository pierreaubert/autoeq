use super::mso::mso_params_per_optimized_sub;
use super::types::MsoSolution;
use roomeq_model::MultiSeatConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct MsoSearchOptions {
    pub(super) optimize_polarity: bool,
    pub(super) allpass_filters_per_sub: usize,
    pub(super) allpass_min_freq: f64,
    pub(super) allpass_max_freq: f64,
}

impl MsoSearchOptions {
    pub(super) fn from_config(config: &MultiSeatConfig, min_freq: f64, max_freq: f64) -> Self {
        let allpass_min_freq = min_freq.max(20.0);
        let usable_allpass_max = max_freq.min(200.0);
        let allpass_filters_per_sub = if allpass_min_freq < usable_allpass_max {
            config.allpass_filters_per_sub
        } else {
            if config.allpass_filters_per_sub > 0 {
                log::warn!(
                    "MSO all-pass filters disabled because [{min_freq:.1}, {max_freq:.1}] Hz has no overlap with the supported 20-200 Hz all-pass band"
                );
            }
            0
        };
        let allpass_max_freq = usable_allpass_max.max(allpass_min_freq);
        Self {
            optimize_polarity: config.optimize_polarity,
            allpass_filters_per_sub,
            allpass_min_freq,
            allpass_max_freq,
        }
    }
}

pub(super) fn decode_mso_params(
    params: &[f64],
    num_subs: usize,
    options: MsoSearchOptions,
) -> MsoSolution {
    let mut gains = vec![0.0; num_subs];
    let mut delays = vec![0.0; num_subs];
    let mut polarities = vec![false; num_subs];
    let mut allpass_filters = vec![Vec::new(); num_subs];
    let per_sub = mso_params_per_optimized_sub(options);

    for sub_idx in 1..num_subs {
        let mut offset = (sub_idx - 1) * per_sub;
        gains[sub_idx] = params[offset];
        offset += 1;
        delays[sub_idx] = params[offset];
        offset += 1;

        if options.optimize_polarity {
            polarities[sub_idx] = params[offset] >= 0.5;
            offset += 1;
        }

        for _ in 0..options.allpass_filters_per_sub {
            let freq = params[offset];
            let q = params[offset + 1];
            allpass_filters[sub_idx].push((freq, q));
            offset += 2;
        }
    }

    (gains, delays, polarities, allpass_filters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allpass_search_is_disabled_when_band_starts_above_200_hz() {
        let config = MultiSeatConfig {
            allpass_filters_per_sub: 3,
            ..MultiSeatConfig::default()
        };

        let unsupported = MsoSearchOptions::from_config(&config, 250.0, 2_000.0);
        assert_eq!(unsupported.allpass_filters_per_sub, 0);
        assert_eq!(unsupported.allpass_min_freq, 250.0);
        assert_eq!(unsupported.allpass_max_freq, 250.0);

        let supported = MsoSearchOptions::from_config(&config, 20.0, 120.0);
        assert_eq!(supported.allpass_filters_per_sub, 3);
        assert_eq!(supported.allpass_max_freq, 120.0);
    }
}
