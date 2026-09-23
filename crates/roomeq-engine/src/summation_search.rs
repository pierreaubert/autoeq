//! Crossover-overlap summation search (Wave 1, step 2).
//!
//! Peak-only alignment is replaced by a polarity/delay/gain search over
//! the full overlap band using `Hsum(f) = Hmain(f) + g Hsub(f)
//! exp(-j2pifτ)`. Candidates are disambiguated by gross timing across
//! the band, so a single-frequency phase match alone never wins.
//!
//! The module also owns the explicit [`DelayLedger`] (seconds and
//! samples, with advances realized as reduced existing delay or added
//! common latency) and [`reverify_combined`], which replays the accepted
//! alignment's combined response at every seat.

// Rust guideline compliant 2026-02-21

use num_complex::Complex;
use std::f64::consts::PI;

/// Candidate grid for the polarity/delay/gain search.
///
/// All entries are explicit caller policy: no hidden defaults decide the
/// alignment. Delays are relative sub-to-main offsets in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchGrid {
    /// Relative delays in seconds to evaluate.
    pub delays_s: Vec<f64>,
    /// Sub gains in dB to evaluate.
    pub gains_db: Vec<f64>,
    /// Whether to evaluate both polarities at every delay/gain.
    pub include_polarity_inversion: bool,
}

impl SearchGrid {
    /// Reject empty grids and nonfinite entries.
    ///
    /// # Errors
    ///
    /// Returns a reason for an empty grid or a nonfinite entry.
    pub fn validate(&self) -> Result<(), String> {
        if self.delays_s.is_empty() {
            return Err(String::from("search grid holds no delays"));
        }
        if self.gains_db.is_empty() {
            return Err(String::from("search grid holds no gains"));
        }
        if !self
            .delays_s
            .iter()
            .all(|delay| delay.is_finite() && *delay >= 0.0)
        {
            return Err(String::from(
                "search grid delays must be finite and nonnegative",
            ));
        }
        if !self.gains_db.iter().all(|gain| gain.is_finite()) {
            return Err(String::from("search grid gains must be finite"));
        }
        Ok(())
    }
}

/// One evaluated alignment candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedCandidate {
    /// Whether the sub polarity is inverted.
    pub polarity_inverted: bool,
    /// Relative sub delay in seconds.
    pub delay_s: f64,
    /// Sub gain in dB.
    pub gain_db: f64,
    /// Band summation error (0 is perfectly constructive everywhere).
    pub band_error: f64,
}

/// Outcome of [`search_summation`].
#[derive(Debug, Clone, PartialEq)]
pub struct SummationSearchOutcome {
    /// Lowest band-error candidate.
    pub best: AlignedCandidate,
    /// Number of evaluated candidates.
    pub evaluated: usize,
}

fn check_response_grid(
    freqs: &[f64],
    main_mag_db: &[f64],
    main_phase_deg: &[f64],
    sub_mag_db: &[f64],
    sub_phase_deg: &[f64],
) -> Result<(), String> {
    let len = freqs.len();
    if len == 0 {
        return Err(String::from("response grid is empty"));
    }
    if main_mag_db.len() != len
        || main_phase_deg.len() != len
        || sub_mag_db.len() != len
        || sub_phase_deg.len() != len
    {
        return Err(String::from(
            "main/sub response grids must share one length; resample explicitly first",
        ));
    }
    if !freqs.iter().all(|freq| freq.is_finite() && *freq > 0.0) {
        return Err(String::from("frequencies must be finite and positive"));
    }
    Ok(())
}

fn to_complex(mag_db: f64, phase_deg: f64) -> Result<Complex<f64>, String> {
    if !mag_db.is_finite() || !phase_deg.is_finite() {
        return Err(String::from("response holds a nonfinite entry"));
    }
    let magnitude = 10.0_f64.powf(mag_db / 20.0);
    let phase_rad = phase_deg * PI / 180.0;
    Ok(Complex::new(
        magnitude * phase_rad.cos(),
        magnitude * phase_rad.sin(),
    ))
}

/// Band summation error of one alignment candidate.
///
/// The error is the band mean of `(ideal - |Hsum|)^2` normalized by the
/// band mean of `ideal^2`, where `ideal = |Hmain| + g|Hsub|` is the
/// perfectly constructive sum. A candidate matching phase at one
/// frequency but cancelling elsewhere scores badly: single-frequency
/// agreement alone cannot win.
///
/// # Errors
///
/// Returns a reason for mismatched grids, an empty overlap band, or
/// nonfinite response entries.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_candidate(
    freqs: &[f64],
    main_mag_db: &[f64],
    main_phase_deg: &[f64],
    sub_mag_db: &[f64],
    sub_phase_deg: &[f64],
    band_hz: [f64; 2],
    polarity_inverted: bool,
    delay_s: f64,
    gain_db: f64,
) -> Result<f64, String> {
    check_response_grid(
        freqs,
        main_mag_db,
        main_phase_deg,
        sub_mag_db,
        sub_phase_deg,
    )?;
    if !band_hz[0].is_finite()
        || !band_hz[1].is_finite()
        || band_hz[1] <= band_hz[0]
        || !delay_s.is_finite()
        || delay_s < 0.0
        || !gain_db.is_finite()
    {
        return Err(String::from("invalid band, delay, or gain"));
    }
    let mut gain = 10.0_f64.powf(gain_db / 20.0);
    if polarity_inverted {
        gain = -gain;
    }
    let mut error_sum = 0.0;
    let mut ideal_sum = 0.0;
    let mut count = 0_usize;
    for index in 0..freqs.len() {
        let freq = freqs[index];
        if freq < band_hz[0] || freq > band_hz[1] {
            continue;
        }
        let main = to_complex(main_mag_db[index], main_phase_deg[index])?;
        let sub = to_complex(sub_mag_db[index], sub_phase_deg[index])?;
        let shift = Complex::new(0.0, -2.0 * PI * freq * delay_s).exp();
        let hsum = main + sub * gain * shift;
        let ideal = main.norm() + gain.abs() * sub.norm();
        error_sum += (ideal - hsum.norm()).powi(2);
        ideal_sum += ideal.powi(2);
        count += 1;
    }
    if count == 0 {
        return Err(String::from("overlap band holds no grid frequencies"));
    }
    if ideal_sum <= 0.0 {
        return Err(String::from("ideal summation energy is zero"));
    }
    let count_f64 = count as f64;
    Ok(error_sum / count_f64 / (ideal_sum / count_f64))
}

/// Search polarity/delay/gain over the full overlap band.
///
/// Returns the lowest band-error candidate. Gross timing across the band
/// disambiguates phase-cycle aliases (for example a 20 ms ambiguity at
/// 50 Hz): aliases agree at one frequency and diverge across the band.
/// Selection needs at least two grid frequencies in the band; the pure
/// metric ([`evaluate_candidate`]) still scores a single bin so tests can
/// demonstrate exactly why one bin alone must never select.
///
/// # Errors
///
/// Returns grid, band, or response reasons from the validators above.
#[allow(clippy::too_many_arguments)]
pub fn search_summation(
    freqs: &[f64],
    main_mag_db: &[f64],
    main_phase_deg: &[f64],
    sub_mag_db: &[f64],
    sub_phase_deg: &[f64],
    band_hz: [f64; 2],
    grid: &SearchGrid,
) -> Result<SummationSearchOutcome, String> {
    grid.validate()?;
    if !band_hz[0].is_finite() || !band_hz[1].is_finite() || band_hz[1] <= band_hz[0] {
        return Err(String::from("overlap band must satisfy finite lo < hi"));
    }
    let band_bins = freqs
        .iter()
        .filter(|freq| **freq >= band_hz[0] && **freq <= band_hz[1])
        .count();
    if band_bins < 2 {
        return Err(String::from(
            "overlap band needs at least two grid frequencies to resolve phase-cycle ambiguity",
        ));
    }
    let polarities: &[bool] = if grid.include_polarity_inversion {
        &[false, true]
    } else {
        &[false]
    };
    let mut best: Option<AlignedCandidate> = None;
    let mut evaluated = 0_usize;
    for polarity in polarities {
        for delay_s in &grid.delays_s {
            for gain_db in &grid.gains_db {
                let error = evaluate_candidate(
                    freqs,
                    main_mag_db,
                    main_phase_deg,
                    sub_mag_db,
                    sub_phase_deg,
                    band_hz,
                    *polarity,
                    *delay_s,
                    *gain_db,
                )?;
                evaluated += 1;
                let replace = best
                    .as_ref()
                    .is_none_or(|current: &AlignedCandidate| error < current.band_error);
                if replace {
                    best = Some(AlignedCandidate {
                        polarity_inverted: *polarity,
                        delay_s: *delay_s,
                        gain_db: *gain_db,
                        band_error: error,
                    });
                }
            }
        }
    }
    best.map_or_else(
        || Err(String::from("search grid evaluated no candidate")),
        |best| Ok(SummationSearchOutcome { best, evaluated }),
    )
}

/// One channel delay entry with its sample-rate binding.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayEntry {
    /// Channel label, for example `"sub-1"`.
    pub label: String,
    /// Delay in seconds.
    pub delay_s: f64,
    /// Sample rate in Hz the sample count refers to.
    pub sample_rate_hz: f64,
}

impl DelayEntry {
    /// Delay in samples, rounded to the nearest whole sample.
    pub fn samples(&self) -> i64 {
        (self.delay_s * self.sample_rate_hz).round() as i64
    }
}

/// Explicit delay ledger in seconds and samples.
///
/// Advances (negative relative shifts) are never applied as negative
/// delays: [`DelayLedger::apply_advance`] reduces existing delay first
/// and adds common latency when the channel has nothing left to give.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DelayLedger {
    /// Per-channel delay entries.
    pub entries: Vec<DelayEntry>,
    /// Common latency in seconds shared by all channels.
    pub common_latency_s: f64,
}

/// How an advance was realized without negative delays.
#[derive(Debug, Clone, PartialEq)]
pub enum AdvanceOutcome {
    /// The channel kept a reduced nonnegative delay.
    ReducedExistingDelay {
        /// Channel label.
        label: String,
        /// Delay before the advance in seconds.
        from_s: f64,
        /// Delay after the advance in seconds.
        to_s: f64,
    },
    /// The channel delay hit zero; the remainder became common latency.
    AddedCommonLatency {
        /// Channel label.
        label: String,
        /// Added common latency in seconds.
        added_common_s: f64,
    },
}

impl DelayLedger {
    /// Reject nonfinite delays, sample rates, latency, and empty labels.
    ///
    /// # Errors
    ///
    /// Returns a reason for the first offending entry or latency value.
    pub fn validate(&self) -> Result<(), String> {
        if !self.common_latency_s.is_finite() || self.common_latency_s < 0.0 {
            return Err(String::from(
                "common latency must be finite and nonnegative",
            ));
        }
        for entry in &self.entries {
            if entry.label.trim().is_empty() {
                return Err(String::from("delay entry label must not be empty"));
            }
            if !entry.delay_s.is_finite() || entry.delay_s < 0.0 {
                return Err(format!(
                    "delay for '{}' must be finite and nonnegative",
                    entry.label
                ));
            }
            if !entry.sample_rate_hz.is_finite() || entry.sample_rate_hz <= 0.0 {
                return Err(format!(
                    "sample rate for '{}' must be finite and positive",
                    entry.label
                ));
            }
        }
        Ok(())
    }

    /// Realize a relative advance of one channel without negative delays.
    ///
    /// When the channel holds at least `advance_s` of delay, the delay is
    /// reduced. Otherwise the channel delay is zeroed and the remainder is
    /// added to the common latency shared by all channels.
    ///
    /// # Errors
    ///
    /// Returns a reason for an unknown channel or an invalid advance.
    pub fn apply_advance(
        &mut self,
        channel_label: &str,
        advance_s: f64,
    ) -> Result<AdvanceOutcome, String> {
        self.validate()?;
        if !advance_s.is_finite() || advance_s < 0.0 {
            return Err(String::from("advance must be finite and nonnegative"));
        }
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.label == channel_label)
            .ok_or_else(|| format!("unknown delay channel '{channel_label}'"))?;
        if entry.delay_s >= advance_s {
            let from_s = entry.delay_s;
            entry.delay_s -= advance_s;
            Ok(AdvanceOutcome::ReducedExistingDelay {
                label: channel_label.to_string(),
                from_s,
                to_s: entry.delay_s,
            })
        } else {
            let added_common_s = advance_s - entry.delay_s;
            entry.delay_s = 0.0;
            self.common_latency_s += added_common_s;
            Ok(AdvanceOutcome::AddedCommonLatency {
                label: channel_label.to_string(),
                added_common_s,
            })
        }
    }
}

/// Per-seat responses retained for combined-response replay.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatCombinedInput<'a> {
    /// Seat identifier; one entry must be the MLP.
    pub seat_id: &'a str,
    /// Shared frequency grid in Hz.
    pub freqs: &'a [f64],
    /// Main magnitude in dB and phase in degrees.
    pub main_mag_db: &'a [f64],
    /// Main phase in degrees.
    pub main_phase_deg: &'a [f64],
    /// Sub magnitude in dB and phase in degrees.
    pub sub_mag_db: &'a [f64],
    /// Sub phase in degrees.
    pub sub_phase_deg: &'a [f64],
}

/// Combined-response replay verdict for one seat.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatReplay {
    /// Seat identifier.
    pub seat_id: String,
    /// Recomputed band summation error with the accepted candidate.
    pub band_error: f64,
    /// Whether the replay sits within tolerance.
    pub replayed_ok: bool,
}

/// Re-verify the accepted alignment's combined response at every seat.
///
/// Each seat recomputes `Hsum` from its retained responses with the
/// accepted candidate; the replay passes when the recomputed band error
/// is finite and within `tolerance`. Every accepted alignment must
/// replay combined, at the MLP and elsewhere.
///
/// # Errors
///
/// Returns a reason for an empty seat list, a nonfinite tolerance, or a
/// response-grid failure.
pub fn reverify_combined(
    seats: &[SeatCombinedInput<'_>],
    candidate: &AlignedCandidate,
    band_hz: [f64; 2],
    tolerance: f64,
) -> Result<Vec<SeatReplay>, String> {
    if seats.is_empty() {
        return Err(String::from("re-verification needs at least one seat"));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(String::from("tolerance must be finite and nonnegative"));
    }
    seats
        .iter()
        .map(|seat| {
            let error = evaluate_candidate(
                seat.freqs,
                seat.main_mag_db,
                seat.main_phase_deg,
                seat.sub_mag_db,
                seat.sub_phase_deg,
                band_hz,
                candidate.polarity_inverted,
                candidate.delay_s,
                candidate.gain_db,
            )?;
            Ok(SeatReplay {
                seat_id: seat.seat_id.to_string(),
                band_error: error,
                replayed_ok: error <= tolerance,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_grid(freqs: &[f64], phase_deg: f64) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; freqs.len()], vec![phase_deg; freqs.len()])
    }

    fn delayed_phase(freqs: &[f64], delay_s: f64) -> Vec<f64> {
        freqs.iter().map(|freq| -360.0 * freq * delay_s).collect()
    }

    #[test]
    fn hsum_matches_analytic_construction() {
        let freqs: Vec<f64> = (20..200).map(|h| h as f64).collect();
        let (mag, phase) = flat_grid(&freqs, 0.0);
        let error = evaluate_candidate(
            &freqs,
            &mag,
            &phase,
            &mag,
            &phase,
            [20.0, 199.0],
            false,
            0.0,
            0.0,
        )
        .unwrap();
        assert!(error < 1e-12, "in-phase equal sum must score zero");
    }

    #[test]
    fn opposite_polarity_flags_cancellation() {
        let freqs: Vec<f64> = (20..200).map(|h| h as f64).collect();
        let (mag, phase) = flat_grid(&freqs, 0.0);
        let error = evaluate_candidate(
            &freqs,
            &mag,
            &phase,
            &mag,
            &phase,
            [20.0, 199.0],
            true,
            0.0,
            0.0,
        )
        .unwrap();
        assert!(error > 0.99, "full cancellation must score near one");
    }

    #[test]
    fn single_frequency_match_alone_fails() {
        let freqs: Vec<f64> = (50..500).map(|h| h as f64).collect();
        let (main_mag, _) = flat_grid(&freqs, 0.0);
        // The main lags by the true 1 ms; the search delays the sub to
        // compensate, so the band winner is the true 1 ms.
        let main_phase = delayed_phase(&freqs, 0.001);
        let (sub_mag, sub_phase) = flat_grid(&freqs, 0.0);
        // 11 ms matches 1 ms exactly at 100 Hz (one extra cycle) but slips
        // across the band; the band search must prefer the true 1 ms.
        let single = evaluate_candidate(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            [99.5, 100.5],
            false,
            0.011,
            0.0,
        )
        .unwrap();
        let single_true = evaluate_candidate(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            [99.5, 100.5],
            false,
            0.001,
            0.0,
        )
        .unwrap();
        assert!(
            single < 1e-9 && single_true < 1e-9,
            "both aliases agree at 100 Hz alone"
        );
        let grid = SearchGrid {
            delays_s: vec![0.001, 0.011],
            gains_db: vec![0.0],
            include_polarity_inversion: false,
        };
        let outcome = search_summation(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            [50.0, 499.0],
            &grid,
        )
        .unwrap();
        assert!((outcome.best.delay_s - 0.001).abs() < 1e-12);
    }

    #[test]
    fn twenty_ms_ambiguity_resolved_by_band() {
        let freqs: Vec<f64> = (40..120).map(|h| h as f64).collect();
        let (main_mag, _) = flat_grid(&freqs, 0.0);
        let main_phase = delayed_phase(&freqs, 0.005);
        let (sub_mag, sub_phase) = flat_grid(&freqs, 0.0);
        // 25 ms differs from 5 ms by exactly one 50 Hz period: identical at
        // 50 Hz alone, divergent across the band.
        let grid = SearchGrid {
            delays_s: vec![0.005, 0.025],
            gains_db: vec![0.0],
            include_polarity_inversion: false,
        };
        let outcome = search_summation(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            [40.0, 119.0],
            &grid,
        )
        .unwrap();
        assert!((outcome.best.delay_s - 0.005).abs() < 1e-12);
        assert_eq!(outcome.evaluated, 2);
    }

    #[test]
    fn delay_ledger_reports_seconds_and_samples() {
        let entry = DelayEntry {
            label: String::from("sub-1"),
            delay_s: 0.005,
            sample_rate_hz: 48_000.0,
        };
        assert_eq!(entry.samples(), 240);
    }

    #[test]
    fn advance_reduces_existing_delay_first() {
        let mut ledger = DelayLedger {
            entries: vec![DelayEntry {
                label: String::from("sub-1"),
                delay_s: 0.010,
                sample_rate_hz: 48_000.0,
            }],
            common_latency_s: 0.0,
        };
        let outcome = ledger.apply_advance("sub-1", 0.003).unwrap();
        assert_eq!(
            outcome,
            AdvanceOutcome::ReducedExistingDelay {
                label: String::from("sub-1"),
                from_s: 0.010,
                to_s: 0.007,
            }
        );
        assert!((ledger.entries[0].delay_s - 0.007).abs() < 1e-12);
        assert_eq!(ledger.common_latency_s, 0.0);
    }

    #[test]
    fn advance_beyond_delay_adds_common_latency() {
        let mut ledger = DelayLedger {
            entries: vec![DelayEntry {
                label: String::from("sub-1"),
                delay_s: 0.002,
                sample_rate_hz: 48_000.0,
            }],
            common_latency_s: 0.0,
        };
        let outcome = ledger.apply_advance("sub-1", 0.005).unwrap();
        assert_eq!(
            outcome,
            AdvanceOutcome::AddedCommonLatency {
                label: String::from("sub-1"),
                added_common_s: 0.003,
            }
        );
        assert_eq!(ledger.entries[0].delay_s, 0.0);
        assert!((ledger.common_latency_s - 0.003).abs() < 1e-12);
    }

    #[test]
    fn accepted_alignment_replays_combined() {
        let freqs: Vec<f64> = (40..120).map(|h| h as f64).collect();
        let (main_mag, _) = flat_grid(&freqs, 0.0);
        let main_phase = delayed_phase(&freqs, 0.005);
        let (sub_mag, sub_phase) = flat_grid(&freqs, 0.0);
        let grid = SearchGrid {
            delays_s: vec![0.005, 0.025],
            gains_db: vec![0.0],
            include_polarity_inversion: true,
        };
        let outcome = search_summation(
            &freqs,
            &main_mag,
            &main_phase,
            &sub_mag,
            &sub_phase,
            [40.0, 119.0],
            &grid,
        )
        .unwrap();
        let seats = vec![
            SeatCombinedInput {
                seat_id: "mlp",
                freqs: &freqs,
                main_mag_db: &main_mag,
                main_phase_deg: &main_phase,
                sub_mag_db: &sub_mag,
                sub_phase_deg: &sub_phase,
            },
            SeatCombinedInput {
                seat_id: "seat-b",
                freqs: &freqs,
                main_mag_db: &main_mag,
                main_phase_deg: &main_phase,
                sub_mag_db: &sub_mag,
                sub_phase_deg: &sub_phase,
            },
            SeatCombinedInput {
                seat_id: "seat-c",
                freqs: &freqs,
                main_mag_db: &main_mag,
                main_phase_deg: &main_phase,
                sub_mag_db: &sub_mag,
                sub_phase_deg: &sub_phase,
            },
        ];
        let replays = reverify_combined(&seats, &outcome.best, [40.0, 119.0], 1e-6).unwrap();
        assert_eq!(replays.len(), 3);
        assert!(replays.iter().all(|replay| replay.replayed_ok));
        let wrong = AlignedCandidate {
            polarity_inverted: true,
            delay_s: outcome.best.delay_s,
            gain_db: 0.0,
            band_error: 0.0,
        };
        let bad = reverify_combined(&seats, &wrong, [40.0, 119.0], 1e-6).unwrap();
        assert!(bad.iter().all(|replay| !replay.replayed_ok));
    }
}
