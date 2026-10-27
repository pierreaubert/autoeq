//! Wolfram oracle group for the RoomEQ engine and synthetic cases.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_re01_dsp_gain_delay.rs"]
mod wolfram_re01_dsp_gain_delay;

#[path = "wolfram_re02_preference_shelves.rs"]
mod wolfram_re02_preference_shelves;

#[path = "wolfram_re03_candidate_loss.rs"]
mod wolfram_re03_candidate_loss;

#[path = "wolfram_re04_pruning_cumulative.rs"]
mod wolfram_re04_pruning_cumulative;

#[path = "wolfram_re05_fir_residual.rs"]
mod wolfram_re05_fir_residual;

#[path = "wolfram_re06_fir_integer_delay.rs"]
mod wolfram_re06_fir_integer_delay;

#[path = "wolfram_re07_null_mask.rs"]
mod wolfram_re07_null_mask;

#[path = "wolfram_re08_split_band_sum.rs"]
mod wolfram_re08_split_band_sum;

#[path = "wolfram_re09_kautz_bank.rs"]
mod wolfram_re09_kautz_bank;

#[path = "wolfram_re10_branch_sum_levels.rs"]
mod wolfram_re10_branch_sum_levels;

#[path = "wolfram_re11_summation_lattice.rs"]
mod wolfram_re11_summation_lattice;

#[path = "wolfram_re12_gd_coherence.rs"]
mod wolfram_re12_gd_coherence;

#[path = "wolfram_re13_mso_seat_response.rs"]
mod wolfram_re13_mso_seat_response;

#[path = "wolfram_re14_combined_curves.rs"]
mod wolfram_re14_combined_curves;

#[path = "wolfram_re15_modal_svd.rs"]
mod wolfram_re15_modal_svd;

#[path = "wolfram_re16_spatial_quadrature.rs"]
mod wolfram_re16_spatial_quadrature;

#[path = "wolfram_re17_interchannel_deviation.rs"]
mod wolfram_re17_interchannel_deviation;

#[path = "wolfram_re18_dba_sum.rs"]
mod wolfram_re18_dba_sum;

#[path = "wolfram_re19_bass_power.rs"]
mod wolfram_re19_bass_power;

#[path = "wolfram_re20_drr_velvet.rs"]
mod wolfram_re20_drr_velvet;

#[path = "wolfram_re21_eq_response_correction.rs"]
mod wolfram_re21_eq_response_correction;

#[path = "wolfram_re22_limiter_gain_envelope.rs"]
mod wolfram_re22_limiter_gain_envelope;

#[path = "wolfram_re23_ctc_regularized.rs"]
mod wolfram_re23_ctc_regularized;

#[path = "wolfram_re24_cardioid.rs"]
mod wolfram_re24_cardioid;

#[path = "wolfram_root01_facade_passthrough.rs"]
mod wolfram_root01_facade_passthrough;

#[path = "wolfram_rs01_flat_tilt.rs"]
mod wolfram_rs01_flat_tilt;

#[path = "wolfram_rs02_coherent_sum.rs"]
mod wolfram_rs02_coherent_sum;

#[path = "wolfram_rs03_tone_bank.rs"]
mod wolfram_rs03_tone_bank;
