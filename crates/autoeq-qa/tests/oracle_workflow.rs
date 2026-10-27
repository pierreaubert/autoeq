//! Wolfram oracle group for workflow, CLI, plotting, and QA cases.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_ac01_cli_spacing_score.rs"]
mod wolfram_ac01_cli_spacing_score;

#[path = "wolfram_ac02_apo_roundtrip_gap.rs"]
mod wolfram_ac02_apo_roundtrip_gap;

#[path = "wolfram_ac03_benchmark_stats.rs"]
mod wolfram_ac03_benchmark_stats;

#[path = "wolfram_h01_fir_bin_rounding.rs"]
mod wolfram_h01_fir_bin_rounding;

#[path = "wolfram_pl01_tonal_regression.rs"]
mod wolfram_pl01_tonal_regression;

#[path = "wolfram_pl02_plot_response_trace.rs"]
mod wolfram_pl02_plot_response_trace;

#[path = "wolfram_qa01_clock_phase.rs"]
mod wolfram_qa01_clock_phase;

#[path = "wolfram_qa02_heldout_perturb.rs"]
mod wolfram_qa02_heldout_perturb;

#[path = "wolfram_qa03_seed_matrix.rs"]
mod wolfram_qa03_seed_matrix;

#[path = "wolfram_rc01_wav_decode_dtft_calibration.rs"]
mod wolfram_rc01_wav_decode_dtft_calibration;

#[path = "wolfram_rc02_route_matrix_prediction.rs"]
mod wolfram_rc02_route_matrix_prediction;

#[path = "wolfram_rw01_intake_overlap.rs"]
mod wolfram_rw01_intake_overlap;

#[path = "wolfram_rw02_windowed_matrix_dft.rs"]
mod wolfram_rw02_windowed_matrix_dft;

#[path = "wolfram_rw03_dsp_fir_convolution.rs"]
mod wolfram_rw03_dsp_fir_convolution;

#[path = "wolfram_rw04_crossover_bounded_sum.rs"]
mod wolfram_rw04_crossover_bounded_sum;

#[path = "wolfram_rw05_electrical_headroom.rs"]
mod wolfram_rw05_electrical_headroom;

#[path = "wolfram_rw06_velvet_support.rs"]
mod wolfram_rw06_velvet_support;

#[path = "wolfram_rw07_scorecard_pruning.rs"]
mod wolfram_rw07_scorecard_pruning;

#[path = "wolfram_rw08_stimulus_binding.rs"]
mod wolfram_rw08_stimulus_binding;

#[path = "wolfram_rw09_scaled_replay.rs"]
mod wolfram_rw09_scaled_replay;

#[path = "wolfram_rw10_from_measurement_slope.rs"]
mod wolfram_rw10_from_measurement_slope;

#[path = "wolfram_rw11_hybrid_grid_delay.rs"]
mod wolfram_rw11_hybrid_grid_delay;
