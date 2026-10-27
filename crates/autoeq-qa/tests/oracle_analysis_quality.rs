//! Wolfram oracle group for analysis, quality, and model cases.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_ra01_hybrid_grid.rs"]
mod wolfram_ra01_hybrid_grid;

#[path = "wolfram_ra01_log_slope.rs"]
mod wolfram_ra01_log_slope;

#[path = "wolfram_ra01_lr24_crossover.rs"]
mod wolfram_ra01_lr24_crossover;

#[path = "wolfram_ra02_arrival_alignment.rs"]
mod wolfram_ra02_arrival_alignment;

#[path = "wolfram_ra02_phase_slope_delay.rs"]
mod wolfram_ra02_phase_slope_delay;

#[path = "wolfram_ra02_probe_delay.rs"]
mod wolfram_ra02_probe_delay;

#[path = "wolfram_ra03_mode_null_masks.rs"]
mod wolfram_ra03_mode_null_masks;

#[path = "wolfram_ra04_decay_severity.rs"]
mod wolfram_ra04_decay_severity;

#[path = "wolfram_ra05_delay_excess_phase.rs"]
mod wolfram_ra05_delay_excess_phase;

#[path = "wolfram_ra06_reflection_cancel.rs"]
mod wolfram_ra06_reflection_cancel;

#[path = "wolfram_ra07_spatial_stats.rs"]
mod wolfram_ra07_spatial_stats;

#[path = "wolfram_ra08_area_interp.rs"]
mod wolfram_ra08_area_interp;

#[path = "wolfram_ra09_rir_weights.rs"]
mod wolfram_ra09_rir_weights;

#[path = "wolfram_ra10_evidence_stats.rs"]
mod wolfram_ra10_evidence_stats;

#[path = "wolfram_rm01_schroeder_spl.rs"]
mod wolfram_rm01_schroeder_spl;

#[path = "wolfram_rm01_spl_level.rs"]
mod wolfram_rm01_spl_level;

#[path = "wolfram_rm02_target_shape.rs"]
mod wolfram_rm02_target_shape;

#[path = "wolfram_rm03_auto_tune.rs"]
mod wolfram_rm03_auto_tune;

#[path = "wolfram_rm03_band_gauss.rs"]
mod wolfram_rm03_band_gauss;

#[path = "wolfram_rm04_phase_thresholds.rs"]
mod wolfram_rm04_phase_thresholds;

#[path = "wolfram_rq01_seat_bin_cvar.rs"]
mod wolfram_rq01_seat_bin_cvar;

#[path = "wolfram_rq02_step_schroeder_t60.rs"]
mod wolfram_rq02_step_schroeder_t60;

#[path = "wolfram_rq03_welch_noise_spl.rs"]
mod wolfram_rq03_welch_noise_spl;

#[path = "wolfram_rq04_headroom_drive_verdicts.rs"]
mod wolfram_rq04_headroom_drive_verdicts;

#[path = "wolfram_rq05_capture_alignment.rs"]
mod wolfram_rq05_capture_alignment;

#[path = "wolfram_rq06_tone_render.rs"]
mod wolfram_rq06_tone_render;

#[path = "wolfram_rq07_abx_binomial.rs"]
mod wolfram_rq07_abx_binomial;

#[path = "wolfram_rq08_delay_oracle.rs"]
mod wolfram_rq08_delay_oracle;
