//! Wolfram oracle group for core, FIR, artifacts, and measurements.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_ar01_impulse_store.rs"]
mod wolfram_ar01_impulse_store;

#[path = "wolfram_c02_free_types.rs"]
mod wolfram_c02_free_types;

#[path = "wolfram_c02_peq_layouts.rs"]
mod wolfram_c02_peq_layouts;

#[path = "wolfram_c03_logmean_slope.rs"]
mod wolfram_c03_logmean_slope;

#[path = "wolfram_c04_gaussian_smooth.rs"]
mod wolfram_c04_gaussian_smooth;

#[path = "wolfram_c04_octave_smooth.rs"]
mod wolfram_c04_octave_smooth;

#[path = "wolfram_c05_minphase_flat.rs"]
mod wolfram_c05_minphase_flat;

#[path = "wolfram_c05_unwrap_delay.rs"]
mod wolfram_c05_unwrap_delay;

#[path = "wolfram_c07_quality_geometry.rs"]
mod wolfram_c07_quality_geometry;

#[path = "wolfram_c08_local_q.rs"]
mod wolfram_c08_local_q;

#[path = "wolfram_erb_quadrature.rs"]
mod wolfram_erb_quadrature;

#[path = "wolfram_f01_fir_flat_designs.rs"]
mod wolfram_f01_fir_flat_designs;

#[path = "wolfram_f02_target_grid_wav.rs"]
mod wolfram_f02_target_grid_wav;

#[path = "wolfram_fir_response.rs"]
mod wolfram_fir_response;

#[path = "wolfram_log_interp.rs"]
mod wolfram_log_interp;

#[path = "wolfram_m01_pir_prefscore.rs"]
mod wolfram_m01_pir_prefscore;

#[path = "wolfram_m02_avg_coherent.rs"]
mod wolfram_m02_avg_coherent;

#[path = "wolfram_m03_mic_clock.rs"]
mod wolfram_m03_mic_clock;

#[path = "wolfram_m04_ingest_adapters.rs"]
mod wolfram_m04_ingest_adapters;

#[path = "wolfram_peq_response.rs"]
mod wolfram_peq_response;
