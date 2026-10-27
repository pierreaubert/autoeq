//! Wolfram oracle group for export cases.
//!
//! Grouped test target (refactor-20261025 Phase 2): each module below
//! is one manifest case file kept at `tests/<module>.rs`. The case ->
//! target/module mapping is declared in `validation-manifest.toml`
//! (`test` names the module, `target` names this group) and enforced
//! by `qa_contract` and `scripts/run_crate_test_matrix.py`.

#[path = "wolfram_ex01_apo_biquad_gain_delay.rs"]
mod wolfram_ex01_apo_biquad_gain_delay;

#[path = "wolfram_ex02_delay_padding_matrix.rs"]
mod wolfram_ex02_delay_padding_matrix;

#[path = "wolfram_ex03_rendered_convolution_taps.rs"]
mod wolfram_ex03_rendered_convolution_taps;
