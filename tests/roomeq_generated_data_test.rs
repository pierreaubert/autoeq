//! Integration tests using FEM-generated room measurement data
//!
//! These tests load pre-computed FEM simulation data and verify that
//! the roomeq optimizer can improve the simulated frequency responses.
//!
//! The smallest stereo convergence case runs in the default suite. Run the
//! complete long-running convergence and multimode matrix with:
//! `cargo test -p autoeq --test roomeq_generated_data_test -- --ignored --test-threads=1`

#[path = "roomeq_generated_data_test/consts.rs"]
mod consts;
#[path = "roomeq_generated_data_test/misc.rs"]
mod misc;
#[path = "roomeq_generated_data_test/run.rs"]
mod run;
#[path = "roomeq_generated_data_test/types.rs"]
mod types;
