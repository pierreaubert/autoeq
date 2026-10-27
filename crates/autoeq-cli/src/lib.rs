//! AutoEQ command-line adapters.

extern crate autoeq_workflow as autoeq;

pub mod autoeq_command;
pub mod benchmark;
pub mod download;
pub mod program_version;

pub mod cli;
pub use autoeq_workflow::qa_println;
pub use cli::{
    Args, PeqModel, display_algorithm_list, display_peq_model_list, display_strategy_list,
    render_algorithm_list, render_peq_model_list, render_strategy_list, validate_args,
    validate_args_or_exit,
};
