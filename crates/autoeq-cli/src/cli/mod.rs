//! AutoEQ - A library for audio equalization and filter optimization
//! Common command-line interface definitions shared across binaries
//!
//! Copyright (C) 2025-2026 Pierre Aubert pierre(at)spinorama(dot)org
//!
//! This program is free software: you can redistribute it and/or modify
//! it under the terms of the GNU General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! This program is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU General Public License for more details.
//!
//! You should have received a copy of the GNU General Public License
//! along with this program.  If not, see <https://www.gnu.org/licenses/>.

mod args;
mod display;
mod enums;
mod parse;
mod peq_model;
mod validate;

pub use args::Args;
pub use display::{
    display_algorithm_list, display_peq_model_list, display_strategy_list, render_algorithm_list,
    render_peq_model_list, render_strategy_list,
};
pub use peq_model::PeqModel;
pub use validate::{validate_args, validate_args_or_exit};
