//! Specific optimization workflows for different system topologies.

use super::crossover;
use super::dba;
use super::types::{
    CardioidConfig, CrossoverConfig, DBAConfig, MultiSubGroup, RoomConfig, SpeakerConfig,
    SubwooferStrategy, SystemConfig,
};
use crate::Curve;
use crate::error::{AutoeqError, Result};
use crate::read::load_source;
use log::info;
use math_audio_dsp::analysis::compute_average_response;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, HashMap};

mod bass_management;

mod all;
mod apply;
mod bass;
mod compute;
#[cfg(test)]
mod executor_tests;
mod generic;
mod home_cinema;
mod mark;
mod misc;
mod multiseat;
mod multisub;
mod optimize;
mod run;
mod stereo;
mod stereo_sub;
#[cfg(test)]
mod tests;
mod types;
mod workflow;

mod supporting_source;

pub use misc::*;
pub use optimize::*;
pub(in crate::roomeq) use types::*;
