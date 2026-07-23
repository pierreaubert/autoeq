mod misc;
#[cfg(test)]
mod tests;
mod try_;

pub(in crate::room_optimization) use misc::source_for_output_channel;
pub(in crate::room_optimization) use try_::*;
