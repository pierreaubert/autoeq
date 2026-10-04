//! Program-version handling for the root-package command launchers.

use clap::Parser;
use std::ffi::OsString;

/// Print the AutoEQ executable version when the shared CLI flag is present.
///
/// The version is supplied by the root-package launcher so it identifies the
/// installed executable rather than this adapter crate or the optimizer crate.
pub fn print_autoeq_version_if_requested(version: &str) -> bool {
    if !autoeq_version_requested(std::env::args_os()) {
        return false;
    }
    println!("autoeq {version}");
    true
}

/// Print the speaker-benchmark executable version when requested.
pub fn print_benchmark_version_if_requested(version: &str) -> bool {
    if !benchmark_version_requested(std::env::args_os()) {
        return false;
    }
    println!("benchmark-autoeq-speaker {version}");
    true
}

fn autoeq_version_requested(args: impl IntoIterator<Item = OsString>) -> bool {
    autoeq::cli::Args::try_parse_from(args).is_ok_and(|parsed| parsed.program_version)
}

fn benchmark_version_requested(args: impl IntoIterator<Item = OsString>) -> bool {
    crate::benchmark::BenchArgs::try_parse_from(args)
        .is_ok_and(|parsed| parsed.base.program_version)
}

#[cfg(test)]
mod tests {
    use super::{autoeq_version_requested, benchmark_version_requested};
    use std::ffi::OsString;

    fn argv(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn autoeq_program_version_is_additive_to_api_version() {
        assert!(autoeq_version_requested(argv(&[
            "autoeq",
            "--program-version"
        ])));
        assert!(autoeq_version_requested(argv(&["autoeq", "-V"])));
        assert!(!autoeq_version_requested(argv(&[
            "autoeq",
            "--version=--program-version",
            "--curve",
            "response.csv",
        ])));
        assert!(!autoeq_version_requested(argv(&[
            "autoeq",
            "--",
            "--program-version",
        ])));
        assert!(!autoeq_version_requested(argv(&[
            "autoeq",
            "--version",
            "2026-10"
        ])));
    }

    #[test]
    fn benchmark_program_version_uses_the_flattened_shared_flag() {
        assert!(benchmark_version_requested(argv(&[
            "benchmark-autoeq-speaker",
            "--program-version",
            "--jobs",
            "2",
        ])));
        assert!(benchmark_version_requested(argv(&[
            "benchmark-autoeq-speaker",
            "-V"
        ])));
        assert!(!benchmark_version_requested(argv(&[
            "benchmark-autoeq-speaker",
            "--",
            "--program-version",
        ])));
    }
}
