//! Binary execution seam for integration tests.
//!
//! Integration tests need to spawn compiled binaries (`autoeq`, `roomeq`, ...).
//! [`BinaryRunner`] abstracts that operation so tests can later be run against
//! an in-process implementation or a deterministic fake without rewriting every
//! test case.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Command, Output};

/// Abstraction over running a named binary with CLI arguments.
pub trait BinaryRunner {
    /// Run `binary_name` with `args` and return the captured output.
    fn run(&self, binary_name: &str, args: &[&str]) -> std::io::Result<Output>;
}

/// Default runner that spawns real processes from the cargo target directory.
///
/// The binary directory is derived from `CARGO_BIN_EXE_autoeq`, so the runner
/// works for any binary built in the same cargo invocation.
#[derive(Debug, Clone)]
pub struct ProcessBinaryRunner {
    bin_dir: PathBuf,
}

impl ProcessBinaryRunner {
    /// Create a runner pointing at the cargo target binary directory.
    pub fn new() -> Self {
        let autoeq_bin = PathBuf::from(env!("CARGO_BIN_EXE_autoeq"));
        Self {
            bin_dir: autoeq_bin.parent().unwrap().to_path_buf(),
        }
    }

    /// Convenience: run the `autoeq` binary.
    pub fn run_autoeq(&self, args: &[&str]) -> Output {
        self.run("autoeq", args)
            .expect("failed to execute autoeq binary")
    }

    /// Convenience: run the `roomeq` binary.
    pub fn run_roomeq(&self, args: &[&str]) -> Output {
        self.run("roomeq", args)
            .expect("failed to execute roomeq binary")
    }
}

impl Default for ProcessBinaryRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryRunner for ProcessBinaryRunner {
    fn run(&self, binary_name: &str, args: &[&str]) -> std::io::Result<Output> {
        Command::new(self.bin_dir.join(binary_name))
            .args(args)
            .output()
    }
}

/// Deterministic fake runner for tests that do not need real binary execution.
///
/// Configure it with a map from `(binary_name, args)` to a pre-canned output.
/// Any unconfigured invocation returns an error output so tests fail visibly.
#[derive(Debug, Default, Clone)]
pub struct MockBinaryRunner {
    /// Key: binary name. Value: list of (argument list, desired output).
    pub responses: std::collections::HashMap<String, Vec<(Vec<String>, std::process::Output)>>,
}

impl MockBinaryRunner {
    /// Register a response for a given binary and argument list.
    pub fn when(
        mut self,
        binary_name: impl Into<String>,
        args: &[&str],
        output: std::process::Output,
    ) -> Self {
        self.responses
            .entry(binary_name.into())
            .or_default()
            .push((args.iter().map(|s| (*s).to_string()).collect(), output));
        self
    }
}

impl BinaryRunner for MockBinaryRunner {
    fn run(&self, binary_name: &str, args: &[&str]) -> std::io::Result<Output> {
        match self.responses.get(binary_name) {
            Some(list) => {
                let needle: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
                for (stored_args, output) in list {
                    if stored_args == &needle {
                        return Ok(output.clone());
                    }
                }
                Ok(std::process::Output {
                    status: std::process::ExitStatus::default(),
                    stdout: vec![],
                    stderr: format!(
                        "MockBinaryRunner: no response registered for {} {:?}",
                        binary_name, args
                    )
                    .into_bytes(),
                })
            }
            None => Ok(std::process::Output {
                status: std::process::ExitStatus::default(),
                stdout: vec![],
                stderr: format!(
                    "MockBinaryRunner: no responses registered for {}",
                    binary_name
                )
                .into_bytes(),
            }),
        }
    }
}

/// Convenience: run the `autoeq` binary using the default process runner.
pub fn run_autoeq(args: &[&str]) -> Output {
    ProcessBinaryRunner::new().run_autoeq(args)
}

/// Convenience: run the `roomeq` binary using the default process runner.
pub fn run_roomeq(args: &[&str]) -> Output {
    ProcessBinaryRunner::new().run_roomeq(args)
}
