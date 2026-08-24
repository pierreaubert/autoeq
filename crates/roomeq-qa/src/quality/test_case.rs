use super::option_override::OptionOverride;
use crate::registry::ScenarioExpect;

pub(super) struct RegisteredTestCase {
    pub(super) id: String,
    pub(super) claims: Vec<String>,
    pub(super) expect: ScenarioExpect,
    pub(super) case: TestCase,
}

#[derive(Clone)]
pub(super) enum TestCase {
    /// Stereo/Home Cinema workflow test (IIR mutations)
    Workflow {
        name: String,
        fem_subdir: String,
        optim_subdir: String,
    },
    /// Generic path test (all 3 modes: IIR, FIR, Mixed)
    Generic {
        name: String,
        fem_subdir: String,
        optim_subdir: String,
    },
    /// Cross-mode convergence: IIR vs FIR vs Mixed frequency response similarity
    CrossModeConvergence {
        name: String,
        fem_subdir: String,
        optim_subdir: String,
        config_path: Option<String>,
        override_dir: Option<String>,
        preserve_system: bool,
        strict: bool,
    },
    /// Per-option A/B test: baseline vs with-option(s)
    /// Supports single options and combinations.
    OptionEffect {
        name: String,
        fem_subdir: String,
        optim_subdir: String,
        options: Vec<OptionOverride>,
    },
}

impl TestCase {
    pub(super) fn name(&self) -> &str {
        match self {
            TestCase::Workflow { name, .. } => name,
            TestCase::Generic { name, .. } => name,
            TestCase::CrossModeConvergence { name, .. } => name,
            TestCase::OptionEffect { name, .. } => name,
        }
    }
}

impl RegisteredTestCase {
    pub(super) fn name(&self) -> &str {
        self.case.name()
    }
}
