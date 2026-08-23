use super::consts::QA_MAXEVAL;
use super::misc::num_cpus;
use crate::registry::QaTier;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "roomeq-qa-coverage")]
#[command(about = "Comprehensive RoomEQ QA with full scenario matrix")]
pub(super) struct Args {
    /// Declarative registry tier (pr, nightly, or weekly)
    #[arg(long, value_enum, default_value = "weekly")]
    pub(super) tier: QaTier,

    /// Run the bounded FEM/IIR smoke suite with safety-level acceptance
    #[arg(long)]
    pub(super) quick: bool,

    /// List all available scenarios
    #[arg(long)]
    pub(super) list: bool,

    /// Show test matrix without running
    #[arg(long)]
    pub(super) matrix: bool,

    /// Run only the correctness-gated home-cinema feature matrix
    #[arg(long)]
    pub(super) home_cinema: bool,

    /// Output JUnit XML to file
    #[arg(long)]
    pub(super) junit: Option<PathBuf>,

    /// Filter by scenario name (substring match)
    #[arg(long)]
    pub(super) scenario: Option<String>,

    /// Filter by test-case name (substring match)
    #[arg(long = "case")]
    pub(super) case_name: Option<String>,

    /// Filter by solver (`fem` or `fast-hybrid`)
    #[arg(long)]
    pub(super) solver: Option<String>,

    /// Filter by mode (iir, fir, mixed, or all)
    #[arg(long)]
    pub(super) mode: Option<String>,

    /// Number of parallel jobs (default: num CPUs)
    #[arg(long)]
    pub(super) jobs: Option<usize>,

    /// Maximum evaluations per optimization (default: 500)
    #[arg(long)]
    pub(super) maxeval: Option<usize>,

    /// Fail if any test fails (default: true, use --no-fail to disable)
    #[arg(long = "no-fail", alias = "fail", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub(super) fail: bool,
}

impl Args {
    pub(super) fn maxeval(&self) -> usize {
        self.maxeval.unwrap_or(QA_MAXEVAL)
    }

    pub(super) fn jobs(&self) -> usize {
        self.jobs.unwrap_or(num_cpus())
    }
}

#[cfg(test)]
mod tests {
    use crate::registry::QaTier;
    use clap::Parser;

    use super::Args;
    use super::QA_MAXEVAL;

    #[test]
    fn defaults_are_expected() {
        let args = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        assert!(args.junit.is_none());
        assert_eq!(args.tier, QaTier::Weekly);
        assert!(!args.home_cinema);
        assert!(args.scenario.is_none());
        assert!(args.case_name.is_none());
        assert!(args.solver.is_none());
        assert!(args.mode.is_none());
        assert!(args.jobs.is_none());
        assert!(args.maxeval.is_none());
        assert!(args.fail);
    }

    #[test]
    fn no_fail_flag_disables_fail() {
        let args = Args::try_parse_from(["roomeq-qa-coverage", "--no-fail"]).unwrap();
        assert!(!args.fail);
    }

    #[test]
    fn maxeval_fallback_uses_const_default() {
        let args = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        assert_eq!(args.maxeval(), QA_MAXEVAL);
    }

    #[test]
    fn maxeval_flag_overrides_fallback() {
        let args = Args::try_parse_from(["roomeq-qa-coverage", "--maxeval", "1234"]).unwrap();
        assert_eq!(args.maxeval(), 1234);
    }

    #[test]
    fn jobs_fallback_uses_num_cpus() {
        let args = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        let expected = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        assert_eq!(args.jobs(), expected);
    }

    #[test]
    fn jobs_flag_overrides_fallback() {
        let args = Args::try_parse_from(["roomeq-qa-coverage", "--jobs", "2"]).unwrap();
        assert_eq!(args.jobs(), 2);
    }
}
