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

    /// Output machine-readable QA verdicts and delivered stage evidence
    #[arg(long)]
    pub(super) json: Option<PathBuf>,

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

    /// Optimizer seeds per case: 1 (deterministic contract) or 5
    /// (median-of-five convergence distribution). Defaults to 1 for the
    /// PR tier and 5 for nightly/weekly tiers.
    #[arg(long, value_parser = parse_seed_runs)]
    pub(super) seed_runs: Option<usize>,

    /// Force metamorphic budget/filter replays on for every case.
    /// Defaults to on for nightly/weekly tiers and off for the PR tier.
    #[arg(long, conflicts_with = "no_metamorphic")]
    pub(super) metamorphic: bool,

    /// Force metamorphic budget/filter replays off for every case.
    #[arg(long = "no-metamorphic")]
    pub(super) no_metamorphic: bool,

    /// Stop scheduling new cases after the first failure. Cases already
    /// running still complete and report.
    #[arg(long)]
    pub(super) fail_fast: bool,

    /// Fail if any test fails (default: true, use --no-fail to disable)
    #[arg(long = "no-fail", alias = "fail", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub(super) fail: bool,
}

fn parse_seed_runs(raw: &str) -> Result<usize, String> {
    let seed_runs = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid --seed-runs value '{raw}': expected 1 or 5"))?;
    match seed_runs {
        1 | 5 => Ok(seed_runs),
        _ => Err(format!(
            "invalid --seed-runs value '{raw}': must be 1 (deterministic contract) or 5 (convergence distribution)"
        )),
    }
}

impl Args {
    pub(super) fn maxeval(&self) -> usize {
        self.maxeval.unwrap_or(QA_MAXEVAL)
    }

    pub(super) fn jobs(&self) -> usize {
        self.jobs.unwrap_or(num_cpus())
    }

    /// Resolved seed population per case. The PR tier is a deterministic
    /// single-seed contract; nightly and weekly tiers retain the
    /// median-of-five convergence distribution. `--seed-runs` overrides.
    pub(super) fn seed_runs(&self) -> usize {
        self.seed_runs.unwrap_or(match self.tier {
            QaTier::Pr => 1,
            QaTier::Nightly | QaTier::Weekly => 5,
        })
    }

    /// Resolved metamorphic replay setting. The doubled-budget and
    /// added-filter replays multiply per-case cost, so they run in the
    /// nightly and weekly tiers by default. `--metamorphic` and
    /// `--no-metamorphic` override.
    pub(super) fn metamorphic(&self) -> bool {
        if self.metamorphic {
            return true;
        }
        if self.no_metamorphic {
            return false;
        }
        match self.tier {
            QaTier::Pr => false,
            QaTier::Nightly | QaTier::Weekly => true,
        }
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

    #[test]
    fn seed_runs_defaults_by_tier() {
        let pr = Args::try_parse_from(["roomeq-qa-coverage", "--tier", "pr"]).unwrap();
        assert_eq!(pr.seed_runs(), 1);
        let nightly = Args::try_parse_from(["roomeq-qa-coverage", "--tier", "nightly"]).unwrap();
        assert_eq!(nightly.seed_runs(), 5);
        let weekly = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        assert_eq!(weekly.seed_runs(), 5);
    }

    #[test]
    fn seed_runs_flag_accepts_one_or_five_only() {
        let one = Args::try_parse_from(["roomeq-qa-coverage", "--seed-runs", "1"]).unwrap();
        assert_eq!(one.seed_runs(), 1);
        let five = Args::try_parse_from(["roomeq-qa-coverage", "--tier", "pr", "--seed-runs", "5"])
            .unwrap();
        assert_eq!(five.seed_runs(), 5);
        for invalid in ["0", "2", "3", "6", "five"] {
            assert!(
                Args::try_parse_from(["roomeq-qa-coverage", "--seed-runs", invalid]).is_err(),
                "accepted --seed-runs {invalid}"
            );
        }
    }

    #[test]
    fn metamorphic_defaults_by_tier_and_flags_override() {
        let pr = Args::try_parse_from(["roomeq-qa-coverage", "--tier", "pr"]).unwrap();
        assert!(!pr.metamorphic());
        let weekly = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        assert!(weekly.metamorphic());
        let forced =
            Args::try_parse_from(["roomeq-qa-coverage", "--tier", "pr", "--metamorphic"]).unwrap();
        assert!(forced.metamorphic());
        let skipped = Args::try_parse_from(["roomeq-qa-coverage", "--no-metamorphic"]).unwrap();
        assert!(!skipped.metamorphic());
        assert!(
            Args::try_parse_from(["roomeq-qa-coverage", "--metamorphic", "--no-metamorphic"])
                .is_err()
        );
    }

    #[test]
    fn fail_fast_defaults_off() {
        let args = Args::try_parse_from(["roomeq-qa-coverage"]).unwrap();
        assert!(!args.fail_fast);
        let fast = Args::try_parse_from(["roomeq-qa-coverage", "--fail-fast"]).unwrap();
        assert!(fast.fail_fast);
    }
}
