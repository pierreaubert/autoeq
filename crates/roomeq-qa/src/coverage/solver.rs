use super::consts::FEM_DIR;

const FAST_HYBRID_DIR: &str = "data_tests/roomeq/generate/fast-hybrid";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Solver {
    Fem,
    FastHybrid,
}

impl Solver {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Solver::Fem => "fem",
            Solver::FastHybrid => "fast-hybrid",
        }
    }

    pub(super) fn dir(&self) -> &'static str {
        match self {
            Solver::Fem => FEM_DIR,
            Solver::FastHybrid => FAST_HYBRID_DIR,
        }
    }
}
