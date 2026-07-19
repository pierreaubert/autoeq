use super::consts::FEM_DIR;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Solver {
    Fem,
}

impl Solver {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Solver::Fem => "fem",
        }
    }

    pub(super) fn dir(&self) -> &'static str {
        match self {
            Solver::Fem => FEM_DIR,
        }
    }
}
