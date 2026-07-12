use clap::ValueEnum;
use std::fmt;

/// PEQ model types that define the structure and constraints of the equalizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PeqModel {
    #[value(name = "pk")]
    Pk,
    #[value(name = "hp-pk")]
    HpPk,
    #[value(name = "hp-pk-lp")]
    HpPkLp,
    #[value(name = "ls-pk")]
    LsPk,
    #[value(name = "ls-pk-hs")]
    LsPkHs,
    #[value(name = "free-pk-free")]
    FreePkFree,
    #[value(name = "free")]
    Free,
}

impl fmt::Display for PeqModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pk => "pk",
            Self::HpPk => "hp-pk",
            Self::HpPkLp => "hp-pk-lp",
            Self::LsPk => "ls-pk",
            Self::LsPkHs => "ls-pk-hs",
            Self::FreePkFree => "free-pk-free",
            Self::Free => "free",
        })
    }
}

impl PeqModel {
    pub fn all() -> Vec<Self> {
        vec![Self::Pk, Self::HpPk, Self::LsPk, Self::HpPkLp, Self::LsPkHs, Self::FreePkFree, Self::Free]
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Pk => "All filters are peak/bell filters",
            Self::HpPk => "First filter is highpass, rest are peak filters",
            Self::HpPkLp => "First filter is highpass, last is lowpass, rest are peak filters",
            Self::LsPk => "First filter is low shelve, rest are peak filters",
            Self::LsPkHs => "First filter is low shelve, last is high shelve, rest are peak filters",
            Self::FreePkFree => "First and last filters can be any type, middle filters are peak",
            Self::Free => "All filters can be any type (peak, highpass, lowpass, shelf)",
        }
    }
}

impl std::str::FromStr for PeqModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pk" => Ok(Self::Pk),
            "hp-pk" => Ok(Self::HpPk),
            "hp-pk-lp" => Ok(Self::HpPkLp),
            "ls-pk" => Ok(Self::LsPk),
            "ls-pk-hs" => Ok(Self::LsPkHs),
            "free-pk-free" => Ok(Self::FreePkFree),
            "free" => Ok(Self::Free),
            _ => Err(format!("Unknown PEQ model: {s}")),
        }
    }
}
