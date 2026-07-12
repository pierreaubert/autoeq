use super::fixtures::{
    delay_oracle, linkwitz_riley4_oracle, log_frequency_grid, parallel_woofers_oracle,
    room_mode_oracle, room_transition_oracle,
};
use super::types::{AcousticOracle, ParallelSourceParameters};
use num_complex::Complex64;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum QaTier {
    Pr,
    Nightly,
    Weekly,
    Release,
}

impl QaTier {
    pub fn includes(self, candidate: Self) -> bool {
        fn rank(tier: QaTier) -> u8 {
            match tier {
                QaTier::Pr => 0,
                QaTier::Nightly => 1,
                QaTier::Weekly => 2,
                QaTier::Release => 3,
            }
        }
        rank(candidate) <= rank(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerTopology {
    Single,
    TwoWay,
    ThreeWay,
    ParallelWoofers(usize),
    MultiSub(usize),
    MultiSeat,
    HeightLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridProfile {
    Dense,
    Sparse,
    Mismatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseAvailability {
    Complete,
    Missing,
    Wrapped,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementNoise {
    Clean,
    Noisy { rms_db: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerBudget {
    Smoke,
    Standard,
    Production,
}

/// Baselines that every optimizer scenario must evaluate before promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Identity,
    AnalyticCorrection,
    CurrentMain,
    Candidate,
}

/// Physical and sampling conditions used to generate a scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticEnvironment {
    pub room_dimensions_m: [f64; 3],
    pub rt60_seconds: f64,
    pub crossover_hz: f64,
    pub training_positions_m: Vec<[f64; 3]>,
    pub held_out_positions_m: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioSpec {
    pub name: String,
    pub topology: SpeakerTopology,
    pub seats: usize,
    pub grid: GridProfile,
    pub phase: PhaseAvailability,
    pub coherence_available: bool,
    pub noise: MeasurementNoise,
    pub seed: u64,
    pub budget: OptimizerBudget,
    /// Held-out positions/sweeps that are evaluated but not fitted.
    pub held_out_measurements: usize,
    pub environment: AcousticEnvironment,
    pub comparison_set: [CandidateKind; 4],
}

fn comparison_set() -> [CandidateKind; 4] {
    [
        CandidateKind::Identity,
        CandidateKind::AnalyticCorrection,
        CandidateKind::CurrentMain,
        CandidateKind::Candidate,
    ]
}

fn seat_position(room: [f64; 3], seed: u64, index: usize, held_out: bool) -> [f64; 3] {
    let offset = if held_out { 53 } else { 7 };
    let sample = |multiplier: u64| {
        ((seed
            .wrapping_mul(multiplier)
            .wrapping_add(index as u64 * 29)
            .wrapping_add(offset)
            % 101) as f64)
            / 100.0
    };
    [
        room[0] * (0.2 + 0.6 * sample(17)),
        room[1] * (0.2 + 0.6 * sample(31)),
        room[2] * (0.4 + 0.15 * sample(43)),
    ]
}

fn environment(seed: u64, seats: usize, held_out: usize) -> AcousticEnvironment {
    let room_dimensions_m = [
        4.2 + (seed % 5) as f64 * 0.45,
        3.4 + ((seed / 3) % 5) as f64 * 0.35,
        2.35 + ((seed / 7) % 4) as f64 * 0.12,
    ];
    let crossover_values = [350.0, 500.0, 800.0, 1_200.0, 1_800.0, 2_500.0];
    AcousticEnvironment {
        room_dimensions_m,
        rt60_seconds: 0.25 + (seed % 7) as f64 * 0.1,
        crossover_hz: crossover_values[seed as usize % crossover_values.len()],
        training_positions_m: (0..seats)
            .map(|index| seat_position(room_dimensions_m, seed, index, false))
            .collect(),
        held_out_positions_m: (0..held_out)
            .map(|index| seat_position(room_dimensions_m, seed, index, true))
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn scenario(
    name: &str,
    topology: SpeakerTopology,
    seats: usize,
    grid: GridProfile,
    phase: PhaseAvailability,
    coherence_available: bool,
    noise: MeasurementNoise,
    seed: u64,
    budget: OptimizerBudget,
    held_out_measurements: usize,
) -> ScenarioSpec {
    ScenarioSpec {
        name: name.to_string(),
        topology,
        seats,
        grid,
        phase,
        coherence_available,
        noise,
        seed,
        budget,
        held_out_measurements,
        environment: environment(seed, seats, held_out_measurements),
        comparison_set: comparison_set(),
    }
}

fn pr_scenarios() -> Vec<ScenarioSpec> {
    vec![
        scenario(
            "identity_single",
            SpeakerTopology::Single,
            1,
            GridProfile::Dense,
            PhaseAvailability::Complete,
            true,
            MeasurementNoise::Clean,
            1,
            OptimizerBudget::Smoke,
            1,
        ),
        scenario(
            "two_way_mismatched_grid",
            SpeakerTopology::TwoWay,
            1,
            GridProfile::Mismatched,
            PhaseAvailability::Wrapped,
            true,
            MeasurementNoise::Clean,
            2,
            OptimizerBudget::Smoke,
            1,
        ),
        scenario(
            "three_way_sparse_missing_phase",
            SpeakerTopology::ThreeWay,
            1,
            GridProfile::Sparse,
            PhaseAvailability::Missing,
            false,
            MeasurementNoise::Noisy { rms_db: 0.5 },
            3,
            OptimizerBudget::Smoke,
            1,
        ),
        scenario(
            "parallel_two_woofers",
            SpeakerTopology::ParallelWoofers(2),
            2,
            GridProfile::Dense,
            PhaseAvailability::Complete,
            true,
            MeasurementNoise::Clean,
            4,
            OptimizerBudget::Smoke,
            1,
        ),
        scenario(
            "parallel_four_woofers_noisy",
            SpeakerTopology::ParallelWoofers(4),
            3,
            GridProfile::Sparse,
            PhaseAvailability::Complete,
            true,
            MeasurementNoise::Noisy { rms_db: 1.0 },
            5,
            OptimizerBudget::Smoke,
            2,
        ),
        scenario(
            "multi_sub_two_seat",
            SpeakerTopology::MultiSub(2),
            3,
            GridProfile::Dense,
            PhaseAvailability::Complete,
            true,
            MeasurementNoise::Noisy { rms_db: 0.5 },
            6,
            OptimizerBudget::Smoke,
            2,
        ),
        scenario(
            "multi_sub_four_missing_phase",
            SpeakerTopology::MultiSub(4),
            4,
            GridProfile::Mismatched,
            PhaseAvailability::Missing,
            false,
            MeasurementNoise::Noisy { rms_db: 1.0 },
            7,
            OptimizerBudget::Smoke,
            2,
        ),
        scenario(
            "height_layout_arrival",
            SpeakerTopology::HeightLayout,
            2,
            GridProfile::Mismatched,
            PhaseAvailability::Complete,
            true,
            MeasurementNoise::Clean,
            8,
            OptimizerBudget::Smoke,
            1,
        ),
    ]
}

pub fn scenario_matrix(tier: QaTier) -> Vec<ScenarioSpec> {
    if tier == QaTier::Pr {
        return pr_scenarios();
    }
    let topologies = [
        SpeakerTopology::Single,
        SpeakerTopology::TwoWay,
        SpeakerTopology::ThreeWay,
        SpeakerTopology::ParallelWoofers(4),
        SpeakerTopology::MultiSub(4),
        SpeakerTopology::MultiSeat,
        SpeakerTopology::HeightLayout,
    ];
    let phases = [
        PhaseAvailability::Complete,
        PhaseAvailability::Missing,
        PhaseAvailability::Wrapped,
    ];
    let grids = [
        GridProfile::Dense,
        GridProfile::Sparse,
        GridProfile::Mismatched,
    ];
    let seats = [1, 3, 5];
    let seeds: &[u64] = match tier {
        QaTier::Nightly => &[11, 29],
        QaTier::Weekly => &[11, 29, 47, 71],
        QaTier::Release => &[11, 29, 47, 71, 101],
        QaTier::Pr => unreachable!(),
    };
    let budget = match tier {
        QaTier::Nightly => OptimizerBudget::Standard,
        QaTier::Weekly => OptimizerBudget::Production,
        QaTier::Release => OptimizerBudget::Production,
        QaTier::Pr => OptimizerBudget::Smoke,
    };
    let mut scenarios = Vec::new();
    for &topology in &topologies {
        for &phase in &phases {
            for &grid in &grids {
                for &seat_count in &seats {
                    for &seed in seeds {
                        let noise = if seed.is_multiple_of(2) {
                            MeasurementNoise::Clean
                        } else {
                            MeasurementNoise::Noisy { rms_db: 0.75 }
                        };
                        scenarios.push(scenario(
                            &format!(
                                "{:?}_{:?}_{:?}_{}seat_seed{}",
                                topology, phase, grid, seat_count, seed
                            )
                            .to_lowercase(),
                            topology,
                            seat_count,
                            grid,
                            phase,
                            phase == PhaseAvailability::Complete,
                            noise,
                            seed,
                            budget,
                            (seat_count / 2).max(1),
                        ));
                    }
                }
            }
        }
    }
    scenarios
}

fn source_parameters(count: usize) -> Vec<ParallelSourceParameters> {
    (0..count)
        .map(|index| ParallelSourceParameters {
            gain_db: -(index as f64) * 0.75,
            delay_ms: index as f64 * 0.8,
            inverted: false,
        })
        .collect()
}

/// Map a scenario to deterministic analytic ground truth.
pub fn oracle_for_scenario(scenario: &ScenarioSpec) -> AcousticOracle {
    let points = match scenario.grid {
        GridProfile::Dense => 257,
        GridProfile::Sparse => 41,
        GridProfile::Mismatched => 127,
    };
    match scenario.topology {
        SpeakerTopology::Single => room_transition_oracle(
            log_frequency_grid(points, 20.0, 20_000.0),
            scenario.environment.room_dimensions_m.iter().product(),
            scenario.environment.rt60_seconds,
            1.0,
        ),
        SpeakerTopology::TwoWay => linkwitz_riley4_oracle(
            log_frequency_grid(points, 20.0, 20_000.0),
            scenario.environment.crossover_hz,
        ),
        SpeakerTopology::ThreeWay => linkwitz_riley4_oracle(
            log_frequency_grid(points, 20.0, 20_000.0),
            scenario.environment.crossover_hz.min(800.0),
        ),
        SpeakerTopology::ParallelWoofers(count) | SpeakerTopology::MultiSub(count) => {
            parallel_woofers_oracle(
                log_frequency_grid(points, 20.0, 300.0),
                source_parameters(count),
            )
        }
        SpeakerTopology::MultiSeat => {
            room_mode_oracle(log_frequency_grid(points, 20.0, 500.0), 63.0, 8.0, 6.0)
        }
        SpeakerTopology::HeightLayout => {
            delay_oracle(log_frequency_grid(points, 80.0, 16_000.0), 1.75)
        }
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Deterministic measurement perturbation for replay and dispersion checks.
pub fn perturb_transfer(
    transfer: &[Complex64],
    noise: MeasurementNoise,
    seed: u64,
) -> Vec<Complex64> {
    let rms_db = match noise {
        MeasurementNoise::Clean => 0.0,
        MeasurementNoise::Noisy { rms_db } => rms_db,
    };
    let mut state = seed.max(1);
    transfer
        .iter()
        .map(|&value| {
            let uniform = xorshift64(&mut state) as f64 / u64::MAX as f64;
            let centered = (uniform - 0.5) * 2.0;
            let magnitude = 10.0_f64.powf(centered * rms_db / 20.0);
            let phase = centered * rms_db.to_radians();
            value * Complex64::from_polar(magnitude, phase)
        })
        .collect()
}
