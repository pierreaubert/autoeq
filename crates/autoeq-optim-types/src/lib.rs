//! Optimizer-neutral contracts shared by AutoEQ optimizers and RoomEQ.
//!
//! This crate owns parser-independent configuration, objective selection,
//! serialized optimizer evidence, and the approved
//! independent-reference registry so neither `autoeq-optim` nor
//! `roomeq-model` has to depend on the other for shared types.

pub mod evidence;
pub mod q_adjustment;
pub mod reference_registry;

pub use evidence::{
    InputNormalizationEvidence, MultiInputNormalizationEvidence, NormalizationPopulation,
    NormalizationReferencePolicy, ParetoCandidateEvidence, ParetoCrowdingDistance,
    ParetoDispatchReport, ParetoSelectionEvidence,
};

pub mod config;
pub mod loss;
pub use config::OptimizationConfig;
pub use loss::LossType;
pub use q_adjustment::QAdjustment;
