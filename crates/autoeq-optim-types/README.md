# autoeq-optim-types

Optimizer-neutral serialized contracts shared across the AutoEQ
optimizer backends and the RoomEQ model/report layer: Pareto and
input-normalization evidence (`evidence`) plus the approved
independent-reference registry (`reference_registry`).

This crate exists so `autoeq-optim` and `roomeq-model` do not depend
on each other for shared types (refactor-20261025 Phase 3c).
Serialized forms are a wire contract: field names, serde attributes,
and schema markers must not change here.
