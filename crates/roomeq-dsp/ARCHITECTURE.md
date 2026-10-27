# roomeq-dsp architecture

- `fir`: group-delay FIR delay realization with explicitly
  allocated common causal support (padding is part of reported
  latency).
- `physical_routing`: resolves legacy RoomEQ ownership into the
  shared physical routing contract; independent of native
  plugin-node types.
- `output`: portable serialized plugin configurations (gain,
  delay, EQ, crossover, convolution, matrix) over
  `roomeq-model` types.

Dependencies: `roomeq-model`, `math-iir-fir`, `serde_json` only.
Nothing here may depend on optimizer backends or workflow code.
