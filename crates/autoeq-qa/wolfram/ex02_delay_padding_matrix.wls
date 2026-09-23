(* autoeq-qa case autoeq-qa.ex02-delay-padding-matrix.v1 (catalogue family EX02).
   Independent evaluation of the shared parallel-stage padding rule:
   shift = delayMs*rate/1000 samples; a stage is fractional when
   Abs[shift - Round[shift]] > 1e-9; the causal support needed is
   earliest = Floor[shift] - kernelHalf (fractional) or Round[shift]
   (integer); pad = Ceiling[Max[-earliest, 0]] maximized over the
   stage. Integer delays realize as exact sample shifts of the taps;
   a fractional unit impulse realizes as a DC-normalized Blackman
   sinc kernel of length 2*kernelHalf + 1 at base Floor[shift]-half.
   The Rust test calls gd_delay_padding_samples and
   realize_gd_fir_delay; agreement checks delay quantization,
   common-padding maximization, and integer-shift exactness. *)

Off[General::munfl];
rate = 48000.;
half = 64;
fracTol = 1*^-9;

padOne[d_] := Module[{shift = N[d*rate/1000, 16], earliest},
  earliest = If[Abs[shift - Round[shift]] > fracTol,
    Floor[shift] - half, Round[shift]];
  Ceiling[Max[-earliest, 0.]]
];
stagePad[delays_] := Max[padOne /@ delays];
fractional[d_] := Abs[N[d*rate/1000, 16] - Round[N[d*rate/1000, 16]]] > fracTol;

groupA = {0., N[5*1000/rate, 16]};
groupB = {0., 0.10478137820238476};
groupC = {-1.};

fracMs = 0.10478137820238476;
fracShift = N[fracMs*rate/1000, 16] + stagePad[groupB];
fracBase = Floor[fracShift] - half;
fracLen = fracBase + 1 + (2*half + 1) - 1;

intDelayMs = N[5*1000/rate, 16];
intShift = Round[N[intDelayMs*rate/1000, 16]];
intTaps = {0.25, 0.5, 0.25};
intRealized = Join[Table[0., {intShift}], intTaps];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ex02-delay-padding-matrix.v1",
  "family" -> "EX02",
  "sample_rate_hz" -> rate,
  "grid" -> "delay values in ms, not a frequency grid",
  "units" -> <|"delay" -> "ms", "padding" -> "samples", "taps" -> "linear-amplitude"|>,
  "calibration" -> "digital-sample delays; padding is causal-support samples",
  "kernel_half" -> half,
  "fractional_tol" -> fracTol,
  "groups" -> {
    <|"delays_ms" -> groupA, "pad_samples" -> stagePad[groupA],
      "fractional_present" -> Or @@ (fractional /@ groupA)|>,
    <|"delays_ms" -> groupB, "pad_samples" -> stagePad[groupB],
      "fractional_present" -> Or @@ (fractional /@ groupB)|>,
    <|"delays_ms" -> groupC, "pad_samples" -> stagePad[groupC],
      "fractional_present" -> Or @@ (fractional /@ groupC)|>
  },
  "integer_delay" -> <|"delay_ms" -> intDelayMs,
    "pad_samples" -> stagePad[{intDelayMs}],
    "shift_samples" -> intShift,
    "input_taps" -> intTaps,
    "realized_taps" -> (N[#, 16] & /@ intRealized),
    "effective_delay_ms" -> N[intShift*1000/rate, 16]|>,
  "fractional_unit" -> <|"delay_ms" -> fracMs,
    "pad_samples" -> stagePad[groupB],
    "shift_samples" -> N[fracShift, 16],
    "base_index" -> fracBase,
    "kernel_length" -> fracLen,
    "dc_gain" -> 1.|>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
