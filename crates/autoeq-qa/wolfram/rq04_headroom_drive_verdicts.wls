(* autoeq-qa case autoeq-qa.rq04-headroom-drive-verdicts.v1 (catalogue family RQ04).
   Independent electrical-headroom and physical-drive evaluation:
   paths from the same input add coherently (complex) before the
   independent-input peak bound sums magnitudes across inputs;
   peak_dBFS = 20 Log10[peak] (absent for exact-zero transfer) and
   the required attenuation is Max[peak_dBFS, 0]. Declared physical
   demand is reference*(peak/referencePeak) with utilization
   demand/limit; equality to a limit passes and any strictly greater
   demand fails within the declared linear domain. The Rust test
   calls evaluate_sampled_electrical_headroom and
   assess_declared_physical_drive. *)

Off[General::munfl];
sr = 48000.;
grid = {100., 1000., 10000.};

subAmp = 2.;
subDb = N[20*Log10[subAmp], 16];
satAmp = 0.5;
satDb = N[20*Log10[satAmp], 16];

subRef = 0.5; subFreqs = {100., 1000.}; subRefDemand = {10., 20.};
subLimits = {12., 18.}; subPeaks = {0.4, 0.8}; subLinear = 1.;
subDemands = subRefDemand*(subPeaks/subRef);
subUtils = subDemands/subLimits;

satRef = 1.; satFreqs = {100., 1000.}; satRefDemand = {5., 10.};
satLimits = {10., 10.}; satPeaks = {0.5, 1.}; satLinear = 1.;
satDemands = satRefDemand*(satPeaks/satRef);
satUtils = satDemands/satLimits;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rq04-headroom-drive-verdicts.v1",
  "family" -> "RQ04",
  "sample_rate_hz" -> sr,
  "grid" -> "explicit 3-point grid in Hz",
  "units" -> <|"transfer" -> "linear amplitude re full scale",
    "headroom" -> "dBFS", "demand" -> "declared RMS volts"|>,
  "calibration" -> "digital full-scale reference; physical volts declared",
  "grid_hz" -> grid,
  "electrical" -> {
    <|"output" -> "sub", "peak_amplitude" -> subAmp,
      "peak_dbfs" -> subDb, "required_attenuation_db" -> subDb,
      "peak_frequency_hz" -> 100.|>,
    <|"output" -> "sat", "peak_amplitude" -> satAmp,
      "peak_dbfs" -> satDb, "required_attenuation_db" -> 0.,
      "peak_frequency_hz" -> 100.|>,
    <|"output" -> "coh", "peak_amplitude" -> 0.,
      "peak_dbfs" -> "none", "required_attenuation_db" -> 0.,
      "peak_frequency_hz" -> 100.|>},
  "physical" -> {
    <|"output" -> "sub", "unit" -> "V RMS",
      "reference_output_peak" -> subRef,
      "linear_valid_output_peak" -> subLinear,
      "frequencies_hz" -> subFreqs,
      "digital_output_peaks" -> subPeaks,
      "demand_at_reference" -> subRefDemand,
      "limits" -> subLimits,
      "demands" -> (N[#, 16] & /@ subDemands),
      "max_utilization" -> N[Max[subUtils], 16],
      "within_declared_linear_domain" -> True,
      "passes_declared_samples" -> False|>,
    <|"output" -> "sat", "unit" -> "V RMS",
      "reference_output_peak" -> satRef,
      "linear_valid_output_peak" -> satLinear,
      "frequencies_hz" -> satFreqs,
      "digital_output_peaks" -> satPeaks,
      "demand_at_reference" -> satRefDemand,
      "limits" -> satLimits,
      "demands" -> (N[#, 16] & /@ satDemands),
      "max_utilization" -> N[Max[satUtils], 16],
      "within_declared_linear_domain" -> True,
      "passes_declared_samples" -> True|>}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
