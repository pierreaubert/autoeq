(* autoeq-qa case autoeq-qa.c05-minphase-flat.v1 (catalogue family C05).
   Analytic minimum-phase oracle: a flat magnitude (constant ln|H|) has
   zero Hilbert transform, so the minimum phase is identically 0 degrees.
   This separates magnitude/min-phase agreement (exact analytic claim)
   from excess-phase, time-domain, and spatial claims, which are NOT
   covered here. The N-class arithmetic bound 0.5 deg records the finite
   log-aware Hilbert pipeline residual measured on this grid. *)

Off[General::munfl];
freqs = N[10^Range[Log10[20.], Log10[20000.], (Log10[20000.] - Log10[20.])/15.], 16];
spl = ConstantArray[85., 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.c05-minphase-flat.v1",
  "family" -> "C05",
  "sample_rate_hz" -> "not-applicable",
  "grid_hz" -> freqs,
  "units" -> "dB-SPL-in-degrees-out",
  "calibration" -> "none",
  "claim" -> "flat-magnitude-minimum-phase-is-zero",
  "freqs_hz" -> freqs,
  "spl_db" -> spl,
  "min_phase_deg" -> ConstantArray[0., 16],
  "arithmetic_bound_deg" -> 0.5
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
