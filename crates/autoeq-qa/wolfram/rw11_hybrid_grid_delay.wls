(* autoeq-qa case autoeq-qa.rw11-hybrid-grid-delay.v1 (catalogue family RW11).
   Independent derivation of the dense-measurement reduction kernels:
   (a) the canonical RoomEQ hybrid grid for frequency_samples = 200
   (4 Hz bass spacing 20..1000 Hz, then log spacing 1..20 kHz with the
   density of 200 samples across 20..20000 Hz), clipped to a synthetic
   measured span [25, 19000] Hz with exact source endpoints retained;
   (b) bulk-delay estimation from pure-delay excess phase
   (phi = -360 f tau, tau = 2.5 ms) by direct least-squares on the
   published linear-phase definition, plus the delay-restored phase
   (min + residual excess - 360 f tau) for a flat (min-phase 0)
   magnitude. The oracle re-derives every grid point and the OLS sums
   directly, never by calling Rust. Catches off-by-one bass bins,
   logstep slips, endpoint drops, delay-sign flips, and
   degree/radian mixups. *)

Off[General::munfl];
n = 200;
logStep = N[Log[1000.]/199, 16];
hi = Ceiling[Log[20.]/logStep];
bassGrid = Table[N[20. + 4*j, 16], {j, 0, 245}];
highGrid = Table[N[1000.*Exp[k*Log[20.]/hi], 16], {k, 1, hi}];
fullGrid = Join[bassGrid, highGrid];

smin = 25.; smax = 19000.;
clipped = Select[fullGrid, smin <= # <= smax &];
If[First[clipped] > smin, PrependTo[clipped, smin]];
If[Last[clipped] < smax, AppendTo[clipped, smax]];

tauMs = 2.5;
dFreqs = {50., 100., 200., 300., 400.};
excessDeg = (N[-360.*#*tauMs/1000, 16] & /@ dFreqs);

(* Independent OLS delay fit from the linear-phase definition. *)
nD = Length[dFreqs];
excessRad = (N[#*Pi/180, 16] & /@ excessDeg);
sf = Total[dFreqs]; sp = Total[excessRad];
sf2 = Total[dFreqs^2]; sfp = Total[dFreqs*excessRad];
fitSlope = (nD*sfp - sf*sp)/(nD*sf2 - sf^2);
fitIntercept = (sp - fitSlope*sf)/nD;
delayMs = N[-fitSlope/(2 Pi)*1000, 16];
residual = Table[N[excessDeg[[k]] - (fitSlope*dFreqs[[k]] + fitIntercept)*180/Pi, 16], {k, nD}];
restored = Table[N[0. + residual[[k]] - 360.*dFreqs[[k]]*tauMs/1000, 16], {k, nD}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw11-hybrid-grid-delay.v1",
  "family" -> "RW11",
  "sample_rate_hz" -> 48000.,
  "grid" -> "hybrid optimizer grid, frequency_samples = 200, clipped to 25..19000 Hz",
  "units" -> "Hz, dB, degrees, ms",
  "calibration" -> "analytic truth: flat 80 dB magnitude with pure 2.5 ms delay phase",
  "frequency_samples" -> n,
  "high_intervals" -> hi,
  "full_grid_hz" -> fullGrid,
  "source_span_hz" -> {smin, smax},
  "clipped_grid_hz" -> clipped,
  "delay_tau_ms" -> tauMs,
  "delay_freqs_hz" -> dFreqs,
  "excess_phase_deg" -> excessDeg,
  "expected_delay_ms" -> delayMs,
  "expected_residual_deg" -> residual,
  "expected_restored_deg" -> restored
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
