(* autoeq-qa case autoeq-qa.o06-initial-guess-bounds.v1 (catalogue family O06).
   Independent derivation of expected initial-guess centers/widths/gains and
   optimizer bounds from synthetic peaks, from the declared conventions:
   - pre-detected problems (freqHz, q, gainDb), sorted by |gain| desc;
   - Pk guess per filter = [Log10[freqHz], qNominal, gainNominal], clamped
     into per-filter bounds; with zero frequency variation the decoded
     center is exactly the problem frequency;
   - documented diversification bands: q within +/-30%, gain within +/-20%
     of nominal (seeded RNG draws; bands are exact inequalities);
   - integrality flags: {freqIndexed, False, False} per filter;
   - setup bounds for Pk with overlap factor 1: range =
     (Log10[maxF] - Log10[minF])/numFilters; filter i lower =
     Max[center - range, minLog, prevLower], upper =
     Max[Min[center + range, maxLog], prevUpper]; gain lower = minDb when
     negative; q upper = frequency-Q policy value over the interval
     (rectangular bound, evaluated at the interval's upper edge). *)

Off[General::munfl];
log10[x_] := N[Log[10, x], 16];

problems = {{500., 5., -3.}, {2000., 2., 4.}};
sortedProblems = {{2000., 2., 4.}, {500., 5., -3.}};

minF = 20.; maxF = 20000.;
minLog = log10[minF]; maxLog = log10[maxF];
perFilter = {{minLog, maxLog}, {0.5, 10.}, {-12., 12.}};

guessCenters = log10[{2000., 500.}];
qNominals = {2., 5.};
gainNominals = {4., -3.};
qBands = {{1.4, 2.6}, {3.5, 6.5}};
gainBands = {{3.2, 4.8}, {-3.6, -2.4}};

integralityIndexed = {True, False, False, True, False, False};
integralityContinuous = {False, False, False, False, False, False};

(* config A: 2 filters, 20..20000 Hz, no Q policy *)
rangeA = (maxLog - minLog)/2.;
fLowA0 = minLog; fHighA0 = minLog + 2.*rangeA - rangeA;
(* i=0: center=minLog, low=Max[center-range,minLog]=minLog,
   high=Min[center+range,maxLog]=minLog+range *)
lowA0 = minLog; highA0 = minLog + rangeA;
(* i=1: center=minLog+range, low=Max[center-range,minLog,prevLow],
   high=Max[Min[center+range,maxLog],prevHigh] *)
lowA1 = Max[minLog + rangeA - rangeA, minLog, lowA0];
highA1 = Max[Min[minLog + rangeA + rangeA, maxLog], highA0];
lowerA = {lowA0, 0.5, -12., lowA1, 0.5, -12.};
upperA = {highA0, 10., 12., highA1, 10., 12.};

(* config B: 1 filter, 20..200 Hz, Schroeder policy 200/2/8 *)
minFB = 20.; maxFB = 200.;
rangeB = (log10[maxFB] - log10[minFB])/1.;
lowB = log10[minFB]; highB = log10[minFB] + rangeB;
(* interval upper edge 200 Hz <= transition 200 Hz -> low cap 2 *)
qUpperB = Min[10., 2.];
lowerB = {lowB, 0.5, -12.};
upperB = {highB, qUpperB, 12.};

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o06-initial-guess-bounds.v1",
  "family" -> "O06",
  "sample_rate_hz" -> 48000.,
  "units" -> "Hz, Log10[Hz], linear Q, dB gain",
  "calibration" -> "n/a (parameter-level case; synthetic peaks, no measurement)",
  "freq_grid_hz" -> {20., 100., 500., 1000., 5000., 20000.},
  "layout" -> "Pk [Log10[Hz], Q, gain] per filter",
  "problems_freq_q_gain" -> problems,
  "sorted_freq_q_gain" -> sortedProblems,
  "num_filters" -> 2,
  "per_filter_bounds" -> perFilter,
  "guess_log_centers" -> N[guessCenters, 16],
  "guess_center_tolerance" -> 1.*^-12,
  "q_nominals" -> qNominals,
  "q_bands" -> qBands,
  "gain_nominals" -> gainNominals,
  "gain_bands" -> gainBands,
  "integrality_indexed" -> integralityIndexed,
  "integrality_continuous" -> integralityContinuous,
  "bounds_a_min_freq" -> minF,
  "bounds_a_max_freq" -> maxF,
  "bounds_a_lower" -> N[lowerA, 16],
  "bounds_a_upper" -> N[upperA, 16],
  "bounds_b_min_freq" -> minFB,
  "bounds_b_max_freq" -> maxFB,
  "bounds_b_policy" -> <|"schroeder_hz" -> 200., "low_max_q" -> 2., "high_max_q" -> 8.|>,
  "bounds_b_lower" -> N[lowerB, 16],
  "bounds_b_upper" -> N[upperB, 16],
  "bounds_tolerance" -> 1.*^-12
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
