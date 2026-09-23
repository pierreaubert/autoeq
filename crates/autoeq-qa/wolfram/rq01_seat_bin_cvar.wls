(* autoeq-qa case autoeq-qa.rq01-seat-bin-cvar.v1 (catalogue family RQ01).
   Independent acceptance-metric evaluation: the seat/bin aggregate
   is the weighted bin mean then the seat mean (bins-then-seats) and
   the seat mean per bin then the weighted bin mean (seats-then-bins)
   with explicit order ids; the worst-tail mean (CVaR-style) is the
   mean of the largest Ceiling[n*tailFraction] values; the
   group-delay residual of a pure bulk delay tau is the constant
   -d(phase)/d(2 Pi f) = tau in ms on interior intervals. The Rust
   test calls aggregate_seat_bin, worst_tail_mean, and
   group_delay_ms. The oracle distinguishes the residual GD of the
   measured transfer from any induced-GD quality bookkeeping, which
   lives elsewhere and is not recomputed here. *)

Off[General::munfl];
values = {{1., 2., 3.}, {4., 5., 6.}};
weights = {1., 2., 1.};
wsum = Total[weights];
binsThenSeats = Mean[(# . weights/wsum) & /@ values];
seatsThenBins = ((Mean /@ Transpose[values]) . weights)/wsum;

cvarValues = {5., 1., 4., 2., 3.};
tailFraction = 0.4;
sorted = Reverse[Sort[cvarValues]];
count = Max[1, Min[Length[sorted], Ceiling[Length[sorted]*tailFraction]]];
cvar = Mean[Take[sorted, count]];

tauMs = 1.5;
gdGrid = Range[20., 200., 20.];
gdExpected = Table[tauMs, {Length[gdGrid] - 1}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rq01-seat-bin-cvar.v1",
  "family" -> "RQ01",
  "sample_rate_hz" -> 48000.,
  "grid" -> "per-seat per-bin table; group-delay grid in Hz",
  "units" -> <|"aggregate" -> "input units (dB)", "cvar" -> "input units (dB)",
    "group_delay" -> "ms"|>,
  "calibration" -> "relative metric comparison; no acoustic calibration",
  "seat_values" -> values,
  "bin_weights" -> weights,
  "bins_then_seats" -> N[binsThenSeats, 16],
  "seats_then_bins" -> N[seatsThenBins, 16],
  "bins_then_seats_id" -> "bins-then-seats-v1",
  "seats_then_bins_id" -> "seats-then-bins-v1",
  "cvar_values" -> cvarValues,
  "tail_fraction" -> tailFraction,
  "cvar_count" -> count,
  "cvar" -> N[cvar, 16],
  "bulk_delay_ms" -> tauMs,
  "gd_grid_hz" -> gdGrid,
  "group_delay_ms" -> (N[#, 16] & /@ gdExpected)
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
