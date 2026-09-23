(* autoeq-qa case autoeq-qa.re20-drr-velvet.v1 (catalogue family RE20).
   Independent magnitude-only DRR exactly as implemented: the primary
   amplitude is split by direct_window_ratio, each share squared, and
   the support power added to the reverberant share; zero-energy sides
   take the +-40 dB fallbacks. Plus the fixed explicit velvet-noise
   integer sequence (xorshift64 spacings, 0-based positions). *)

Off[General::munfl];
freqs = {100., 500., 2000., 8000.};
mag[db_] := 10^(db/20);
drr[primDb_, supDb_, gainDb_, ratio_] := Module[{dir, rev, revAfter},
   dir = (mag[primDb] ratio)^2;
   rev = (mag[primDb] (1 - ratio))^2;
   revAfter = rev + (mag[supDb] mag[gainDb])^2;
   {ratioDb[dir, rev], ratioDb[dir, revAfter]}];
(* guard form matching the implementation fallbacks exactly:
   reverberant <= 0 (or non-finite) -> +40; else direct <= 0 -> -40. *)
ratioDb[dir_, rev_] := N[10 Log10[dir/rev], 16] /;
   TrueQ[rev > 0] && TrueQ[dir > 0];
ratioDb[dir_, rev_] := 40. /; ! TrueQ[rev > 0];
ratioDb[dir_, rev_] := -40. /; TrueQ[rev > 0] && ! TrueQ[dir > 0];
fitA = Transpose[drr[80., 74., #, 0.5] & /@ {0., 0., -6., 0.}];
(* exact -40 dB fallback: ratio 0 makes the direct share exactly zero *)
fitB = Transpose[drr[80., 74., 0., 0.] & /@ freqs];
(* exact +40 dB fallback: ratio 1 zeroes the primary share and the
   -1e308 dB support gain underflows its magnitude to exactly zero *)
fitC = Transpose[drr[80., 74., -1.*10^308, 1.] & /@ freqs];
(* tiny but nonzero energies take no fallback: exact finite ratios *)
fitD = Transpose[drr[#, 80., 0., 0.5] & /@ {-1000., -1000., -1000., -1000.}];
(* fixed velvet-noise sequence: n=64, density=0.25, seed=7 *)
mod = 2^64;
xs[s_] := Module[{x = If[s == 0, 16045690984503098046, s]},
   x = Mod[BitXor[x, Mod[x 2^13, mod]], mod];
   x = Mod[BitXor[x, Quotient[x, 2^7]], mod];
   x = Mod[BitXor[x, Mod[x 2^17, mod]], mod];
   x];
nTaps = 64; density = 0.25; seed = 7;
meanSpacing = N[Max[1/density, 1.], 16];
taps = Table[0., {nTaps}];
pos = 0;
state = Mod[seed + 11400714819323198485, mod];
While[pos < nTaps,
  state = xs[state];
  taps[[pos + 1]] = If[EvenQ[state], 1., -1.];
  state = xs[state];
  pos += Floor[1. + meanSpacing*(N[state, 16]/N[2^64, 16])]];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re20-drr-velvet.v1",
  "family" -> "RE20",
  "sample_rate_hz" -> 48000.,
  "freqs_hz" -> freqs,
  "grid" -> "explicit per-bin vectors (no resampling)",
  "units" -> <|"spl" -> "dB", "drr" -> "dB"|>,
  "calibration" -> "mic-1",
  "fixture_a" -> <|"primary_db" -> 80., "support_db" -> 74.,
    "support_gain_db" -> {0., 0., -6., 0.}, "direct_window_ratio" -> 0.5,
    "drr_before_db" -> fitA[[1]], "drr_after_db" -> fitA[[2]]|>,
  "fixture_zero_direct" -> <|"primary_db" -> 80., "support_db" -> 74.,
    "support_gain_db" -> {0., 0., 0., 0.}, "direct_window_ratio" -> 0.,
    "drr_before_db" -> fitB[[1]], "drr_after_db" -> fitB[[2]]|>,
  "fixture_zero_reverberant" -> <|"primary_db" -> 80., "support_db" -> 74.,
    "support_gain_db" -> {-1.*10^308, -1.*10^308, -1.*10^308, -1.*10^308},
    "direct_window_ratio" -> 1.,
    "drr_before_db" -> fitC[[1]], "drr_after_db" -> fitC[[2]]|>,
  "fixture_tiny_no_fallback" -> <|"primary_db" -> -1000., "support_db" -> 80.,
    "support_gain_db" -> {0., 0., 0., 0.}, "direct_window_ratio" -> 0.5,
    "drr_before_db" -> fitD[[1]], "drr_after_db" -> fitD[[2]]|>,
  "velvet" -> <|"n_taps" -> nTaps, "density" -> density, "seed" -> seed,
    "taps" -> taps|>
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
