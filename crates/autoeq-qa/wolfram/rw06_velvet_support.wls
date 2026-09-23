(* autoeq-qa case autoeq-qa.rw06-velvet-support.v1 (catalogue family RW06).
   Independent delayed-support velvet sequence: Marsaglia xorshift64
   (state ^= state<<13; state ^= state>>7; state ^= state<<17, all mod
   2^64, with the documented zero-state substitute) drives signs
   (+1 for even state, -1 for odd) and gaps. The f64 conversions use
   IEEE-754 round-half-even, emulated here with exact rational
   arithmetic; the gap increment is Floor of the rounded
   1 + meanSpacing*(state/2^64) with meanSpacing = Max[1/density, 1].
   The oracle never calls generate_velvet_noise; it restates the
   published PRNG, so it catches seed/additive-constant slips,
   shift-count swaps, and sign errors. Also pinned: the support
   channel name "<role>_support" and the population mean/std of a
   small gain vector (variance leg of the final-sum check). *)

Off[General::munfl];
m64 = 2^64;
xorshift[s_] := Module[{a, b, c},
  a = BitXor[s, Mod[2^13*s, m64]];
  b = BitXor[a, Quotient[a, 2^7]];
  c = BitXor[b, Mod[2^17*b, m64]];
  Mod[c, m64]
];
toF64[u_] := Module[{e, ulp, q, r, f},
  If[u == 0, Return[0]];
  e = Floor[Log2[u]];
  ulp = 2^(e - 52);
  q = u/ulp;
  r = Floor[q];
  f = q - r;
  (Which[f < 1/2, r, f > 1/2, r + 1, EvenQ[r], r, True, r + 1])*ulp
];

nTaps = 32;
density = 0.25;
seed = 12345;
meanSpacing = Max[1/density, 1];

state = Mod[seed + 16^^9e3779b97f4a7c15, m64];
taps = Table[0, {nTaps}];
pos = 0;
While[pos < nTaps,
  state = xorshift[state];
  taps[[pos + 1]] = If[EvenQ[state], 1, -1];
  state = xorshift[state];
  t = 1 + meanSpacing*toF64[state]/m64;
  e2 = Floor[Log2[t]];
  ulp2 = 2^(e2 - 52);
  q2 = t/ulp2;
  r2 = Floor[q2];
  f2 = q2 - r2;
  rounded = Which[f2 < 1/2, r2, f2 > 1/2, r2 + 1, EvenQ[r2], r2, True, r2 + 1];
  pos += Floor[rounded*ulp2];
];

nnz = Count[taps, _?(# != 0 &)];
energy = Total[taps^2];

summaryVals = {1., 2., 3., 4.};
smean = N[Mean[summaryVals], 16];
sstd = N[Sqrt[Mean[(summaryVals - Mean[summaryVals])^2]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw06-velvet-support.v1",
  "family" -> "RW06",
  "sample_rate_hz" -> 48000.,
  "grid" -> "32-tap support sequence; tap index origin 0",
  "units" -> "dimensionless taps (+1/-1/0); dB summary vector",
  "calibration" -> "deterministic seed; no acoustic level",
  "n_taps" -> nTaps,
  "density" -> N[density, 16],
  "seed" -> seed,
  "velvet_taps" -> taps,
  "nonzero_count" -> nnz,
  "energy" -> energy,
  "support_channel_name" -> "L_support",
  "summary_values" -> summaryVals,
  "summary_mean" -> smean,
  "summary_std" -> sstd
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
