(* autoeq-qa case autoeq-qa.o10-bark-sharpness.v1 (catalogue family O10).
   Independent evaluation of the pinned EPA auditory-proxy formulas from
   their published definitions: Zwicker's Bark conversion
   z = 13 ArcTan[0.00076 f] + 3.5 ArcTan[(f/7500)^2], the critical
   bandwidth CBW(f) = 25 + 75 (1 + 1.4 (f/1000)^2)^0.69, the DIN 45692
   sharpness weight g(z) = 1 for z <= 15.8 else
   0.15 Exp[0.42 (z-15.8)] + 0.85 at integer bands 1..24, and Zwicker
   sharpness S = 0.11 Sum[N'(z) g(z) z]/max[N, 0.001] on a hand-designed
   two-tone specific loudness (1.0 sone/Bark at band 9, 0.5 at band 18).
   The Bark inverse has no closed form: the oracle inverts with
   high-precision FindRoot (independent of the Rust Newton iteration)
   and the comparison uses the pinned 2% round-trip tolerance. These are
   exact implementation checks of proxies, not perceptual-validity
   claims. *)

Off[General::munfl];
hzToBark[f_] := N[13 ArcTan[0.00076 f] + 3.5 ArcTan[(f/7500.)^2], 16];
cbw[f_] := N[25 + 75 (1 + 1.4 (f/1000.)^2)^0.69, 16];
gweight[z_] := N[If[z <= 15.8, 1., 0.15 Exp[0.42 (z - 15.8)] + 0.85], 16];

probeFreqs = {100., 1000., 10000.};
barkValues = hzToBark /@ probeFreqs;
cbwValues = cbw /@ {500., 5000.};
weights = Table[gweight[z], {z, 1, 24}];

specific = Table[0., {24}];
specific[[9]] = 1.0;
specific[[18]] = 0.5;
totalLoud = N[Total[specific], 16];
sharp = N[0.11 Total[Table[specific[[z]] weights[[z]] z, {z, 1, 24}]]/Max[totalLoud, 0.001], 16];

(* Independent high-precision inversion for the round-trip check:
   bisection on the monotone Zwicker curve, not Newton. *)
zOf[f_] := 13 ArcTan[0.00076 f] + 3.5 ArcTan[(f/7500.)^2];
invertHz[z_] := Module[{lo = 1., hi = 25000., mid = 0.},
  Do[mid = (lo + hi)/2; If[zOf[mid] < z, lo = mid, hi = mid], {200}];
  N[(lo + hi)/2, 16]];
roundTrip = invertHz /@ barkValues;

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o10-bark-sharpness.v1",
  "family" -> "O10",
  "sample_rate_hz" -> 48000.,
  "grid" -> "24 integer Bark bands; probe freqs {100,1000,10000} Hz",
  "units" -> "Hz, Bark, sone/Bark, acum",
  "calibration" -> "proxy formulas only; no listening-level calibration asserted",
  "probe_freqs_hz" -> probeFreqs,
  "bark_values" -> barkValues,
  "bark_roundtrip_hz" -> roundTrip,
  "cbw_freqs_hz" -> {500., 5000.},
  "critical_bandwidth_hz" -> cbwValues,
  "sharpness_weights" -> weights,
  "specific_loudness" -> specific,
  "total_loudness_sone" -> totalLoud,
  "sharpness_acum" -> sharp
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
