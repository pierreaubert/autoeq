(* autoeq-qa case autoeq-qa.re19-bass-power.v1 (catalogue family RE19).
   Independent input-to-output bass transfer: two routes (redirected
   main with input trim, inverted LFE) apply gain, delay slope and
   polarity to the sub phasor and sum coherently (no crossover, so the
   crossover factor is exactly one). Programme power follows H R H^H
   for the three declared correlation matrices (coherent, uncorrelated,
   rho = 0.5). *)

Off[General::munfl];
sr = 48000.;
freqs = {30., 50., 70., 100., 120.};
subSpl = 85.;
g1 = -2.; trim1 = 1.; tau1 = 1.5;
g2 = 3.;
h1[f_] := N[10^((subSpl + g1 + trim1)/20), 16] *
   Exp[-I N[2 Pi f tau1/1000., 16]];
h2[f_] := N[10^((subSpl + g2)/20), 16] * Exp[I N[180. Degree, 16]];
hh1 = h1[#] & /@ freqs;
hh2 = h2[#] & /@ freqs;
comb = hh1 + hh2;
splOf[z_] := N[20 Log10[Max[Abs[z], 10^-12]], 16];
argOf[z_] := N[Arg[z]/Degree, 16];
pcoh = N[Abs[#]^2 & /@ comb, 16];
punc = N[Abs[hh1]^2 + Abs[hh2]^2, 16];
prho = N[punc + 2*0.5*Re[hh1 Conjugate[hh2]], 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re19-bass-power.v1",
  "family" -> "RE19",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "grid" -> "sub output grid reused directly (no resampling)",
  "units" -> <|"spl" -> "dB", "phase" -> "degrees", "delay" -> "ms", "power" -> "linear"|>,
  "calibration" -> "mic-1",
  "sub_spl_db" -> subSpl,
  "route_redirected" -> <|"gain_db" -> g1, "input_trim_db" -> trim1,
    "delay_ms" -> tau1, "polarity_inverted" -> False|>,
  "route_lfe" -> <|"gain_db" -> g2, "input_trim_db" -> 0.,
    "delay_ms" -> 0., "polarity_inverted" -> True|>,
  "route_transfer_re_im" -> {complexPair[#] & /@ hh1, complexPair[#] & /@ hh2},
  "combined_re_im" -> (complexPair[#] & /@ comb),
  "combined_spl_db" -> (splOf[#] & /@ comb),
  "combined_phase_deg" -> (argOf[#] & /@ comb),
  "correlation_coherent" -> {{1., 1.}, {1., 1.}},
  "correlation_uncorrelated" -> {{1., 0.}, {0., 1.}},
  "correlation_partial" -> {{1., 0.5}, {0.5, 1.}},
  "power_coherent" -> pcoh,
  "power_uncorrelated" -> punc,
  "power_partial_rho_0p5" -> prho
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
