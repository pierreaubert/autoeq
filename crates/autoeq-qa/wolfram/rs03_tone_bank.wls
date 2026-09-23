(* autoeq-qa case autoeq-qa.rs03-tone-bank.v1 (catalogue family RS03).
   Independent deterministic probe signal: a zero-phase cosine tone bank
   (110/165/220 Hz at amplitudes 0.5/0.3/0.2) sampled at 8000 Hz for 0.2 s
   (22/33/44 integer cycles in 1600 samples). Direct-sample sums give
   energy, peak and RMS; the closed-form DTFT magnitudes at the component
   bins are a*N/2 = {25, 15, 10} since every cross term integrates to
   zero over integer cycles. *)

Off[General::munfl];
sr = 8000.;
dur = 0.2;
n = 1600;
seed = 3;
comps = {{110., 0.5}, {165., 0.3}, {220., 0.2}};

samples = Table[
  N[Sum[c[[2]]*Cos[2*Pi*c[[1]]*(i - 1)/sr], {c, comps}], 16],
  {i, 1, n}];
energy = N[Total[samples^2], 16];
peak = N[Max[Abs[samples]], 16];
rms = N[Sqrt[energy/n], 16];
energyClosed = N[n/2*Total[{0.5, 0.3, 0.2}^2], 16];
dtftClosed = N[{0.5, 0.3, 0.2}*n/2, 16];

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rs03-tone-bank.v1",
  "family" -> "RS03",
  "wolfram_engine" -> "15.0.0",
  "sample_rate_hz" -> sr,
  "units" -> "linear amplitude, Hz, s",
  "calibration" -> "unit-amplitude cosine bank, no perceptual weighting",
  "duration_s" -> dur,
  "seed" -> seed,
  "frames" -> n,
  "components_hz" -> {110., 165., 220.},
  "amplitudes" -> {0.5, 0.3, 0.2},
  "samples_head" -> Take[samples, 16],
  "samples_tail" -> Take[samples, -8],
  "energy" -> energy,
  "energy_closed_form" -> energyClosed,
  "peak" -> peak,
  "rms" -> rms,
  "dtft_closed_form" -> dtftClosed
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
