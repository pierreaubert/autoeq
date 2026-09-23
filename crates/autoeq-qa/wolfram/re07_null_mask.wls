(* autoeq-qa case autoeq-qa.re07-null-mask.v1 (catalogue family RE07).
   Independent evaluation of the per-driver null-mask rule (the versioned
   contract in roomeq-engine fir/per_driver.rs, never a transliteration of
   its iterators): a bin is unboostable when its SPL sits more than 6 dB
   below the half-octave neighbourhood maximum envelope, or when its
   coherence gamma^2 is below 0.8 (non-finite counts as masked), or when
   its SNR against the noise floor is below 10 dB. Envelope and gates are
   written declaratively with Select/Max. Fixture: third-octave grid with
   one deep null (-14 dB), one shallow dip (-4 dB, must stay clean and
   proves the 6 dB threshold), one low-coherence bin, one low-SNR bin,
   and a 0.79/0.81 coherence boundary pair proving the 0.8 gate. *)

Off[General::munfl];
sr = 48000.;
freqs = Table[N[20*2^(i/3), 16], {i, 0, 11}];
spl = Table[80., {12}];
spl[[6]] = 66.;
spl[[9]] = 76.;
coh = Table[0.98, {12}];
coh[[4]] = 0.5;
coh[[10]] = 0.79;
coh[[11]] = 0.81;
noise = spl - 25.;
noise[[8]] = spl[[8]] - 5.;

inBand[f_, g_] := Abs[N[Log2[g/f], 16]] <= 0.5;
envelope = Table[
  N[Max[Pick[spl, inBand[#, freqs[[i]]] & /@ freqs]], 16],
  {i, 1, 12}];
mask = Table[
  If[spl[[i]] < envelope[[i]] - 6. || coh[[i]] < 0.8 ||
    spl[[i]] - noise[[i]] < 10., 1, 0],
  {i, 1, 12}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re07-null-mask.v1",
  "family" -> "RE07",
  "sample_rate_hz" -> sr,
  "grid" -> "third-octave 20 Hz .. 20*2^(11/3) Hz, 12 points",
  "units" -> "Hz, dB SPL, coherence gamma^2",
  "calibration" -> "absolute calibrated branch SPL; flat 80 dB with declared null/dip bins",
  "rule" -> "masked iff spl < half-octave-max - 6 dB, or coherence < 0.8, or spl - noise < 10 dB",
  "freqs_hz" -> freqs,
  "spl_db" -> spl,
  "coherence" -> coh,
  "noise_floor_db" -> noise,
  "envelope_db" -> envelope,
  "mask" -> mask,
  "protected_bins" -> Total[mask]
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
