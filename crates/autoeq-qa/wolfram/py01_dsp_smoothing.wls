(* autoeq-qa case autoeq-qa.py01-dsp-smoothing.v1 (catalogue family PY01).
   Independent evaluation of the triangular-log viewer smoothing used by
   scripts/src/dsp.py smooth_octave: for each center f_c, window
   ratio = 2^(of/2), weight(f) = max(0, 1 - |log10(f)-log10(f_c)|/log10(ratio))
   over grid points inside [f_c/ratio, f_c*ratio], smoothed = weighted mean.
   Published-definition oracle; never transliterates the Python loops. *)

Off[General::munfl];
freqs = {100., 105., 110., 120., 140.};
spl = {70., 72., 71., 73., 72.};
of = 1./3.;
ratio = N[2^(of/2), 16];
halfWidth = N[Log10[ratio], 16];
weight[f_, fc_] := Module[{d = N[Abs[Log10[f] - Log10[fc]], 16]},
  Max[0., 1. - d/halfWidth]];
smoothAt[i_] := Module[{fc = freqs[[i]], sel, ws, vs},
  sel = Select[Range[Length[freqs]],
    (freqs[[#]] >= fc/ratio && freqs[[#]] <= fc*ratio) &];
  ws = weight[freqs[[#]], fc] & /@ sel;
  vs = spl[[#]] & /@ sel;
  If[Total[ws] > 0, (ws.vs)/Total[ws], spl[[i]]]];
smoothed = N[smoothAt[#] & /@ Range[Length[freqs]], 16];
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.py01-dsp-smoothing.v1",
  "family" -> "PY01",
  "sample_rate_hz" -> 48000.,
  "grid_hz" -> freqs,
  "units" -> "dB SPL",
  "calibration" -> "triangular-log viewer weights, ratio=2^(of/2)",
  "octave_fraction" -> N[of, 16],
  "spl_db" -> spl,
  "smoothed_db" -> smoothed
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
