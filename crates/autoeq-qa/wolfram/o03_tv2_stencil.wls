(* autoeq-qa case autoeq-qa.o03-tv2-stencil.v1 (catalogue family O03).
   Independent evaluation of the TV^2 smoothness regularizer: the nonuniform
   log10-frequency second-difference stencil
     slopeFwd = (y[i+1]-y[i])/(l[i+1]-l[i]),
     slopeBwd = (y[i]-y[i-1])/(l[i]-l[i-1]),
     curvature = (slopeFwd-slopeBwd)/((dxFwd+dxBwd)/2)
   with l = Log10[f], averaged as tv2Weight*Mean[w*|curvature|^p] over centers
   strictly inside [minFreq, maxFreq] (neighbors outside the band still enter
   the stencil through the slopes). Below schroederHz the term weight is
   modalWeightScale. This implementation is a bin-count mean, not an
   interval-weighted integral. y = 2*(Log10 f)^2 has exact curvature 4
   everywhere, computed here directly through the stencil rather than assumed.
   Compared against compute_smoothness_penalty. Units: dB over decades. *)

Off[General::munfl];
freqs = {40., 90., 200., 500., 1200., 3000., 8000.};
y = N[2*Log10[freqs]^2, 16];
tv2Weight = 0.5;
schroederHz = 500.;
modalScale = 0.1;
minF = 20.; maxF = 20000.;

l = N[Log10[freqs], 16];
m = Length[freqs];
centers = Range[2, m - 1];
curv = Table[
  Module[{dxF = l[[i + 1]] - l[[i]], dxB = l[[i]] - l[[i - 1]],
    sF = (y[[i + 1]] - y[[i]])/(l[[i + 1]] - l[[i]]),
    sB = (y[[i]] - y[[i - 1]])/(l[[i]] - l[[i - 1]])},
   N[(sF - sB)/((dxF + dxB)/2), 16]],
  {i, centers}];
centerFreqs = freqs[[centers]];
penExp1 = N[tv2Weight*Mean[Abs[curv]], 16];
penExp2 = N[tv2Weight*Mean[curv^2], 16];
wSch = Table[N[If[centerFreqs[[i]] < schroederHz, modalScale, 1.], 16], {i, Length[centers]}];
penSchroeder = N[tv2Weight*Total[wSch*Abs[curv]]/Length[centers], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.o03-tv2-stencil.v1",
  "family" -> "O03",
  "sample_rate_hz" -> Null,
  "sample_rate_note" -> "dB-per-decade curvature on a declared grid; no sampling involved",
  "grid_hz" -> freqs,
  "units" -> "dB per decade^2",
  "calibration" -> "correction-curve curvature; no SPL calibration",
  "min_freq_hz" -> minF,
  "max_freq_hz" -> maxF,
  "response_db" -> y,
  "tv2_weight" -> tv2Weight,
  "schroeder_hz" -> schroederHz,
  "modal_weight_scale" -> modalScale,
  "center_freqs_hz" -> centerFreqs,
  "curvatures" -> curv,
  "penalty_exp1" -> penExp1,
  "penalty_exp2" -> penExp2,
  "penalty_exp1_schroeder" -> penSchroeder,
  "constant_response_penalty" -> 0.,
  "affine_log_response_penalty" -> 0.
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
