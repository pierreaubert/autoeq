(* autoeq-qa case autoeq-qa.re05-fir-residual.v1 (catalogue family RE05).
   Residual-target reconstruction (target minus measurement, dB domain) plus
   realized candidate FIR evaluated by an explicit finite DTFT sum, then
   applied as corrected = meas + 20 log10|H|. The symmetric-tap candidate is
   linear-phase; the oracle checks arg(H) + w*(N-1)/2 against 0/Pi explicitly
   so excess-phase agreement is never manufactured from magnitudes alone. *)

Off[General::munfl];
sr = 48000.;
freqs = {40., 100., 400., 1000., 4000., 12000.};
meas = N[72. - 8.*Exp[-((Log[2, #/90.])/0.6)^2] + 5.*Exp[-((Log[2, #/2500.])/0.8)^2] & /@ freqs, 16];
target = N[Table[72., {Length[freqs]}], 16];
residual = N[target - meas, 16];
taps = {0.05, 0.12, 0.20, 0.26, 0.26, 0.20, 0.12, 0.05};
ntaps = Length[taps];
dtft[f_] := N[Sum[taps[[k]]*Exp[-I*2 Pi*f*(k - 1)/sr], {k, ntaps}], 16];
h = dtft /@ freqs;
complexPair[v_] := {N[Re[v], 16], N[Im[v], 16]};
magDb[v_] := N[20*Log10[Abs[v]], 16];
corrected = N[meas + (magDb /@ h), 16];
minMagDb = N[Min[magDb /@ h], 16];
linPhaseDev = N[Max[Table[Module[{w = N[2 Pi*freqs[[i]]/sr, 16], ph = N[Arg[h[[i]]], 16], c},
  c = Mod[ph + w*(ntaps - 1)/2, 2 Pi];
  Min[c, Abs[c - Pi], Abs[c - 2 Pi]]], {i, Length[freqs]}]], 16];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re05-fir-residual.v1",
  "family" -> "RE05",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "units" -> "dB",
  "calibration" -> "nominal curve levels in dB; no absolute SPL calibration",
  "meas_db" -> meas,
  "target_db" -> target,
  "residual_db" -> residual,
  "taps" -> taps,
  "fir_re_im" -> (complexPair /@ h),
  "corrected_db" -> corrected,
  "min_fir_mag_db" -> minMagDb,
  "linear_phase_dev_rad" -> linPhaseDev
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
