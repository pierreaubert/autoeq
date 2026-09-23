(* autoeq-qa case autoeq-qa.re13-mso-seat-response.v1 (catalogue family RE13).
   Independent complex matrix-vector product per frequency/seat: two
   subwoofers drive two seats; each path applies its gain (dB) and delay
   (ms, negative phase slope) to the measured phasor, then seats sum
   coherently. A shared EQ curve (dB) is added after the sum and must
   preserve relative seat-to-seat differences. Direct sums only; never
   transliterated from the Rust implementation. *)

Off[General::munfl];
sr = 48000.;
freqs = {40., 60., 80., 100., 140., 180.};
gainsDb = {0., -6.};
delaysMs = {0., 2.};
sub0Spl = 80.;
sub0Phase = 0.;
sub1Spl = {77., 74.};
sub1Phase = {-20., 15.};
eqDb = {1., -1., 0.5, -0.5, 1.5, -2.};

path[spl_, phDeg_, gainDb_, delayMs_, f_] :=
  N[10^((spl + gainDb)/20), 16] *
   Exp[I N[(phDeg - 360. f delayMs/1000.) Degree, 16]];
seatSum[seat_] := Table[
   path[sub0Spl, sub0Phase, gainsDb[[1]], delaysMs[[1]], freqs[[i]]] +
    path[sub1Spl[[seat]], sub1Phase[[seat]], gainsDb[[2]], delaysMs[[2]],
      freqs[[i]]],
   {i, Length[freqs]}];
z0 = seatSum[1];
z1 = seatSum[2];
splOf[z_] := N[20 Log10[Max[Abs[z], 10^-12]], 16];
complexPair[value_] := {N[Re[value], 16], N[Im[value], 16]};
payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.re13-mso-seat-response.v1",
  "family" -> "RE13",
  "sample_rate_hz" -> sr,
  "freqs_hz" -> freqs,
  "grid" -> "shared identical measurement grid (no resampling)",
  "units" -> <|"spl" -> "dB", "phase" -> "degrees", "delay" -> "ms"|>,
  "calibration" -> "mic-1",
  "gains_db" -> gainsDb,
  "delays_ms" -> delaysMs,
  "sub0_spl_db" -> sub0Spl,
  "sub0_phase_deg" -> sub0Phase,
  "sub1_spl_db_per_seat" -> sub1Spl,
  "sub1_phase_deg_per_seat" -> sub1Phase,
  "seat_response_re_im" -> {complexPair[#] & /@ z0, complexPair[#] & /@ z1},
  "seat_spl_db" -> {splOf[#] & /@ z0, splOf[#] & /@ z1},
  "shared_eq_db" -> eqDb,
  "corrected_seat_spl_db" -> {(splOf[#] & /@ z0) + eqDb, (splOf[#] & /@ z1) + eqDb}
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
