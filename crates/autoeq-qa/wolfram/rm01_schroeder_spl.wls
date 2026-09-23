(* autoeq-qa case autoeq-qa.rm01-schroeder-spl.v1 (catalogue family RM01).
   Schroeder's engineering formula f_S = 2000*Sqrt[RT60/V] with V from
   explicit room dimensions (default RT60 = 0.4 s), and the SPL
   calibration forward/inverse pair dbspl = 20*Log10[Max[peak, eps32]] +
   offset / peak = Clip[10^((target-offset)/20), {0,1}] with single-
   precision epsilon 2^-23. Compared against
   roomeq_model::RoomDimensions and SplCalibration. *)

Off[General::munfl];
eps32 = N[2^-23, 16];
rooms = {{5.2, 4.1, 2.8, 0.4}, {6.0, 5.0, 3.0, 0.25}, {4.0, 3.5, 2.6, 0.6}};
schroeder[{l_, w_, h_, rt_}] := N[2000*Sqrt[rt/(l*w*h)], 16];
schroederHz = schroeder /@ rooms;
degenerateRooms = {{0., 4.1, 2.8, 0.4}, {5.2, 4.1, 2.8, 0.}, {5.2, 4.1, 2.8, -0.3}};

offset = 100.;
peaks = {1., 0.5, 0.25, 0.001, 1.*^-9};
forward[p_] := N[20*Log10[Max[p, eps32]] + offset, 16];
splDb = forward /@ peaks;
targets = {100., 94., 80., 40., 110.};
inverse[t_] := Min[Max[N[10^((t - offset)/20), 16], 0.], 1.];
peakLevels = inverse /@ targets;

payload = <|
  "schema_version" -> 1,
  "case" -> "autoeq-qa.rm01-schroeder-spl.v1",
  "family" -> "RM01",
  "wolfram_engine" -> "15.0.0",
  "formula" -> "f_S = 2000*Sqrt[RT60/V]; dbspl = 20*Log10[peak]+offset",
  "rooms_lwh_rt" -> rooms,
  "schroeder_hz" -> schroederHz,
  "degenerate_rooms_lwh_rt" -> degenerateRooms,
  "spl_offset_db" -> offset,
  "peak_levels" -> peaks,
  "spl_db" -> splDb,
  "spl_targets" -> targets,
  "peak_for_target" -> peakLevels
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
