(* autoeq-qa case autoeq-qa.ar01-impulse-store.v1 (catalogue family AR01).
   Artifact-transport fixture of record: a 16-sample impulse
   h[n] = 0.9^n Cos[2 Pi 3 n/16], with channel/rate metadata. The Rust
   test stores these exact samples (f32 little-endian bytes plus a
   metadata sidecar) through the artifact API and reloads them
   byte-exact. No signal transform is asserted here; this case pins
   transport identity only. *)

Off[General::munfl];
nFrames = 16;
rate = 48000.;
channels = 1;
taps = Table[N[0.9^n*Cos[2 Pi*3*n/16], 16], {n, 0, nFrames - 1}];

payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.ar01-impulse-store.v1",
  "family" -> "AR01",
  "sample_rate_hz" -> rate,
  "units" -> <|"taps" -> "linear-amplitude"|>,
  "calibration" -> "fixture-of-record-no-acoustic-calibration",
  "frames" -> nFrames,
  "channels" -> channels,
  "encoding" -> "f32-le",
  "impulse" -> taps
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
