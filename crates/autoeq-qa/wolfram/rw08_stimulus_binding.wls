(* autoeq-qa case autoeq-qa.rw08-stimulus-binding.v1 (catalogue family RW08).
   Independent derivation of delivered listening/capture/export adapter
   identities: the stimulus hash is SHA-256 over the exact rendered PCM
   bytes (computed here by the engine's own Hash, never by Rust
   sha256_hex); routing labels come from an independent presentation
   table; the LFE gain-stage count is an independent Count over the
   planned route gain pairs (the no-double-LFE invariant: exactly one
   stage owns the LFE channel). Catches hash-algorithm swaps, label
   swaps, and duplicated/missing LFE gain stages. *)

Off[General::munfl];
pcmBytes = Range[0, 15];
hashHex = IntegerString[Hash[ByteArray[pcmBytes], "SHA256"], 16, 64];
sampleRate = 48000.;
calibration = "spl-cal-94db";

labelFor[p_] := Switch[p,
  "SingleSpeakerMono", "mono",
  "IdenticalLrSum", "lr-sum",
  "Spatial", "spatial",
  _, "unknown"];
presentations = {"SingleSpeakerMono", "IdenticalLrSum", "Spatial"};
labels = (labelFor[#] & /@ presentations);

lfeChannel = "LFE";
routes = {
  <|"route" -> "Isolated:L", "gains" -> {{"L", 0.0}}|>,
  <|"route" -> "Lfe:sub", "gains" -> {{lfeChannel, -6.0}, {"sub", -1.0}}|>
};
gainPairs[r_] := r[["gains"]];
expectedLfeStages = Count[Flatten[gainPairs /@ routes, 1], {lfeChannel, _}];
normGains[g_] := ({#[[1]], N[#[[2]], 16]} & /@ g);
normRoute[r_] := <|"route" -> r[["route"]], "gains" -> normGains[r[["gains"]]]|>;


payload = <|
  "schema_version" -> 1,
  "wolfram_engine" -> "15.0.0",
  "case" -> "autoeq-qa.rw08-stimulus-binding.v1",
  "family" -> "RW08",
  "sample_rate_hz" -> sampleRate,
  "grid" -> "n/a: byte-identity binding over 16 rendered PCM bytes",
  "units" -> "SHA-256 hex digest, routing labels, stage counts",
  "calibration" -> calibration,
  "processing_state" -> "final-delivered",
  "baseline_graph_id" -> "graph-baseline-immutable",
  "candidate_graph_id" -> "graph-candidate-immutable",
  "full_graph_id" -> "graph-full-immutable",
  "pruned_graph_id" -> "graph-pruned-immutable",
  "programme_id" -> "programme-pink-noise-30s",
  "programme_wav_hash" -> "wav-source-hash-immutable",
  "presentation" -> "Spatial",
  "matching_method" -> "loudness-matched-at-1khz",
  "matched_within_db" -> 0.2,
  "absolute_level_db_spl" -> 76.0,
  "pcm_bytes" -> pcmBytes,
  "stimulus_hash" -> hashHex,
  "presentations" -> presentations,
  "route_labels" -> labels,
  "lfe_channel" -> lfeChannel,
  "routes" -> (normRoute /@ routes),
  "expected_lfe_stages" -> expectedLfeStages
|>;
WriteString[$Output, ExportString[payload, "RawJSON", "Compact" -> True], "\n"];
