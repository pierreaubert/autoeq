# autoeq-report-native

Native filesystem adapter for saved RoomEQ report inputs. This crate owns
legacy artifact staging and report hydration; `roomeq-workflow` owns bundle
verification and snapshot capture, and `roomeq-engine` owns DSP evaluation.
The shared WASM renderer remains independent of filesystem access.

Use `load_saved_bundle(path)` for a saved graph and its sibling assets, or
`load_legacy_input(bytes, artifact_directory)` for unmanifested JSON. The
returned `LoadedReportInput` retains exact graph bytes, unknown JSON fields,
the hydrated graph, indexed measurement bytes, and verified FIR resources.
`hydrated_report_json()` fills missing measurements without replacing inline
values. `hydrated_report_input_json()` also carries index provenance.

`prepare_dsp_replay(sample_rate)` computes channel electrical transfer on the
initial measurement grid. Missing or unsupported evidence is explicitly
unavailable; global routing and per-driver acoustic response are outside this
channel-only API. FIR replay requires captured, manifest-verified mono
float32 WAV data at the requested sample rate. No file is reopened during
replay or report hydration.

Legacy staging rejects symlinks, unsafe indexed paths, oversized members,
oversized trees, and detected changes during copying. Legacy data is never
promoted to manifest-verified evidence.

Run `cargo test -p autoeq-report-native --lib --all-features` for
hydration, integrity, snapshot, staging-limit, and canonical replay tests.

