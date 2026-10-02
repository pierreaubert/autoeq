#[cfg(test)]
#[path = "../../tests/common/apo.rs"]
mod apo;

#[cfg(test)]
mod tests {
    use super::apo::parse_apo_filters;
    use crate::autoeq_command::save::{
        ProductExportContext, publish_profiled_pair_with_test_hook, save_peq_to_file,
        save_profiled_apo_to_file,
    };
    use autoeq::cli::Args;
    use autoeq::loss::LossType;
    use clap::Parser;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_peq_to_file_apo_format() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);

        // Example optimized parameters (3 filters)
        // Note: Frequency parameters must be in log10 scale
        let x = vec![
            500.0f64.log10(),
            2.0,
            -3.0, // Filter 1
            1000.0f64.log10(),
            5.0,
            2.0, // Filter 2
            3000.0f64.log10(),
            3.0,
            -1.0, // Filter 3
        ];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("speaker-flat filters should save");

        // Verify file was created
        let apo_path = output_path.parent().unwrap().join("iir-autoeq-flat.txt");
        assert!(apo_path.exists());

        // Verify content
        let content = fs::read_to_string(&apo_path).unwrap();
        assert!(content.contains("AutoEQ"));
        let filters = parse_apo_filters(&content).expect("generated APO must parse");
        assert_eq!(filters.len(), 3);
        for (actual, (index, freq_hz, gain_db, q)) in filters.iter().zip([
            (1, 500.0, -3.0, 2.0),
            (2, 1000.0, 2.0, 5.0),
            (3, 3000.0, -1.0, 3.0),
        ]) {
            assert_eq!(actual.index, index);
            assert_eq!(actual.kind, "PK");
            assert!(
                (actual.freq_hz - freq_hz).abs() < 1e-9,
                "frequency: actual={}, expected={freq_hz}",
                actual.freq_hz
            );
            assert!(
                (actual.gain_db - gain_db).abs() < 1e-9,
                "gain: actual={}, expected={gain_db}",
                actual.gain_db
            );
            assert!(
                (actual.q - q).abs() < 1e-9,
                "Q: actual={}, expected={q}",
                actual.q
            );
        }
    }

    #[tokio::test]
    async fn test_save_peq_to_file_score_loss() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-score"]);

        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerScore, None)
            .await
            .expect("speaker-score filters should save");

        // Verify filename for score loss
        let apo_path = output_path.parent().unwrap().join("iir-autoeq-score.txt");
        assert!(apo_path.exists());
        let content = fs::read_to_string(&apo_path).unwrap();
        let filters = parse_apo_filters(&content).expect("generated APO must parse");
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].kind, "PK");
        assert!(
            (filters[0].freq_hz - 500.0).abs() < 1e-9,
            "frequency: actual={}, expected=500",
            filters[0].freq_hz
        );
        assert!((filters[0].gain_db + 2.0).abs() < 1e-9);
        assert!((filters[0].q - 2.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_save_peq_creates_multiple_formats() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);

        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("all configured filter formats should save");

        // Check APO format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.txt")
                .exists()
        );

        // Check RME format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.tmreq")
                .exists()
        );

        // Check Apple format
        assert!(
            output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat.aupreset")
                .exists()
        );
    }

    #[tokio::test]
    async fn test_save_peq_with_pareto_export_writes_sidecar() {
        use crate::autoeq_command::save::ParetoExport;
        use autoeq::optim::pareto::ParetoFilter;

        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);
        let x = vec![500.0f64.log10(), 2.0, -2.0];

        let filters = vec![
            ParetoFilter {
                params: x.clone(),
                flatness_loss: 1.5,
                score_loss: Some(0.7),
                num_filters: 1,
                converged: true,
            },
            ParetoFilter {
                params: x.clone(),
                flatness_loss: 1.2,
                score_loss: None,
                num_filters: 2,
                converged: false,
            },
        ];
        let export = ParetoExport::from_pareto_filters(
            &filters,
            vec!["flatness_loss".to_string(), "score_loss".to_string()],
            "fewest filters within tolerance",
            0,
            vec![
                vec![("mic-1".to_string(), 1.4), ("mic-2".to_string(), 1.6)],
                vec![("mic-1".to_string(), 1.1)],
            ],
        );

        save_peq_to_file(
            &args,
            &x,
            &output_path,
            &LossType::SpeakerFlat,
            Some(&export),
        )
        .await
        .expect("preset with pareto export should save");

        let sidecar = output_path
            .parent()
            .unwrap()
            .join("iir-autoeq-flat-pareto.json");
        assert!(
            sidecar.exists(),
            "pareto sidecar must sit next to the preset"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(
            parsed["objective_labels"],
            serde_json::json!(["flatness_loss", "score_loss"])
        );
        assert_eq!(
            parsed["selection_policy"],
            serde_json::json!("fewest filters within tolerance")
        );
        assert_eq!(parsed["selected_index"], serde_json::json!(0));
        assert_eq!(
            parsed["candidates"][0]["objectives"],
            serde_json::json!([1.5, 0.7])
        );
        assert!(parsed["candidates"][1]["objectives"][1].is_null());
        assert_eq!(
            parsed["candidates"][0]["per_measurement_losses"],
            serde_json::json!([["mic-1", 1.4], ["mic-2", 1.6]])
        );
        assert_eq!(parsed["candidates"][1]["num_filters"], serde_json::json!(2));
    }

    #[tokio::test]
    async fn test_save_peq_without_pareto_writes_no_sidecar() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_output");

        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);
        let x = vec![500.0f64.log10(), 2.0, -2.0];

        save_peq_to_file(&args, &x, &output_path, &LossType::SpeakerFlat, None)
            .await
            .expect("preset without pareto export should save");

        assert!(
            !output_path
                .parent()
                .unwrap()
                .join("iir-autoeq-flat-pareto.json")
                .exists(),
            "no sidecar expected without pareto metadata"
        );
    }

    #[tokio::test]
    async fn profiled_apo_export_reports_the_parameters_parsed_from_its_text() {
        use autoeq::iir::{Biquad, BiquadFilterType};
        use autoeq::workflow::{
            DeviceProfile, DeviceRange, PreparedProduct, ProductMode, ProductRenderer,
            ProductRequest, ProductSource, ProductTarget, TargetCompatibilityStatus, TargetProfile,
        };

        let temp_dir = TempDir::new().unwrap();
        let source_path = temp_dir.path().join("source.csv");
        let target_path = temp_dir.path().join("target.csv");
        fs::write(
            &source_path,
            "frequency,spl\n20,70\n100,71\n1000,72\n20000,73\n",
        )
        .unwrap();
        fs::write(
            &target_path,
            "frequency,spl\n20,0\n100,0\n1000,0\n20000,0\n",
        )
        .unwrap();
        let source_record = autoeq::read::read_record_from_csv(&source_path).unwrap();
        let target_record = autoeq::read::read_record_from_csv(&target_path).unwrap();
        let target_profile = TargetProfile::new(target_record, vec![]).unwrap();
        let compatibility = target_profile.assess_compatibility(&source_record);
        assert_eq!(compatibility.status, TargetCompatibilityStatus::Unknown);
        let profile = DeviceProfile {
            id: "test-playback-chain".into(),
            playback_device_id: "coreaudio:test-output".into(),
            renderer: ProductRenderer::EqualizerApo,
            sample_rate_hz: 48_000.0,
            maximum_filter_count: 4,
            supported_peq_models: vec!["pk".into()],
            supported_filter_types: vec!["PK".into(), "LSC".into(), "HSC".into()],
            frequency_hz: DeviceRange {
                minimum: 20.0,
                maximum: 20_000.0,
            },
            q: DeviceRange {
                minimum: 0.5,
                maximum: 10.0,
            },
            gain_db: DeviceRange {
                minimum: -12.0,
                maximum: 12.0,
            },
            preamp_db: Some(-3.56),
        };
        let request = ProductRequest {
            mode: ProductMode::Speaker,
            source: ProductSource::Csv {
                path: source_path,
                measurement_rig: None,
            },
            target: ProductTarget::Csv {
                path: target_path,
                supported_measurement_rigs: vec![],
            },
            device_profile: profile.clone(),
            reject_declared_target_mismatch: false,
        };
        let prepared = PreparedProduct {
            mode: ProductMode::Speaker,
            source_record,
            target_profile,
            target_compatibility: compatibility,
            spin_curves: None,
        };
        let filter = Biquad::new(BiquadFilterType::Peak, 500.49, 48_000.0, 1.236, -3.456);
        let realized_filters = profile.apo_serialized_filters(48_000.0, &[filter]).unwrap();
        let realized_preamp = profile.apo_serialized_preamp_db().unwrap();
        let args = Args::parse_from(["autoeq-test", "--loss", "speaker-flat"]);
        let output_path = temp_dir.path().join("result");

        save_profiled_apo_to_file(
            &args,
            &realized_filters,
            realized_preamp,
            &output_path,
            &LossType::SpeakerFlat,
            ProductExportContext {
                request: &request,
                prepared: &prepared,
                compatibility: &prepared.target_compatibility,
                max_filter_transfer_delta_db: 0.0,
                verification_frequencies_hz: &[50.0, 100.0, 500.0, 1_000.0, 10_000.0],
            },
        )
        .await
        .unwrap();

        let apo_path = temp_dir.path().join("iir-autoeq-flat.txt");
        let text = fs::read_to_string(&apo_path).unwrap();
        assert!(text.lines().any(|line| line == "Preamp: -3.6 dB"));
        let parsed = parse_apo_filters(&text).expect("profiled APO text should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].freq_hz, 500.0);
        assert_eq!(parsed[0].q, 1.24);
        assert_eq!(parsed[0].gain_db, -3.46);

        let parsed_filter = Biquad::new(
            BiquadFilterType::Peak,
            parsed[0].freq_hz,
            48_000.0,
            parsed[0].q,
            parsed[0].gain_db,
        );
        for frequency in [50.0, 100.0, 500.0, 1_000.0, 10_000.0] {
            assert!(
                (parsed_filter.log_result(frequency) - realized_filters[0].log_result(frequency))
                    .abs()
                    < 1e-12
            );
        }

        let provenance_path = temp_dir
            .path()
            .join("iir-autoeq-flat.product-provenance.json");
        let preset_path = temp_dir.path().join("iir-autoeq-flat.txt");
        let preset_bytes = fs::read(&preset_path).unwrap();
        let sidecar_bytes = fs::read(&provenance_path).unwrap();
        autoeq::workflow::verify_apo_preset_binding(&preset_bytes, &sidecar_bytes).unwrap();
        let mut edited_preset = preset_bytes.clone();
        edited_preset.extend_from_slice(b"# altered\n");
        assert!(
            autoeq::workflow::verify_apo_preset_binding(&edited_preset, &sidecar_bytes).is_err()
        );

        let shelves = [
            Biquad::new(BiquadFilterType::Lowshelf, 100.0, 48_000.0, 0.71, 3.0),
            Biquad::new(BiquadFilterType::Highshelf, 10_000.0, 48_000.0, 1.37, -3.0),
        ];
        save_profiled_apo_to_file(
            &args,
            &shelves,
            realized_preamp,
            &output_path,
            &LossType::SpeakerFlat,
            ProductExportContext {
                request: &request,
                prepared: &prepared,
                compatibility: &prepared.target_compatibility,
                max_filter_transfer_delta_db: 0.0,
                verification_frequencies_hz: &[50.0, 100.0, 500.0, 1_000.0, 10_000.0],
            },
        )
        .await
        .expect("verified 12 dB LSC/HSC shelves should publish");
        let shelf_preset_bytes = fs::read(&preset_path).unwrap();
        let shelf_sidecar_bytes = fs::read(&provenance_path).unwrap();
        let shelf_text = String::from_utf8(shelf_preset_bytes.clone()).unwrap();
        assert!(
            shelf_text
                .lines()
                .any(|line| line == "Filter 1: ON LSC 12 dB Fc 100 Hz Gain +3.00 dB")
        );
        assert!(
            shelf_text
                .lines()
                .any(|line| line == "Filter 2: ON HSC 12 dB Fc 10000 Hz Gain -3.00 dB")
        );
        assert!(!shelf_text.contains(" ON LS ") && !shelf_text.contains(" ON HS "));
        autoeq::workflow::verify_apo_preset_binding(&shelf_preset_bytes, &shelf_sidecar_bytes)
            .unwrap();
        let shelf_manifest: serde_json::Value =
            serde_json::from_slice(&shelf_sidecar_bytes).unwrap();
        assert_eq!(shelf_manifest["schema_version"], 3);
        assert_eq!(shelf_manifest["realized_filters"][0]["type"], "LSC");
        assert_eq!(
            shelf_manifest["realized_filters"][0]["q"],
            serde_json::Value::Null
        );
        assert_eq!(
            shelf_manifest["realized_filters"][0]["slope_db_per_octave"],
            12
        );
        assert_eq!(
            shelf_manifest["realized_filters"][0]["frequency_convention"],
            "center_frequency_fc"
        );
        assert_eq!(shelf_manifest["realized_filters"][1]["type"], "HSC");
        assert_eq!(
            shelf_manifest["realized_filters"][1]["q"],
            serde_json::Value::Null
        );
        assert_eq!(
            shelf_manifest["realized_filters"][1]["slope_db_per_octave"],
            12
        );
        assert_eq!(
            shelf_manifest["realized_filters"][1]["frequency_convention"],
            "center_frequency_fc"
        );
        assert_eq!(
            shelf_manifest["apo_serialization"]["emitted_text_verification"]["source_shelf_contract"]
                ["equalizer_apo_source_revision"],
            "bbfcc3e5024cbb9d61ba75fc88d78605cc4c9687"
        );
        assert_eq!(
            shelf_manifest["apo_serialization"]["emitted_text_verification"]["source_shelf_contract"]
                ["max_scaled_coefficient_delta_epsilon"],
            16
        );
        assert_eq!(
            shelf_manifest["apo_serialization"]["emitted_text_verification"]["source_shelf_contract"]
                ["max_sampled_transfer_delta_db"],
            1.0e-10
        );
        assert!(
            shelf_manifest["apo_serialization"]["emitted_text_verification"]["max_shelf_scaled_coefficient_delta"]
                .as_f64()
                .is_some_and(|delta| delta <= 16.0 * f64::EPSILON)
        );
        assert!(
            shelf_manifest["apo_serialization"]["emitted_text_verification"]["max_shelf_source_transfer_delta_db"]
                .as_f64()
                .is_some_and(|delta| delta <= 1.0e-10)
        );

        let unsupported = Biquad::new(BiquadFilterType::Bandpass, 1_000.0, 48_000.0, 1.0, 0.0);
        let unsupported_refusal = save_profiled_apo_to_file(
            &args,
            &[unsupported],
            realized_preamp,
            &output_path,
            &LossType::SpeakerFlat,
            ProductExportContext {
                request: &request,
                prepared: &prepared,
                compatibility: &prepared.target_compatibility,
                max_filter_transfer_delta_db: 0.0,
                verification_frequencies_hz: &[50.0, 100.0, 500.0, 1_000.0, 10_000.0],
            },
        )
        .await
        .expect_err("unsupported stage must fail before replacing the bound pair");
        assert!(
            unsupported_refusal
                .to_string()
                .contains("unverified filter type")
        );
        assert_eq!(fs::read(&preset_path).unwrap(), shelf_preset_bytes);
        assert_eq!(fs::read(&provenance_path).unwrap(), shelf_sidecar_bytes);

        let provenance: serde_json::Value = serde_json::from_slice(&sidecar_bytes).unwrap();
        assert_eq!(provenance["schema_version"], 3);
        assert_eq!(provenance["target_compatibility"]["status"], "unknown");
        assert_eq!(provenance["apo_serialization"]["realized_preamp_db"], -3.6);
        assert_eq!(
            provenance["apo_serialization"]["max_filter_transfer_delta_db"],
            0.0
        );
        assert_eq!(provenance["realized_filters"][0]["frequency_hz"], 500.0);
        assert_eq!(provenance["realized_filters"][0]["q"], 1.24);
        assert_eq!(provenance["realized_filters"][0]["gain_db"], -3.46);
        assert_eq!(
            provenance["apo_serialization"]["emitted_text_verification"]["sample_rate_hz"],
            48_000.0
        );
        assert_eq!(
            provenance["apo_serialization"]["emitted_text_verification"]["max_transfer_delta_db"],
            0.0
        );
        assert_eq!(
            provenance["apo_serialization"]["emitted_text_verification"]["channel_scope"],
            "inherited_from_including_equalizer_apo_configuration"
        );
        assert_eq!(
            provenance["apo_serialization"]["emitted_text_verification"]["consumer_parser_used"],
            false
        );
        assert!(!temp_dir.path().join("iir-autoeq-flat.tmreq").exists());
        assert!(!temp_dir.path().join("iir-autoeq-flat.aupreset").exists());

        assert_eq!(prepared.source_record.curve.freq.len(), 4);

        // Profile fields become APO comments. An embedded command must be
        // rejected before either previously published file changes.
        let mut injected_request = request.clone();
        injected_request.device_profile.id = "profile\nInclude: injected.txt".into();
        let error = save_profiled_apo_to_file(
            &args,
            &realized_filters,
            realized_preamp,
            &output_path,
            &LossType::SpeakerFlat,
            ProductExportContext {
                request: &injected_request,
                prepared: &prepared,
                compatibility: &prepared.target_compatibility,
                max_filter_transfer_delta_db: 0.0,
                verification_frequencies_hz: &[50.0, 100.0, 500.0, 1_000.0, 10_000.0],
            },
        )
        .await
        .expect_err("comment-injected APO commands cannot be verified or published");
        assert!(error.to_string().contains("unsupported command"));
        assert_eq!(fs::read(&preset_path).unwrap(), shelf_preset_bytes);
        assert_eq!(fs::read(&provenance_path).unwrap(), shelf_sidecar_bytes);
    }

    #[tokio::test]
    async fn sidecar_publish_failure_restores_the_previous_bound_preset_pair() {
        let temp_dir = TempDir::new().unwrap();
        let preset_path = temp_dir.path().join("iir-autoeq-flat.txt");
        let sidecar_path = temp_dir
            .path()
            .join("iir-autoeq-flat.product-provenance.json");
        let old_preset = b"old preset bytes\n";
        let old_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": autoeq_artifacts::sha256_hex(old_preset)
            }
        }))
        .unwrap();
        fs::write(&preset_path, old_preset).unwrap();
        fs::write(&sidecar_path, &old_sidecar).unwrap();

        let new_preset = b"new preset bytes\n";
        let new_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": autoeq_artifacts::sha256_hex(new_preset)
            }
        }))
        .unwrap();
        let error = publish_profiled_pair_with_test_hook(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            || Err(std::io::Error::other("injected sidecar failure")),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("prior APO preset was restored"));
        assert_eq!(fs::read(&preset_path).unwrap(), old_preset);
        assert_eq!(fs::read(&sidecar_path).unwrap(), old_sidecar);
        autoeq::workflow::verify_apo_preset_binding(
            &fs::read(&preset_path).unwrap(),
            &fs::read(&sidecar_path).unwrap(),
        )
        .unwrap();

        let concurrent_preset = b"concurrent replacement\n";
        let error = publish_profiled_pair_with_test_hook(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            || {
                fs::write(&preset_path, concurrent_preset)?;
                Err(std::io::Error::other("injected sidecar failure"))
            },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("changed concurrently"));
        assert_eq!(fs::read(&preset_path).unwrap(), concurrent_preset);
        assert_eq!(fs::read(&sidecar_path).unwrap(), old_sidecar);
    }

    #[tokio::test]
    async fn profiled_export_refuses_to_replace_an_oversized_prior_preset() {
        let temp_dir = TempDir::new().unwrap();
        let preset_path = temp_dir.path().join("iir-autoeq-flat.txt");
        let sidecar_path = temp_dir
            .path()
            .join("iir-autoeq-flat.product-provenance.json");
        let prior_preset = vec![b'x'; 16 * 1024 * 1024 + 1];
        fs::write(&preset_path, &prior_preset).unwrap();
        fs::write(&sidecar_path, b"prior sidecar").unwrap();

        let new_preset = b"new preset bytes\n";
        let new_sidecar = serde_json::to_vec(&serde_json::json!({
            "apo_serialization": {
                "preset_sha256": autoeq_artifacts::sha256_hex(new_preset)
            }
        }))
        .unwrap();
        let error = publish_profiled_pair_with_test_hook(
            &preset_path,
            new_preset,
            &sidecar_path,
            &new_sidecar,
            || Ok(()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("rollback safety limit"));
        assert_eq!(
            fs::metadata(&preset_path).unwrap().len(),
            prior_preset.len() as u64
        );
        assert_eq!(fs::read(&sidecar_path).unwrap(), b"prior sidecar");
    }

    #[test]
    fn test_apo_roundtrip_gap_small_for_integer_hz_filters() {
        use crate::autoeq_command::save::{
            APO_ROUNDTRIP_WARN_THRESHOLD, apo_roundtrip_objective_gap,
        };
        use autoeq::PeqModel;
        use autoeq::optim::{ObjectiveData, ObjectiveDataBuilder};
        use ndarray::Array1;

        let freqs = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let deviation = Array1::from_vec(vec![2.0, 1.5, 1.0, 1.2, 0.8]);
        let target = Array1::zeros(freqs.len());
        let objective: ObjectiveData = ObjectiveDataBuilder::new(
            freqs,
            target,
            deviation,
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .min_spacing_oct(0.1)
        .max_db(10.0)
        .min_db(-10.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 3)
        .build()
        .expect("valid test objective data");

        // Exact integer-Hz centers: serialization is (near-)lossless.
        let x = vec![500.0f64.log10(), 2.0, -3.0, 1000.0f64.log10(), 5.0, 2.0];
        let gap = apo_roundtrip_objective_gap(&x, 48000.0, PeqModel::Pk, &objective)
            .expect("finite objectives must produce a gap");
        assert!(
            gap < APO_ROUNDTRIP_WARN_THRESHOLD,
            "integer-Hz filters must stay under the warn threshold, got {gap:.3e}"
        );
    }

    #[test]
    fn test_apo_roundtrip_gap_finite_for_fractional_low_freq_high_q() {
        use crate::autoeq_command::save::apo_roundtrip_objective_gap;
        use autoeq::PeqModel;
        use autoeq::optim::{ObjectiveData, ObjectiveDataBuilder};
        use ndarray::Array1;

        let freqs = Array1::from_vec(vec![20.0, 30.0, 45.0, 60.0, 90.0, 120.0]);
        let deviation = Array1::from_vec(vec![3.0, 2.5, 2.0, 1.5, 1.0, 0.8]);
        let target = Array1::zeros(freqs.len());
        let objective: ObjectiveData = ObjectiveDataBuilder::new(
            freqs,
            target,
            deviation,
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .min_spacing_oct(0.01)
        .max_db(10.0)
        .min_db(-10.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 3)
        .build()
        .expect("valid test objective data");

        // Fractional center with high Q: worst case for integer-Hz rounding.
        let x = vec![31.7f64.log10(), 12.0, -6.0];
        let gap = apo_roundtrip_objective_gap(&x, 48000.0, PeqModel::Pk, &objective)
            .expect("finite objectives must produce a gap");
        assert!(
            gap.is_finite(),
            "low-frequency/high-Q rounding gap must stay measurable, got {gap:?}"
        );
    }
}
