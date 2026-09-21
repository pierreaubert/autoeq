//! Listening-stimulus binding for the V3 battery (Wave 3, step 7).
//!
//! Workflow renders full/pruned and baseline/corrected stimuli from
//! frozen delivered graphs and binds the rendered bytes to immutable
//! chain identities through `roomeq-quality` bindings. Any change to a
//! graph, the stimulus bytes, the sample rate, the calibration, or the
//! processing state voids the binding: re-render and re-preregister
//! instead of reusing the protocol. Perceptual proxy diagnostics stay
//! advisory until the G7 lane lands; this module packages evidence,
//! never a listening outcome.

// Rust guideline compliant 2026-02-21

use roomeq_quality::{ChainStimulusBinding, SourcePresentation, sha256_hex};

/// Level-match record for one rendered listening stimulus.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct StimulusLevelMatch {
    /// Matching method, e.g. `"loudness-matched-at-1khz"`.
    pub matching_method: String,
    /// Residual mismatch bound in dB the match guarantees.
    pub matched_within_db: f64,
    /// Absolute playback level in dB SPL the match was verified at.
    pub absolute_level_db_spl: f64,
    /// Absolute playback calibration identity.
    pub calibration_id: String,
}

impl StimulusLevelMatch {
    /// Validate the match record.
    ///
    /// # Errors
    ///
    /// Returns an error on blank provenance, non-finite levels, or a
    /// negative mismatch bound.
    pub fn validate(&self) -> Result<(), String> {
        if self.matching_method.trim().is_empty() || self.calibration_id.trim().is_empty() {
            return Err(String::from(
                "listening stimuli need a matching method and a calibration identity",
            ));
        }
        if !self.matched_within_db.is_finite() || self.matched_within_db < 0.0 {
            return Err(String::from(
                "listening mismatch bound must be finite and non-negative",
            ));
        }
        if !self.absolute_level_db_spl.is_finite() {
            return Err(String::from(
                "listening stimuli need a finite absolute playback level",
            ));
        }
        Ok(())
    }
}

/// Render specification for one listening stimulus.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ListeningRenderSpec {
    /// Immutable baseline graph identity rendered.
    pub baseline_graph_id: String,
    /// Immutable corrected candidate graph identity rendered.
    pub candidate_graph_id: String,
    /// Immutable full chain the pruned arm derives from.
    pub full_graph_id: String,
    /// Immutable pruned chain identity, when the pruned arm is staged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruned_graph_id: Option<String>,
    /// Programme material identity rendered.
    pub programme_id: String,
    /// Hash of the source programme WAV the render started from.
    pub programme_wav_hash: String,
    /// Playback sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Absolute playback calibration identity.
    pub calibration_id: String,
    /// Processing state, e.g. `"final-delivered"`.
    pub processing_state: String,
    /// Source presentation routed for this stimulus.
    pub presentation: SourcePresentation,
    /// Level matching at a recorded absolute playback level.
    pub level_match: StimulusLevelMatch,
}

impl ListeningRenderSpec {
    /// Validate the render specification.
    ///
    /// # Errors
    ///
    /// Returns an error on blank identities, a non-positive sample
    /// rate, or an unvalidated level match.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("baseline graph", &self.baseline_graph_id),
            ("candidate graph", &self.candidate_graph_id),
            ("full chain graph", &self.full_graph_id),
            ("programme", &self.programme_id),
            ("programme WAV hash", &self.programme_wav_hash),
            ("calibration", &self.calibration_id),
            ("processing state", &self.processing_state),
        ] {
            if value.trim().is_empty() {
                return Err(format!("listening render spec needs a {field} identity"));
            }
        }
        if let Some(pruned) = &self.pruned_graph_id
            && pruned.trim().is_empty()
        {
            return Err(String::from(
                "listening render spec needs a non-blank pruned graph identity when staged",
            ));
        }
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(String::from(
                "listening render spec needs a positive sample rate",
            ));
        }
        self.level_match.validate()?;
        if self.level_match.calibration_id != self.calibration_id {
            return Err(String::from(
                "listening level-match calibration must equal the render calibration",
            ));
        }
        Ok(())
    }
}

/// Bind rendered stimulus bytes to their frozen chain identities.
///
/// The stimulus hash is the SHA-256 of the rendered bytes actually
/// played: descriptors alone are not stimuli.
///
/// # Errors
///
/// Returns an error when the spec is invalid or the rendered bytes are
/// empty.
pub fn bind_rendered_stimulus(
    spec: &ListeningRenderSpec,
    rendered_pcm: &[u8],
) -> Result<ChainStimulusBinding, String> {
    spec.validate()?;
    if rendered_pcm.is_empty() {
        return Err(String::from(
            "listening binding needs rendered stimulus bytes, not a descriptor",
        ));
    }
    Ok(ChainStimulusBinding {
        baseline_graph_id: spec.baseline_graph_id.clone(),
        candidate_graph_id: spec.candidate_graph_id.clone(),
        full_graph_id: spec.full_graph_id.clone(),
        pruned_graph_id: spec.pruned_graph_id.clone(),
        stimulus_hash: sha256_hex(rendered_pcm),
        sample_rate_hz: spec.sample_rate_hz,
        calibration_id: spec.calibration_id.clone(),
        processing_state: spec.processing_state.clone(),
    })
}

/// Check a binding is still current for a render spec and bytes.
///
/// A newly finalized graph bound to stale earlier stimuli fails here,
/// as does any restated stimulus, sample rate, calibration, or
/// processing-state change.
///
/// # Errors
///
/// Returns an error when any bound identity or the stimulus hash
/// differs from the current rendering.
pub fn check_binding_current(
    spec: &ListeningRenderSpec,
    binding: &ChainStimulusBinding,
    rendered_pcm: &[u8],
) -> Result<(), String> {
    let current = bind_rendered_stimulus(spec, rendered_pcm)?;
    binding.verify_unchanged(&current)
}

/// Routing label of a presentation for condition identities.
pub fn route_label(presentation: SourcePresentation) -> &'static str {
    presentation.as_str()
}

#[cfg(test)]
mod listening_stimuli_tests {
    use super::*;

    fn matched() -> StimulusLevelMatch {
        StimulusLevelMatch {
            matching_method: String::from("loudness-matched-at-1khz"),
            matched_within_db: 0.2,
            absolute_level_db_spl: 76.0,
            calibration_id: String::from("spl-cal-94db"),
        }
    }

    fn spec(presentation: SourcePresentation) -> ListeningRenderSpec {
        ListeningRenderSpec {
            baseline_graph_id: String::from("graph-baseline-immutable"),
            candidate_graph_id: String::from("graph-candidate-immutable"),
            full_graph_id: String::from("graph-full-immutable"),
            pruned_graph_id: Some(String::from("graph-pruned-immutable")),
            programme_id: String::from("resonance-strings-01"),
            programme_wav_hash: String::from("source-wav-hash"),
            sample_rate_hz: 48_000.0,
            calibration_id: String::from("spl-cal-94db"),
            processing_state: String::from("final-delivered"),
            presentation,
            level_match: matched(),
        }
    }

    fn pcm() -> Vec<u8> {
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    }

    #[test]
    fn listening_binding_rejects_stale_graph() {
        let rendered = spec(SourcePresentation::SingleSpeakerMono);
        let binding = bind_rendered_stimulus(&rendered, &pcm()).unwrap();
        assert!(check_binding_current(&rendered, &binding, &pcm()).is_ok());
        let mut retuned = rendered.clone();
        retuned.candidate_graph_id = String::from("graph-candidate-retuned");
        assert!(check_binding_current(&retuned, &binding, &pcm()).is_err());
    }

    #[test]
    fn listening_binding_rejects_restated_stimulus() {
        let rendered = spec(SourcePresentation::Spatial);
        let binding = bind_rendered_stimulus(&rendered, &pcm()).unwrap();
        assert!(check_binding_current(&rendered, &binding, &pcm()).is_ok());
        assert!(check_binding_current(&rendered, &binding, &[9, 9, 9]).is_err());
        assert!(bind_rendered_stimulus(&rendered, &[]).is_err());
    }

    #[test]
    fn listening_level_match_required() {
        let rendered = spec(SourcePresentation::SingleSpeakerMono);
        assert!(bind_rendered_stimulus(&rendered, &pcm()).is_ok());
        let mut unmatched = rendered.clone();
        unmatched.level_match.matched_within_db = -1.0;
        assert!(bind_rendered_stimulus(&unmatched, &pcm()).is_err());
        let mut wrong_cal = rendered.clone();
        wrong_cal.level_match.calibration_id = String::from("other-cal");
        assert!(bind_rendered_stimulus(&wrong_cal, &pcm()).is_err());
    }

    #[test]
    fn listening_three_presentations_distinct() {
        assert_ne!(
            route_label(SourcePresentation::SingleSpeakerMono),
            route_label(SourcePresentation::IdenticalLrSum)
        );
        assert_ne!(
            route_label(SourcePresentation::SingleSpeakerMono),
            route_label(SourcePresentation::Spatial)
        );
        for presentation in [
            SourcePresentation::SingleSpeakerMono,
            SourcePresentation::Spatial,
            SourcePresentation::IdenticalLrSum,
        ] {
            let rendered = spec(presentation);
            let binding = bind_rendered_stimulus(&rendered, &pcm()).unwrap();
            assert!(check_binding_current(&rendered, &binding, &pcm()).is_ok());
        }
    }

    #[test]
    fn listening_changed_chain_invalidates() {
        let rendered = spec(SourcePresentation::IdenticalLrSum);
        let binding = bind_rendered_stimulus(&rendered, &pcm()).unwrap();
        let mut changed = binding.clone();
        changed.processing_state = String::from("preview");
        assert!(binding.verify_unchanged(&changed).is_err());
        let mut changed = binding.clone();
        changed.sample_rate_hz = 44_100.0;
        assert!(binding.verify_unchanged(&changed).is_err());
    }
}
