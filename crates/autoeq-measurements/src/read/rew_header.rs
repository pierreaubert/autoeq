//! Declared facts from REW text-export headers.
//!
//! REW writes measurement metadata as `*` comment lines above the data
//! table (microphone, stimulus, smoothing, timing notes). CSV exports of
//! `.mdat` captures carry the same keys as `#` comment lines (see
//! `utils/mdat2csv.py`), which the curve loaders already skip. This
//! module transcribes either comment form into a facts struct so evidence
//! intake can cite declarations with provenance. Every field is a
//! declaration, never an independently verified claim: an
//! `acoustic_timing_reference` declaration does not by itself verify a
//! shared time zero, and a target level never authenticates absolute
//! playback calibration.

use std::path::Path;

/// Declared facts transcribed from a REW export header.
///
/// All fields default to absent; parsing never fails. Headers vary across
/// REW versions, so unknown lines are ignored and partially present
/// headers yield partially present facts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RewHeaderFacts {
    /// REW version string, e.g. `"V5.40 beta 124"`.
    pub rew_version: Option<String>,
    /// Microphone token from the Source line, e.g. `"UMIK-2"`.
    pub microphone: Option<String>,
    /// Whether a Format line declares an acoustic timing reference.
    ///
    /// `false` means "no declaration seen", not "declared absent": files
    /// without headers and headers without a Format line both read `false`.
    pub acoustic_timing_reference: bool,
    /// Clock adjustment in ppm from the Note line, when stated.
    pub clock_adjustment_ppm: Option<f64>,
    /// Estimated IR delay in ms from the Note line, when stated.
    pub estimated_ir_delay_ms: Option<f64>,
    /// Full Note line content, preserving reference-channel and offset detail.
    pub timing_note: Option<String>,
    /// Smoothing declaration, e.g. `"Variable"` or `"1/12 octave"`.
    pub smoothing: Option<String>,
    /// Export frequency resolution in points per octave, when stated.
    pub frequency_step_ppo: Option<f64>,
    /// Stimulus description without the timing-reference suffix.
    pub stimulus: Option<String>,
    /// REW target level in dB. Not absolute playback calibration.
    pub target_level_db: Option<f64>,
    /// Measurement name from the header.
    pub measurement_name: Option<String>,
    /// Capture date string, transcribed verbatim.
    pub dated: Option<String>,
    /// Conversion provenance, e.g. `"2.2.mdat via mdat2csv.py"`.
    ///
    /// Present only on derived exports; original REW text exports leave
    /// this absent. Records where the curve bytes came from, never a
    /// claim about their acoustic validity.
    pub converted_from: Option<String>,
}

impl RewHeaderFacts {
    /// Whether no declaration of any kind was transcribed.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Parse REW header facts from export text.
///
/// Scans `*` (REW text export) and `#` (CSV-embedded preservation form)
/// comment lines and transcribes recognized declarations.
/// Unknown lines and unparsable values are skipped, so the result is
/// partial rather than wrong when headers vary by REW version.
///
/// # Examples
///
/// ```rust
/// use autoeq_measurements::read::{parse_rew_header, RewHeaderFacts};
///
/// let facts = parse_rew_header("* Format: 512k Log Swept Sine using an acoustic timing reference\n");
/// assert!(facts.acoustic_timing_reference);
/// assert_eq!(facts.microphone, None);
/// ```
pub fn parse_rew_header(text: &str) -> RewHeaderFacts {
    let mut facts = RewHeaderFacts::default();
    for raw in text.lines() {
        let Some(line) = strip_comment(raw) else {
            continue;
        };
        if let Some(rest) = after(line, "Measurement data measured by REW") {
            facts.rew_version = Some(rest.trim().to_string());
        } else if let Some(rest) = after(line, "Source:") {
            facts.microphone = first_parenthesized(rest);
        } else if let Some(rest) = after(line, "Format:") {
            facts.acoustic_timing_reference |= rest.contains("acoustic timing reference");
            facts.stimulus = Some(
                rest.split(" using an acoustic timing reference")
                    .next()
                    .unwrap_or(rest)
                    .trim()
                    .to_string(),
            );
        } else if let Some(rest) = after(line, "Dated:") {
            facts.dated = Some(rest.trim().to_string());
        } else if let Some(rest) = after(line, "Target level:") {
            facts.target_level_db = first_number(rest);
        } else if let Some(rest) = after(line, "Note:") {
            facts.timing_note = Some(rest.trim().to_string());
            if facts.clock_adjustment_ppm.is_none() {
                facts.clock_adjustment_ppm = scan_number_after(rest, "Clock adjustment:");
            }
            if facts.estimated_ir_delay_ms.is_none() {
                facts.estimated_ir_delay_ms = scan_number_after(rest, "Delay");
            }
        } else if let Some(rest) = after(line, "Measurement:") {
            facts.measurement_name = Some(rest.trim().to_string());
        } else if let Some(rest) = after(line, "Smoothing:") {
            facts.smoothing = Some(rest.trim().to_string());
        } else if let Some(rest) = after(line, "Frequency Step:") {
            facts.frequency_step_ppo = first_number(rest);
        } else if let Some(rest) = after(line, "Converted from:") {
            facts.converted_from = Some(rest.trim().to_string());
        }
    }
    facts
}

/// Read REW header facts from an export file.
///
/// Reads the file as text and parses its `*` comment lines. Fails only on
/// I/O errors; a file without headers yields default (absent) facts.
pub fn read_rew_header_facts(path: &Path) -> std::io::Result<RewHeaderFacts> {
    let text = std::fs::read_to_string(path)?;
    Ok(parse_rew_header(&text))
}

/// Strip one leading `*` or `#` comment marker, if present.
fn strip_comment(line: &str) -> Option<&str> {
    let trimmed = line.trim_start_matches('\u{feff}').trim_start();
    trimmed
        .strip_prefix('*')
        .or_else(|| trimmed.strip_prefix('#'))
        .map(str::trim)
}

/// Split off a recognized `Prefix:` marker.
fn after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    line.strip_prefix(prefix).map(str::trim)
}

/// Parse the first whitespace-delimited token as a number.
fn first_number(text: &str) -> Option<f64> {
    text.split_whitespace().next()?.parse().ok()
}

/// Parse the number following a marker such as `"Clock adjustment:"`.
fn scan_number_after(haystack: &str, marker: &str) -> Option<f64> {
    let rest = haystack.split_once(marker)?.1;
    let token: String = rest
        .chars()
        .skip_while(|c| c.is_whitespace() || *c == ':')
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '+' | '-' | 'e' | 'E'))
        .collect();
    if token.is_empty() {
        return None;
    }
    token.parse().ok()
}

/// Extract the first parenthesized token, e.g. the microphone model.
fn first_parenthesized(text: &str) -> Option<String> {
    let rest = text.split_once('(')?.1;
    let token = rest.split(')').next()?.trim();
    if token.is_empty() {
        return None;
    }
    Some(token.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASCILAB_HEADER: &str = "\
* Measurement data measured by REW V5.40 beta 124
* Source: EXCL: Line (UMIK-2), LINE_IN (Hauptlautstärke), L, volume: 1.000. Timing signal peak level -25.5 dBFS, measurement signal peak level -5.3 dBFS
* Format: 512k Log Swept Sine, 1 sweep at -25.0 dBFS using an acoustic timing reference
* Dated: Aug 12, 2026 10:08:36 AM
* REW Settings:
*  C-weighting compensation: Off
*  Target level: 75.0 dB
* Note: ; Delay -0.0158 ms (-5.4 mm, -0.21 in) using estimated IR delay relative to Acoustic reference played from LINE_OUT L with no timing offset Clock adjustment: -30.7 ppm
* Measurement: L C8C-BX8C-P1_No EQ Aug 12
* Smoothing: Variable
* Frequency Step: 96 ppo
* Start Frequency: 20.141602 Hz
*
* Freq(Hz), SPL(dB), Phase(degrees)
20.141602, 91.525, 163.3452
";

    #[test]
    fn parses_ascilab_header_facts() {
        let facts = parse_rew_header(ASCILAB_HEADER);
        assert_eq!(facts.rew_version.as_deref(), Some("V5.40 beta 124"));
        assert_eq!(facts.microphone.as_deref(), Some("UMIK-2"));
        assert!(facts.acoustic_timing_reference);
        assert!((facts.clock_adjustment_ppm.unwrap() - -30.7).abs() < 1e-9);
        assert!((facts.estimated_ir_delay_ms.unwrap() - -0.0158).abs() < 1e-9);
        assert!(
            facts
                .timing_note
                .as_deref()
                .is_some_and(|note| note.contains("Acoustic reference"))
        );
        assert_eq!(facts.smoothing.as_deref(), Some("Variable"));
        assert!((facts.frequency_step_ppo.unwrap() - 96.0).abs() < 1e-9);
        assert_eq!(
            facts.stimulus.as_deref(),
            Some("512k Log Swept Sine, 1 sweep at -25.0 dBFS")
        );
        assert!((facts.target_level_db.unwrap() - 75.0).abs() < 1e-9);
        assert_eq!(
            facts.measurement_name.as_deref(),
            Some("L C8C-BX8C-P1_No EQ Aug 12")
        );
        assert_eq!(facts.dated.as_deref(), Some("Aug 12, 2026 10:08:36 AM"));
    }

    #[test]
    fn plain_csv_yields_absent_facts() {
        let facts = parse_rew_header("freq_hz,spl_db,phase_deg\n20.0,70.0,0.0\n");
        assert_eq!(facts, RewHeaderFacts::default());
        assert!(facts.is_empty());
    }

    #[test]
    fn csv_embedded_hash_headers_parse_like_star_headers() {
        let csv = "\
# Measurement: L C8C-BX8C-P1_No EQ Aug 12
# Dated: Aug 12, 2026 10:08:36 AM
# Format: 512k Log Swept Sine using an acoustic timing reference
# Frequency Step: 96 ppo
# Converted from: 2.2.mdat via mdat2csv.py
freq_hz,spl_db,phase_deg
20.141602,91.525,163.3452
";
        let facts = parse_rew_header(csv);
        assert!(!facts.is_empty());
        assert_eq!(
            facts.measurement_name.as_deref(),
            Some("L C8C-BX8C-P1_No EQ Aug 12")
        );
        assert!(facts.acoustic_timing_reference);
        assert!((facts.frequency_step_ppo.unwrap() - 96.0).abs() < 1e-9);
        assert_eq!(
            facts.converted_from.as_deref(),
            Some("2.2.mdat via mdat2csv.py")
        );
        // Unrelated `#` comments stay inert.
        assert_eq!(facts.microphone, None);
    }

    #[test]
    fn partial_header_keeps_declared_facts_only() {
        let facts = parse_rew_header(
            "* Source: Line (UMIK-1)\n* Format: 256k Log Swept Sine\n20.0, 70.0\n",
        );
        assert_eq!(facts.microphone.as_deref(), Some("UMIK-1"));
        assert!(!facts.acoustic_timing_reference);
        assert_eq!(facts.stimulus.as_deref(), Some("256k Log Swept Sine"));
        assert_eq!(facts.smoothing, None);
        assert_eq!(facts.clock_adjustment_ppm, None);
    }

    #[test]
    fn reads_facts_from_export_file() {
        let dir = std::env::temp_dir().join(format!(
            "autoeq_rew_header_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sweep.txt");
        std::fs::write(&path, ASCILAB_HEADER).unwrap();
        let facts = read_rew_header_facts(&path).unwrap();
        assert_eq!(facts.microphone.as_deref(), Some("UMIK-2"));
        assert!(facts.acoustic_timing_reference);
        std::fs::remove_dir_all(&dir).ok();
    }
}
