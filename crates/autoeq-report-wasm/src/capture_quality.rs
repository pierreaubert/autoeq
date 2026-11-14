//! Renderer-neutral input and assembly for capture-quality report sections.
//!
//! This module accepts measurements already computed by an upstream capture
//! workflow. It formats evidence for the existing report schema and performs
//! no acoustic analysis. Unknown evidence remains unknown and is never
//! converted into a numeric pass value.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::report_table::ReportTable;
use crate::schema::{ReportPayload, ReportProvenance, Section};

/// Current version of the capture-quality input contract.
pub const CAPTURE_QUALITY_INPUT_VERSION: u32 = 1;

const CAPTURE_QUALITY_BUCKET: &str = "DSP analysis";
const CAPTURE_QUALITY_SUBHEAD: &str = "Capture quality";

/// A capture-quality report containing source identities and measured findings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureQualityReport {
    /// Input contract version; missing values in legacy `{}` inputs mean version one.
    #[serde(default = "default_capture_quality_input_version")]
    pub schema_version: u32,
    /// Captured sources represented in the report.
    #[serde(default)]
    pub sources: Vec<CaptureQualitySource>,
}

impl Default for CaptureQualityReport {
    fn default() -> Self {
        Self {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: Vec::new(),
        }
    }
}

fn default_capture_quality_input_version() -> u32 {
    CAPTURE_QUALITY_INPUT_VERSION
}

/// One source or take and its measured quality findings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureQualitySource {
    /// Stable identifier for the source within this report.
    pub source_id: String,
    /// Optional logical output or input channel name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    /// Optional microphone identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mic: Option<String>,
    /// Optional measurement position identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    /// Optional take number, starting at one when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub take: Option<u32>,
    /// Stimulus configuration recorded for this source, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stimulus: Option<CaptureStimulusMetadata>,
    /// Findings measured for this source.
    #[serde(default)]
    pub findings: Vec<CaptureQualityFinding>,
}

/// Stimulus settings associated with an acquired source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureStimulusMetadata {
    /// Signal name, such as `"octave_sweep"` or `"pink_noise"`.
    pub signal_type: String,
    /// Sample rate used for acquisition, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Generated stimulus duration in seconds, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    /// Stimulus frequency range, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<CaptureQualityBand>,
    /// Peak stimulus level in dBFS, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level_dbfs: Option<f64>,
}

/// A typed capture-quality metric name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "label", rename_all = "snake_case")]
pub enum CaptureQualityMetric {
    /// Total harmonic distortion.
    Thd,
    /// Broadband signal-to-noise ratio.
    BroadbandSnr,
    /// Coherent signal-to-noise ratio.
    CoherentSnr,
    /// Measured clipping fraction.
    ClipFraction,
    /// Coherence between reference and captured signals.
    Coherence,
    /// Repeatability across repeated takes.
    Repeatability,
    /// A metric introduced by an upstream capture workflow.
    Other(String),
}

impl CaptureQualityMetric {
    fn label(&self) -> &str {
        match self {
            Self::Thd => "THD",
            Self::BroadbandSnr => "Broadband SNR",
            Self::CoherentSnr => "Coherent SNR",
            Self::ClipFraction => "Clipping fraction",
            Self::Coherence => "Coherence",
            Self::Repeatability => "Repeatability",
            Self::Other(label) => label,
        }
    }
}

/// Evidence state assigned by the upstream quality policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureQualityStatus {
    /// The measured value meets its stated criterion.
    Pass,
    /// The measured value warrants review.
    Warning,
    /// The measured value fails its stated criterion.
    Fail,
    /// The evidence needed for a decision was unavailable.
    Unknown,
}

impl CaptureQualityStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "Pass",
            Self::Warning => "Warning",
            Self::Fail => "Fail",
            Self::Unknown => "Unknown",
        }
    }
}

/// A frequency range associated with a metric or stimulus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CaptureQualityBand {
    /// Inclusive lower band edge in hertz.
    pub low_hz: f64,
    /// Inclusive upper band edge in hertz.
    pub high_hz: f64,
}

/// Comparison direction for a metric threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureThresholdComparison {
    /// The measured value is expected to be at or below the threshold.
    LessThanOrEqual,
    /// The measured value is expected to be at or above the threshold.
    GreaterThanOrEqual,
    /// The measured value is expected to equal the threshold.
    EqualTo,
}

impl CaptureThresholdComparison {
    fn symbol(self) -> &'static str {
        match self {
            Self::LessThanOrEqual => "≤",
            Self::GreaterThanOrEqual => "≥",
            Self::EqualTo => "=",
        }
    }
}

/// A numeric decision boundary for a finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureQualityThreshold {
    /// Inclusive or exclusive comparison direction.
    pub comparison: CaptureThresholdComparison,
    /// Boundary value in `unit`.
    pub value: f64,
    /// Unit for the boundary value.
    pub unit: String,
}

/// One measured metric, its decision, and suggested next action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureQualityFinding {
    /// Metric measured by the upstream analysis.
    pub metric: CaptureQualityMetric,
    /// Status assigned by the upstream quality policy.
    pub status: CaptureQualityStatus,
    /// Finite measured value; absent only when status is `Unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Unit for the measured value, such as `%`, `dB`, or `ratio`.
    pub unit: String,
    /// Frequency band used by the measurement, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<CaptureQualityBand>,
    /// Name of the measurement method, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Name of the quality policy that interpreted this evidence, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    /// Policy threshold, when the metric has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<CaptureQualityThreshold>,
    /// Traceable reason that an unknown metric could not be evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unknown_reason: Option<String>,
    /// Suggested action for the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<String>,
}

/// Invalid input prevents a quality section from being appended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureQualityReportError {
    message: String,
}

impl CaptureQualityReportError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CaptureQualityReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CaptureQualityReportError {}

/// Validates and appends capture findings as an existing-schema HTML section.
///
/// Source metadata and findings remain inputs only; this function does not
/// estimate metrics, infer missing values, or change report schema fields.
/// An empty report leaves the payload untouched.
///
/// # Examples
/// ```
/// use autoeq_report_wasm::{
///     CaptureQualityFinding, CaptureQualityMetric, CaptureQualityReport,
///     CaptureQualitySource, CaptureQualityStatus, ReportPayload,
///     append_capture_quality_section,
/// };
///
/// let finding = CaptureQualityFinding {
///     metric: CaptureQualityMetric::Thd,
///     status: CaptureQualityStatus::Warning,
///     value: Some(12.0),
///     unit: "%".to_string(),
///     band: None,
///     method: Some("harmonic analysis".to_string()),
///     policy: Some("single_take".to_string()),
///     threshold: None,
///     unknown_reason: None,
///     recommendation: Some("Check the capture level.".to_string()),
/// };
/// let report = CaptureQualityReport {
///     schema_version: 1,
///     sources: vec![CaptureQualitySource {
///         source_id: "left-main".to_string(),
///         channel: Some("Left".to_string()),
///         mic: None,
///         position: None,
///         take: Some(1),
///         stimulus: None,
///         findings: vec![finding],
///     }],
/// };
/// let mut payload = ReportPayload::new("Room report");
/// append_capture_quality_section(&mut payload, &report)?;
/// assert_eq!(payload.sections.len(), 1);
/// # Ok::<(), autoeq_report_wasm::CaptureQualityReportError>(())
/// ```
///
/// # Errors
/// Returns an error for unsupported schemas, missing identities, inconsistent
/// unknown evidence, empty units, invalid bands, or non-finite values.
pub fn append_capture_quality_section(
    payload: &mut ReportPayload,
    report: &CaptureQualityReport,
) -> Result<(), CaptureQualityReportError> {
    if payload.schema != crate::schema::SCHEMA_VERSION {
        return Err(CaptureQualityReportError::new(format!(
            "unsupported report payload schema {:?}; expected {:?}",
            payload.schema,
            crate::schema::SCHEMA_VERSION
        )));
    }
    if let Some(section) = capture_quality_section(report)? {
        payload.sections.push(Section::Html {
            html: section.html(),
            tab: None,
        });
    }
    Ok(())
}

/// Capture findings and the source identifiers for browser actions.
#[derive(Debug, Clone)]
pub struct CaptureQualitySection {
    /// Findings in source order.
    pub table: ReportTable,
    action_sources: Vec<Option<String>>,
}

impl CaptureQualitySection {
    pub(crate) fn browser_section(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "html",
            "html": self.html(),
            "tab": null,
            "bucket": CAPTURE_QUALITY_BUCKET,
            "subhead": CAPTURE_QUALITY_SUBHEAD,
        })
    }

    fn html(&self) -> String {
        let mut html = String::from("<section class=\"capture-quality\"><h2>");
        escape_html_into(&mut html, &self.table.title);
        html.push_str("</h2><table><thead><tr>");
        for column in &self.table.columns {
            html.push_str("<th>");
            escape_html_into(&mut html, column);
            html.push_str("</th>");
        }
        html.push_str("<th>Actions</th></tr></thead><tbody>");
        for (row, source) in self.table.rows.iter().zip(&self.action_sources) {
            html.push_str("<tr>");
            for cell in row {
                push_cell(&mut html, cell);
            }
            html.push_str("<td>");
            if let Some(source) = source {
                html.push_str(
                    "<div class=\"capture-quality-actions\" hidden data-capture-source-id=\"",
                );
                escape_html_into(&mut html, source);
                html.push_str("\"><button type=\"button\" data-report-capture-action=\"select\" disabled>Select</button><button type=\"button\" data-report-capture-action=\"retake\" disabled>Review retake</button></div>");
            }
            html.push_str("</td></tr>");
        }
        html.push_str("</tbody></table></section>");
        html
    }
}

pub(crate) fn capture_quality_section(
    report: &CaptureQualityReport,
) -> Result<Option<CaptureQualitySection>, CaptureQualityReportError> {
    if report.schema_version != CAPTURE_QUALITY_INPUT_VERSION {
        return Err(CaptureQualityReportError::new(format!(
            "unsupported capture-quality input schema {}; expected {}",
            report.schema_version, CAPTURE_QUALITY_INPUT_VERSION
        )));
    }
    validate_report(report)?;
    if report.sources.is_empty() {
        return Ok(None);
    }

    let mut rows = Vec::new();
    let mut action_sources = Vec::new();
    for source in &report.sources {
        if source.findings.is_empty() {
            rows.push(source_row(source, None));
            action_sources.push(Some(source.source_id.clone()));
        } else {
            for (index, finding) in source.findings.iter().enumerate() {
                rows.push(source_row(source, Some(finding)));
                action_sources.push((index == 0).then(|| source.source_id.clone()));
            }
        }
    }
    Ok(Some(CaptureQualitySection {
        table: ReportTable {
            title: CAPTURE_QUALITY_SUBHEAD.into(),
            note: String::new(),
            columns: [
                "Source",
                "Channel",
                "Mic",
                "Position",
                "Take",
                "Stimulus",
                "Policy",
                "Metric",
                "Status",
                "Value",
                "Band",
                "Method",
                "Threshold",
                "Evidence and recommendation",
            ]
            .map(String::from)
            .into(),
            rows,
        },
        action_sources,
    }))
}

/// Appends capture findings to a versioned JSON payload without dropping extensions.
///
/// Existing top-level and section fields are preserved byte-for-value. The new
/// section uses the same DSP bucket and Capture quality subhead as the host
/// report integration. An empty source list leaves a valid payload unchanged.
///
/// # Examples
/// ```
/// use autoeq_report_wasm::{
///     CaptureQualityReport, append_capture_quality_section_to_value,
/// };
///
/// let mut payload = serde_json::json!({
///     "schema": "autoeq-report-data-v1",
///     "title": "Room report",
///     "sections": []
/// });
/// append_capture_quality_section_to_value(
///     &mut payload,
///     &CaptureQualityReport::default(),
/// )?;
/// assert!(payload["sections"].as_array().unwrap().is_empty());
/// # Ok::<(), autoeq_report_wasm::CaptureQualityReportError>(())
/// ```
///
/// # Errors
/// Returns an error for an unsupported payload or input version, malformed
/// payload structure, or invalid capture-quality evidence. The payload remains
/// unchanged when this function returns an error.
pub fn append_capture_quality_section_to_value(
    payload: &mut serde_json::Value,
    report: &CaptureQualityReport,
) -> Result<(), CaptureQualityReportError> {
    validate_json_payload(payload)?;
    let Some(section) = capture_quality_section(report)? else {
        return Ok(());
    };
    let sections = payload
        .get_mut("sections")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| CaptureQualityReportError::new("report payload sections are missing"))?;
    sections.push(section.browser_section());
    Ok(())
}

fn validate_json_payload(payload: &serde_json::Value) -> Result<(), CaptureQualityReportError> {
    let object = payload
        .as_object()
        .ok_or_else(|| CaptureQualityReportError::new("report payload must be a JSON object"))?;
    let schema = object
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| CaptureQualityReportError::new("report payload schema is missing"))?;
    if schema != crate::schema::SCHEMA_VERSION {
        return Err(CaptureQualityReportError::new(format!(
            "unsupported report payload schema {schema:?}; expected {:?}",
            crate::schema::SCHEMA_VERSION
        )));
    }
    if object
        .get("title")
        .and_then(serde_json::Value::as_str)
        .is_none()
    {
        return Err(CaptureQualityReportError::new(
            "report payload title is missing",
        ));
    }
    let sections = object
        .get("sections")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CaptureQualityReportError::new("report payload sections are missing"))?;
    for (index, section) in sections.iter().enumerate() {
        serde_json::from_value::<Section>(section.clone()).map_err(|error| {
            CaptureQualityReportError::new(format!(
                "report payload section {index} is invalid: {error}"
            ))
        })?;
    }
    if let Some(provenance) = object.get("provenance")
        && !provenance.is_null()
    {
        serde_json::from_value::<ReportProvenance>(provenance.clone()).map_err(|error| {
            CaptureQualityReportError::new(format!("report payload provenance is invalid: {error}"))
        })?;
    }
    Ok(())
}

fn validate_report(report: &CaptureQualityReport) -> Result<(), CaptureQualityReportError> {
    for (source_index, source) in report.sources.iter().enumerate() {
        if source.source_id.trim().is_empty() {
            return Err(CaptureQualityReportError::new(format!(
                "source {source_index} has an empty source_id"
            )));
        }
        if let Some(stimulus) = &source.stimulus {
            validate_stimulus(source_index, stimulus)?;
        }
        for (finding_index, finding) in source.findings.iter().enumerate() {
            validate_finding(source_index, finding_index, finding)?;
        }
    }
    Ok(())
}

fn validate_stimulus(
    source_index: usize,
    stimulus: &CaptureStimulusMetadata,
) -> Result<(), CaptureQualityReportError> {
    if stimulus.signal_type.trim().is_empty() {
        return Err(CaptureQualityReportError::new(format!(
            "source {source_index} has an empty stimulus signal_type"
        )));
    }
    if stimulus.sample_rate_hz == Some(0) {
        return Err(CaptureQualityReportError::new(format!(
            "source {source_index} has a zero stimulus sample rate"
        )));
    }
    if stimulus
        .duration_secs
        .is_some_and(|duration| !duration.is_finite() || duration <= 0.0)
    {
        return Err(CaptureQualityReportError::new(format!(
            "source {source_index} has an invalid stimulus duration"
        )));
    }
    if stimulus.level_dbfs.is_some_and(|level| !level.is_finite()) {
        return Err(CaptureQualityReportError::new(format!(
            "source {source_index} has a non-finite stimulus level"
        )));
    }
    if let Some(band) = stimulus.band {
        validate_band(source_index, None, band)?;
    }
    Ok(())
}

fn validate_finding(
    source_index: usize,
    finding_index: usize,
    finding: &CaptureQualityFinding,
) -> Result<(), CaptureQualityReportError> {
    let where_ = format!("source {source_index} finding {finding_index}");
    if finding.unit.trim().is_empty() {
        return Err(CaptureQualityReportError::new(format!(
            "{where_} has an empty unit"
        )));
    }
    if let CaptureQualityMetric::Other(label) = &finding.metric
        && label.trim().is_empty()
    {
        return Err(CaptureQualityReportError::new(format!(
            "{where_} has an empty metric label"
        )));
    }
    if finding
        .policy
        .as_deref()
        .is_some_and(|policy| policy.trim().is_empty())
    {
        return Err(CaptureQualityReportError::new(format!(
            "{where_} has an empty policy label"
        )));
    }
    match finding.status {
        CaptureQualityStatus::Unknown => {
            if finding.value.is_some() {
                return Err(CaptureQualityReportError::new(format!(
                    "{where_} is Unknown but has a measured value"
                )));
            }
            if !finding
                .unknown_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            {
                return Err(CaptureQualityReportError::new(format!(
                    "{where_} is Unknown without an unknown_reason"
                )));
            }
        }
        _ => {
            if finding.value.is_none() {
                return Err(CaptureQualityReportError::new(format!(
                    "{where_} is not Unknown but has no measured value"
                )));
            }
            if finding.unknown_reason.is_some() {
                return Err(CaptureQualityReportError::new(format!(
                    "{where_} has an unknown_reason for a known status"
                )));
            }
        }
    }
    if finding.value.is_some_and(|value| !value.is_finite()) {
        return Err(CaptureQualityReportError::new(format!(
            "{where_} has a non-finite value"
        )));
    }
    if let Some(band) = finding.band {
        validate_band(source_index, Some(finding_index), band)?;
    }
    if let Some(threshold) = &finding.threshold {
        if !threshold.value.is_finite() {
            return Err(CaptureQualityReportError::new(format!(
                "{where_} has a non-finite threshold"
            )));
        }
        if threshold.unit.trim().is_empty() || threshold.unit != finding.unit {
            return Err(CaptureQualityReportError::new(format!(
                "{where_} threshold unit must match the measured unit"
            )));
        }
    }
    Ok(())
}

fn validate_band(
    source_index: usize,
    finding_index: Option<usize>,
    band: CaptureQualityBand,
) -> Result<(), CaptureQualityReportError> {
    let where_ = finding_index.map_or_else(
        || format!("source {source_index} stimulus band"),
        |index| format!("source {source_index} finding {index} band"),
    );
    if !band.low_hz.is_finite()
        || !band.high_hz.is_finite()
        || band.low_hz <= 0.0
        || band.high_hz <= band.low_hz
    {
        return Err(CaptureQualityReportError::new(format!(
            "{where_} must have finite, positive, increasing frequency bounds"
        )));
    }
    Ok(())
}

fn source_row(
    source: &CaptureQualitySource,
    finding: Option<&CaptureQualityFinding>,
) -> Vec<String> {
    let mut row = vec![
        source.source_id.clone(),
        display_cell(source.channel.as_deref()),
        display_cell(source.mic.as_deref()),
        display_cell(source.position.as_deref()),
        display_cell(source.take.map(|take| take.to_string()).as_deref()),
        display_cell(source.stimulus.as_ref().map(format_stimulus).as_deref()),
    ];
    if let Some(finding) = finding {
        row.extend([
            display_cell(finding.policy.as_deref()),
            finding.metric.label().into(),
            finding.status.label().into(),
            display_cell(
                finding
                    .value
                    .map(|value| format!("{value} {}", finding.unit))
                    .as_deref(),
            ),
            display_cell(finding.band.map(format_band).as_deref()),
            display_cell(finding.method.as_deref()),
            display_cell(finding.threshold.as_ref().map(format_threshold).as_deref()),
            format_evidence(finding),
        ]);
    } else {
        row.extend(["—", "—", "No findings", "—", "—", "—", "—", "—"].map(String::from));
    }
    row
}

fn display_cell(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or("—")
        .into()
}

fn push_cell(html: &mut String, value: &str) {
    html.push_str("<td>");
    escape_html_into(html, value);
    html.push_str("</td>");
}

fn format_stimulus(stimulus: &CaptureStimulusMetadata) -> String {
    let mut parts = vec![stimulus.signal_type.clone()];
    if let Some(sample_rate_hz) = stimulus.sample_rate_hz {
        parts.push(format!("{sample_rate_hz} Hz"));
    }
    if let Some(duration_secs) = stimulus.duration_secs {
        parts.push(format!("{duration_secs:.3} s"));
    }
    if let Some(band) = stimulus.band {
        parts.push(format_band(band));
    }
    if let Some(level_dbfs) = stimulus.level_dbfs {
        parts.push(format!("{level_dbfs:.2} dBFS"));
    }
    parts.join(" · ")
}

fn format_band(band: CaptureQualityBand) -> String {
    format!("{}–{} Hz", band.low_hz, band.high_hz)
}

fn format_threshold(threshold: &CaptureQualityThreshold) -> String {
    format!(
        "{} {} {}",
        threshold.comparison.symbol(),
        threshold.value,
        threshold.unit
    )
}

fn format_evidence(finding: &CaptureQualityFinding) -> String {
    let mut parts = Vec::new();
    if let Some(reason) = &finding.unknown_reason {
        parts.push(format!("Unknown reason: {reason}"));
    }
    if let Some(recommendation) = &finding.recommendation {
        parts.push(format!("Recommendation: {recommendation}"));
    }
    if parts.is_empty() {
        "—".to_string()
    } else {
        parts.join("; ")
    }
}

fn escape_html_into(html: &mut String, text: &str) {
    for character in text.chars() {
        match character {
            '&' => html.push_str("&amp;"),
            '<' => html.push_str("&lt;"),
            '>' => html.push_str("&gt;"),
            '"' => html.push_str("&quot;"),
            '\'' => html.push_str("&#39;"),
            _ => html.push(character),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_with_findings(findings: Vec<CaptureQualityFinding>) -> CaptureQualitySource {
        CaptureQualitySource {
            source_id: "speaker-left/mic-2/position-1".to_string(),
            channel: Some("Left <main>".to_string()),
            mic: Some("UMIK-1".to_string()),
            position: Some("Seat A".to_string()),
            take: Some(2),
            stimulus: Some(CaptureStimulusMetadata {
                signal_type: "octave_sweep".to_string(),
                sample_rate_hz: Some(48_000),
                duration_secs: Some(8.25),
                band: Some(CaptureQualityBand {
                    low_hz: 20.0,
                    high_hz: 20_000.0,
                }),
                level_dbfs: Some(-6.0),
            }),
            findings,
        }
    }

    fn warning_finding() -> CaptureQualityFinding {
        CaptureQualityFinding {
            metric: CaptureQualityMetric::Thd,
            status: CaptureQualityStatus::Warning,
            value: Some(12.5),
            unit: "%".to_string(),
            band: Some(CaptureQualityBand {
                low_hz: 100.0,
                high_hz: 1_000.0,
            }),
            method: Some("Farina harmonic impulse response".to_string()),
            policy: Some("single_take".to_string()),
            threshold: Some(CaptureQualityThreshold {
                comparison: CaptureThresholdComparison::LessThanOrEqual,
                value: 10.0,
                unit: "%".to_string(),
            }),
            unknown_reason: None,
            recommendation: Some("Reduce playback level and retake.".to_string()),
        }
    }

    fn unknown_finding() -> CaptureQualityFinding {
        CaptureQualityFinding {
            metric: CaptureQualityMetric::BroadbandSnr,
            status: CaptureQualityStatus::Unknown,
            value: None,
            unit: "dB".to_string(),
            band: Some(CaptureQualityBand {
                low_hz: 20.0,
                high_hz: 20_000.0,
            }),
            method: Some("quiet/active window ratio".to_string()),
            policy: Some("single_take".to_string()),
            threshold: Some(CaptureQualityThreshold {
                comparison: CaptureThresholdComparison::GreaterThanOrEqual,
                value: 10.0,
                unit: "dB".to_string(),
            }),
            unknown_reason: Some("No measurable quiet-window noise floor.".to_string()),
            recommendation: Some("Check microphone routing and ambient capture.".to_string()),
        }
    }

    #[test]
    fn dashboard_renders_warning_unknown_identity_and_stimulus() {
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![
                warning_finding(),
                unknown_finding(),
            ])],
        };
        let mut payload = ReportPayload::new("Room report");

        append_capture_quality_section(&mut payload, &report).expect("append quality section");

        let json = serde_json::to_string(&payload).expect("serialize payload");
        assert!(json.contains("capture-quality"));
        assert!(json.contains("speaker-left/mic-2/position-1"));
        assert!(json.contains("Farina harmonic impulse response"));
        assert!(json.contains("single_take"));
        assert!(json.contains("12.5 %"));
        assert!(json.contains("Unknown reason: No measurable quiet-window noise floor."));
        assert!(json.contains("Recommendation: Check microphone routing and ambient capture."));
        assert!(json.contains("Left &lt;main&gt;"));
        assert!(json.contains("octave_sweep · 48000 Hz · 8.250 s"));
        assert!(!json.contains("Left <main>"));
    }

    #[test]
    fn source_controls_are_hidden_until_host_ready_and_escape_untrusted_ids() {
        let mut source = source_with_findings(vec![warning_finding(), unknown_finding()]);
        source.source_id = "source\" data-x=\"y\"><script>alert('x')</script>".to_string();
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source],
        };
        let mut payload = ReportPayload::new("Room report");

        append_capture_quality_section(&mut payload, &report).expect("append quality section");

        let html = match &payload.sections[0] {
            Section::Html { html, .. } => html,
            _ => panic!("quality section is HTML"),
        };
        assert!(
            html.contains("class=\"capture-quality-actions\" hidden data-capture-source-id=\"")
        );
        assert!(html.contains(
            "data-capture-source-id=\"source&quot; data-x=&quot;y&quot;&gt;&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"
        ));
        assert!(!html.contains("data-capture-source-id=\"source\" data-x="));
        assert!(!html.contains("<script>alert('x')</script>"));
        assert!(html.contains("data-report-capture-action=\"select\" disabled>Select</button>"));
        assert!(
            html.contains("data-report-capture-action=\"retake\" disabled>Review retake</button>")
        );
        assert_eq!(html.matches("data-report-capture-action=").count(), 2);
        assert!(
            html.contains("Unknown"),
            "unknown quality evidence remains visible"
        );
    }

    #[test]
    fn shell_actions_use_the_versioned_sotf_host_contract() {
        let shell = include_str!("../shell/template.html");
        for contract in [
            "window.sotfReportHost",
            "host.version !== 1",
            "host.reportId",
            "host.send({action: \"ready\", renderer: \"autoeq-report-wasm\"})",
            "host.send({action: \"error\", message})",
            "showError(\"Could not start the 2D report renderer: \" + error)",
            "sotf-report-host-ready",
            "const labels = captureActionLabels[reportLocale(host?.locale)];",
            "if (labels[action]) button.textContent = labels[action];",
            "selection: {source_id: sourceId, microphone_id: null, take_id: null}",
            "action !== \"select\" && action !== \"retake\"",
            "html[data-report-theme=\"dark\"]",
            "Sélectionner",
            "Auswählen",
            "Seleccionar",
        ] {
            assert!(
                shell.contains(contract),
                "missing shell bridge contract: {contract}"
            );
        }
    }

    #[test]
    fn empty_quality_report_preserves_legacy_payload_shape() {
        let mut payload = ReportPayload::new("Legacy report");
        let original = serde_json::to_value(&payload).expect("serialize initial payload");

        append_capture_quality_section(&mut payload, &CaptureQualityReport::default())
            .expect("empty report is valid");

        assert_eq!(
            serde_json::to_value(&payload).expect("serialize payload"),
            original
        );
        let decoded: ReportPayload = serde_json::from_value(serde_json::json!({
            "schema": crate::schema::SCHEMA_VERSION,
            "title": "Old payload",
            "sections": []
        }))
        .expect("old payload without provenance remains supported");
        assert!(decoded.provenance.is_none());

        let decoded_input: CaptureQualityReport =
            serde_json::from_str("{}").expect("legacy empty input uses version one");
        assert_eq!(decoded_input.schema_version, CAPTURE_QUALITY_INPUT_VERSION);
        assert!(decoded_input.sources.is_empty());
    }

    #[test]
    fn value_api_preserves_existing_extensions_and_adds_shared_grouping() {
        let mut payload = serde_json::json!({
            "schema": crate::schema::SCHEMA_VERSION,
            "title": "Room report",
            "sections": [{
                "kind": "html",
                "html": "<p>summary</p>",
                "tab": null,
                "bucket": "Acoustics analysis",
                "subhead": "Summary",
                "flat": true,
                "future_section_field": {"keep": 7}
            }],
            "future_payload_field": {"keep": true}
        });
        let original_section = payload["sections"][0].clone();
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![
                warning_finding(),
                unknown_finding(),
            ])],
        };

        append_capture_quality_section_to_value(&mut payload, &report)
            .expect("append quality section to JSON payload");

        let sections = payload["sections"].as_array().expect("sections array");
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0], original_section);
        assert_eq!(payload["future_payload_field"]["keep"], true);
        let quality_section = &sections[1];
        assert_eq!(quality_section["kind"], "html");
        assert_eq!(quality_section["bucket"], CAPTURE_QUALITY_BUCKET);
        assert_eq!(quality_section["subhead"], CAPTURE_QUALITY_SUBHEAD);
        let html = quality_section["html"].as_str().expect("quality HTML");
        assert!(html.contains("Unknown"));
        assert!(html.contains("No measurable quiet-window noise floor."));
    }

    #[test]
    fn value_api_empty_sources_are_a_no_op_after_schema_validation() {
        let mut payload = serde_json::json!({
            "schema": crate::schema::SCHEMA_VERSION,
            "title": "Room report",
            "sections": [],
            "extension": [1, 2, 3]
        });
        let original = payload.clone();

        append_capture_quality_section_to_value(&mut payload, &CaptureQualityReport::default())
            .expect("empty valid input is a no-op");

        assert_eq!(payload, original);

        payload["schema"] = serde_json::Value::String("unsupported".to_string());
        let unsupported = payload.clone();
        assert!(
            append_capture_quality_section_to_value(&mut payload, &CaptureQualityReport::default())
                .is_err()
        );
        assert_eq!(payload, unsupported);
    }

    #[test]
    fn value_api_rejects_invalid_inputs_without_mutating_payload() {
        let mut payload = serde_json::json!({
            "schema": crate::schema::SCHEMA_VERSION,
            "title": "Room report",
            "sections": []
        });
        let original = payload.clone();
        let mut invalid_finding = warning_finding();
        invalid_finding.value = Some(f64::INFINITY);
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![invalid_finding])],
        };

        assert!(append_capture_quality_section_to_value(&mut payload, &report).is_err());
        assert_eq!(payload, original);

        let unsupported_report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION + 1,
            sources: vec![source_with_findings(vec![warning_finding()])],
        };
        assert!(
            append_capture_quality_section_to_value(&mut payload, &unsupported_report).is_err()
        );
        assert_eq!(payload, original);

        let valid_report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![warning_finding()])],
        };
        let mut unsupported_payload = payload.clone();
        unsupported_payload["schema"] =
            serde_json::Value::String("unsupported-report-schema".to_string());
        let before = unsupported_payload.clone();
        assert!(
            append_capture_quality_section_to_value(&mut unsupported_payload, &valid_report)
                .is_err()
        );
        assert_eq!(unsupported_payload, before);
    }

    #[test]
    fn rejects_wrong_payload_schema_without_mutating_existing_sections() {
        let mut payload = ReportPayload::new("Existing report");
        payload.sections.push(Section::Html {
            html: "<p>existing</p>".to_string(),
            tab: None,
        });
        payload.schema = "unsupported-report-schema".to_string();
        let before = serde_json::to_value(&payload).expect("serialize original payload");
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![warning_finding()])],
        };

        let error = append_capture_quality_section(&mut payload, &report)
            .expect_err("unsupported payload schemas must be rejected");

        assert!(
            error
                .to_string()
                .contains("unsupported report payload schema")
        );
        assert_eq!(
            serde_json::to_value(&payload).expect("serialize unchanged payload"),
            before
        );
    }

    #[test]
    fn rejects_unsupported_input_schema_without_mutation() {
        let mut payload = ReportPayload::new("Existing report");
        payload.sections.push(Section::Html {
            html: "<p>existing</p>".to_string(),
            tab: None,
        });
        let before = serde_json::to_value(&payload).expect("serialize original payload");
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION + 1,
            sources: vec![source_with_findings(vec![warning_finding()])],
        };

        let error = append_capture_quality_section(&mut payload, &report)
            .expect_err("unsupported input versions must be rejected");

        assert!(
            error
                .to_string()
                .contains("unsupported capture-quality input schema")
        );
        assert_eq!(
            serde_json::to_value(&payload).expect("serialize unchanged payload"),
            before
        );
    }

    #[test]
    fn rejects_unknown_metric_with_value_and_missing_reason() {
        let mut with_value = unknown_finding();
        with_value.value = Some(0.0);
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![with_value])],
        };
        let mut payload = ReportPayload::new("Report");
        assert!(append_capture_quality_section(&mut payload, &report).is_err());
        assert!(payload.sections.is_empty());

        let mut missing_reason = unknown_finding();
        missing_reason.unknown_reason = None;
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![missing_reason])],
        };
        assert!(append_capture_quality_section(&mut payload, &report).is_err());
        assert!(payload.sections.is_empty());
    }

    #[test]
    fn rejects_non_finite_values_and_invalid_bands() {
        let mut non_finite = warning_finding();
        non_finite.value = Some(f64::NAN);
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![non_finite])],
        };
        let mut payload = ReportPayload::new("Report");
        assert!(append_capture_quality_section(&mut payload, &report).is_err());

        let mut invalid_band = warning_finding();
        invalid_band.band = Some(CaptureQualityBand {
            low_hz: 500.0,
            high_hz: 100.0,
        });
        let report = CaptureQualityReport {
            schema_version: CAPTURE_QUALITY_INPUT_VERSION,
            sources: vec![source_with_findings(vec![invalid_band])],
        };
        assert!(append_capture_quality_section(&mut payload, &report).is_err());
        assert!(payload.sections.is_empty());
    }

    #[test]
    fn equality_threshold_keeps_its_comparator_and_unit() {
        let threshold = CaptureQualityThreshold {
            comparison: CaptureThresholdComparison::EqualTo,
            value: 4.0,
            unit: "samples".to_string(),
        };
        assert_eq!(format_threshold(&threshold), "= 4 samples");
    }
}
