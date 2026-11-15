//! Shared assembled report content and the projection to the browser format.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::capture_quality::{
    CaptureQualityReport, CaptureQualityReportError, CaptureQualitySection, capture_quality_section,
};
use crate::report_table::{ReportTable, ReportTableGroup, render_table_html};
use crate::time_of_flight::render_timing_html;

/// An assembled report with sections in dashboard order.
#[derive(Debug, Clone)]
pub struct ReportDocument {
    header: Value,
    /// Ordered sections, including dashboard and channel placement.
    pub sections: Vec<ReportSection>,
    /// Reasons that report preparation could not supply particular evidence.
    pub preparation_status: Value,
}

/// A section in the assembled report.
#[derive(Debug, Clone)]
pub enum ReportSection {
    /// A structured table usable by both renderers.
    Table(TableSection),
    /// A report section containing multiple related tables.
    Tables(TableGroupSection),
    /// One or more paragraphs of report text.
    Text(TextSection),
    /// An ordered group of diagnostic text, tables, and charts.
    Diagnostics(DiagnosticSection),
    /// Deployed filters grouped by source and processing pass.
    Filters(FilterSection),
    /// Saved playback eligibility and binding notices.
    PlaybackNotices(Vec<ReportNotice>),
    /// Validated capture findings and source identities.
    CaptureQuality(CaptureQualitySection),
    /// An existing chart or a section whose content is still HTML.
    Legacy(Value),
}

/// A table and its placement in the report.
#[derive(Debug, Clone)]
pub struct TableSection {
    /// Text and cells in display order.
    pub table: ReportTable,
    style: TableStyle,
    placement: Map<String, Value>,
}

impl TableSection {
    /// Returns the dashboard bucket, if the section belongs to one.
    pub fn bucket(&self) -> Option<&str> {
        self.placement.get("bucket").and_then(Value::as_str)
    }

    /// Returns the dashboard heading, if the section belongs to one.
    pub fn subhead(&self) -> Option<&str> {
        self.placement.get("subhead").and_then(Value::as_str)
    }

    /// Returns the channel tab, if the section is channel-specific.
    pub fn tab(&self) -> Option<&str> {
        self.placement.get("tab").and_then(Value::as_str)
    }

    fn browser_section(&self) -> Value {
        let mut output = Map::new();
        output.insert("kind".into(), json!("html"));
        output.insert("html".into(), json!(self.style.render(&self.table)));
        output.extend(self.placement.clone());
        Value::Object(output)
    }
}

/// Related tables and their placement in the report.
#[derive(Debug, Clone)]
pub struct TableGroupSection {
    /// Group heading and tables.
    pub group: ReportTableGroup,
    placement: Map<String, Value>,
}

impl TableGroupSection {
    /// Returns the dashboard bucket, if the section belongs to one.
    pub fn bucket(&self) -> Option<&str> {
        self.placement.get("bucket").and_then(Value::as_str)
    }

    /// Returns the dashboard heading, if the section belongs to one.
    pub fn subhead(&self) -> Option<&str> {
        self.placement.get("subhead").and_then(Value::as_str)
    }

    /// Returns the channel tab, if the section is channel-specific.
    pub fn tab(&self) -> Option<&str> {
        self.placement.get("tab").and_then(Value::as_str)
    }

    fn browser_section(&self) -> Value {
        let mut output = Map::new();
        output.insert("kind".into(), json!("html"));
        output.insert(
            "html".into(),
            json!(crate::crossover_config::render_crossover_tables(
                &self.group
            )),
        );
        output.extend(self.placement.clone());
        Value::Object(output)
    }
}

/// Ordered paragraphs and their dashboard placement.
#[derive(Debug, Clone)]
pub struct TextSection {
    /// Optional heading shown before the paragraphs.
    pub heading: Option<String>,
    /// Plain text in display order.
    pub paragraphs: Vec<String>,
    format: TextFormat,
    placement: Map<String, Value>,
}

impl TextSection {
    /// Returns the dashboard bucket.
    pub fn bucket(&self) -> Option<&str> {
        self.placement.get("bucket").and_then(Value::as_str)
    }

    /// Returns the dashboard heading.
    pub fn subhead(&self) -> Option<&str> {
        self.placement.get("subhead").and_then(Value::as_str)
    }

    /// Returns the channel tab, if any.
    pub fn tab(&self) -> Option<&str> {
        self.placement.get("tab").and_then(Value::as_str)
    }

    fn browser_section(&self) -> Value {
        use std::fmt::Write as _;
        let mut html = String::new();
        if matches!(self.format, TextFormat::FilterSummary) {
            html.push_str("<div class=\"filters-section\">\n");
        }
        if let Some(heading) = &self.heading {
            let heading = crate::report_helpers::escape_html(heading);
            if matches!(self.format, TextFormat::FilterSummary) {
                let _ = writeln!(html, "    <h3>{heading}</h3>");
            } else {
                let _ = writeln!(html, "<h2>{heading}</h2>");
            }
        }
        for paragraph in &self.paragraphs {
            let text = crate::report_helpers::escape_html(paragraph);
            match self.format {
                TextFormat::Plain => {
                    let _ = write!(html, "<p>{text}</p>");
                }
                TextFormat::Caption => {
                    let _ = write!(html, "<p class=\"epa-footer\">{text}</p>");
                }
                TextFormat::CaptionLine => {
                    let _ = writeln!(html, "<p class=\"epa-footer\">{text}</p>");
                }
                TextFormat::FilterSummary => {
                    let _ = writeln!(html, "    <div class=\"filter-list\">{text}</div>");
                }
            }
        }
        if matches!(self.format, TextFormat::FilterSummary) {
            html.push_str("</div>\n");
        }
        let mut output = Map::new();
        output.insert("kind".into(), json!("html"));
        output.insert("html".into(), json!(html));
        output.extend(self.placement.clone());
        Value::Object(output)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TextFormat {
    #[default]
    Plain,
    Caption,
    CaptionLine,
    FilterSummary,
}

/// A diagnostic panel containing context and separately headed evidence tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePanel {
    /// Panel heading.
    pub title: String,
    /// Explanatory text.
    pub paragraphs: Vec<String>,
    /// Tables, or explanatory notes when a group has no rows.
    pub groups: Vec<ReportTable>,
}

/// A numbered DSP stage shown in the signal-flow diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFlowStage {
    /// Node number, also shown on the chart.
    pub number: usize,
    /// Display name of the processing stage.
    pub label: String,
    /// Label on the expandable parameters.
    pub heading: String,
    /// Saved parameters, formatted as JSON.
    pub parameters: String,
}

/// An EPA score change, with display values computed by the report producer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpaDelta {
    /// One or both values are absent.
    Unavailable,
    /// The change is below the report's display threshold.
    Unchanged,
    /// The score changed in the favorable direction.
    Improved(String),
    /// The score changed in the unfavorable direction.
    Worsened(String),
}

impl EpaDelta {
    /// Text shown in the delta column.
    pub fn text(&self) -> &str {
        match self {
            Self::Unavailable => "-",
            Self::Unchanged => "=",
            Self::Improved(text) | Self::Worsened(text) => text,
        }
    }

    pub(crate) fn browser_html(&self) -> String {
        let style = match self {
            Self::Unavailable | Self::Unchanged => "color:#999",
            Self::Improved(_) => "color:#2ecc71;font-weight:600",
            Self::Worsened(_) => "color:#e74c3c;font-weight:600",
        };
        format!(
            "<span style=\"{style}\">{}</span>",
            crate::report_helpers::escape_html(self.text())
        )
    }
}

/// Before/after values for one psychoacoustic metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpaScoreRow {
    /// Display label including any unit.
    pub metric: String,
    /// Score before correction.
    pub before: String,
    /// Score after correction.
    pub after: String,
    /// Change with its display classification.
    pub delta: EpaDelta,
}

/// Heading for the per-channel EPA score table.
pub const EPA_TITLE: &str = "EPA Psychoacoustic Scores";
/// Column labels for the per-channel EPA score table.
pub const EPA_COLUMNS: [&str; 4] = ["Metric", "Before EQ", "After EQ", "Δ"];
/// How to read the per-channel EPA score table.
pub const EPA_CAPTION: &str = "Higher is better for Preference / Evaluation / Total loudness / Loudness balance; lower is better for Activity / Sharpness deviation / Roughness. Δ cells are green when the change improves the metric.";

/// Formatted resonance evidence for one detected candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceCell {
    /// Frequency, relative level, and decay time in display order.
    pub lines: [String; 3],
    /// RGB color identifying frequency.
    pub color: u32,
}

/// Whether resonance detection ran and what it found for a speaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResonanceEvidence {
    /// Saved evidence could not be validated.
    Unavailable,
    /// The validated detector output contains no candidates.
    NoCandidates,
    /// Detected candidates sorted by frequency.
    Candidates(Vec<ResonanceCell>),
}

impl ResonanceEvidence {
    /// The report message for rows without candidates.
    pub fn status(&self) -> Option<&'static str> {
        match self {
            Self::Unavailable => Some("Unavailable"),
            Self::NoCandidates => Some("No candidates detected"),
            Self::Candidates(_) => None,
        }
    }
}

/// Resonance evidence for one speaker or measured driver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceRow {
    /// Display name of the speaker or driver.
    pub name: String,
    /// Validated detection result.
    pub evidence: ResonanceEvidence,
}

/// Heading of the shared resonance summary.
pub const RESONANCE_TITLE: &str = "Room resonance summary — all speakers";
/// Explanation of the resonance summary cells.
pub const RESONANCE_NOTE: &str = "Pre-correction detections. Each cell gives frequency, 60 ms level relative to that speaker’s full-grid peak, and fitted decay time. Colours identify frequency, not severity. These candidates are not a geometric identification of room modes.";

/// Assessment category for a result summary cell.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricTone {
    /// Favorable result.
    Good,
    /// Intermediate result.
    Warning,
    /// Unfavorable result.
    Poor,
    /// A measured input or analysis is missing.
    Pending,
    /// The prerequisites for making the assessment were not met.
    Unassessed,
}

impl MetricTone {
    /// The metric background color, including opacity, as RGBA.
    pub fn background_rgba(self) -> u32 {
        match self {
            Self::Good => 0x2ecc7133,
            Self::Warning => 0xf1c40f33,
            Self::Poor => 0xe74c3c33,
            Self::Pending | Self::Unassessed => 0xeeeeeeff,
        }
    }

    /// An optional foreground color as RGB.
    pub fn foreground_rgb(self) -> Option<u32> {
        match self {
            Self::Pending => Some(0x888888),
            Self::Unassessed => Some(0x666666),
            _ => None,
        }
    }
}

/// Formatted value and explanation for an acoustic result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCell {
    /// Display value.
    pub value: String,
    /// Assessment category.
    pub tone: MetricTone,
    /// Explanation shown on inspection.
    pub detail: Option<String>,
}

impl MetricCell {
    pub(crate) fn pending(field: &str) -> Self {
        Self {
            value: "pending".into(),
            tone: MetricTone::Pending,
            detail: Some(format!("needs roomeq field: {field}")),
        }
    }

    pub(crate) fn browser_html(&self) -> String {
        use crate::report_helpers::escape_html;
        let style = match self.tone {
            MetricTone::Pending => "background:#eee;color:#888".into(),
            MetricTone::Unassessed => "background:#eee;color:#666".into(),
            tone => format!("background:#{:08x}", tone.background_rgba()),
        };
        let detail = self
            .detail
            .as_ref()
            .map(|text| format!(" title=\"{}\"", escape_html(text)))
            .unwrap_or_default();
        format!(
            "<td style=\"{style}\"{detail}>{}</td>",
            escape_html(&self.value)
        )
    }
}

/// Acoustic summary values for one speaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsRow {
    /// Speaker name.
    pub speaker: String,
    /// Metrics in the order of [RESULTS_COLUMNS], after the speaker name.
    pub metrics: [MetricCell; 6],
}

/// Heading of the acoustic results summary.
pub const RESULTS_TITLE: &str = "Results summary";
/// Explanation of the acoustic results and their evidence requirements.
pub const RESULTS_NOTE: &str = "Acoustic values use the imported room impulse response. Operational response is the share of final EQ decisions confirmed as delivered; “Not assessed” means the saved ledger has no final EQ assessment. Missing acoustic values require a measured room IR and valid analysis.";
/// Column labels for the acoustic results summary.
pub const RESULTS_COLUMNS: [&str; 7] = [
    "Speaker",
    "Operational room response (%)",
    "Early reflection level (dB)",
    "Early vs late ratio (dB)",
    "T60 flatness in window (%)",
    "T60 within ITU recommendation (%)",
    "Deepest notch < 300 Hz (dB)",
];

/// Shared explanation accompanying the directional arrival plot.
pub const ARRIVAL_PLOT_NOTE: &str =
    "Azimuth uses the recorded session axes. Mirror-ambiguous arrivals are omitted.";

/// A display block within a diagnostic section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosticBlock {
    /// Main heading for a diagnostic.
    Heading(String),
    /// A heading for an individual capture comparison.
    MajorHeading(String),
    /// The verdict and process result from the recorded capture run.
    VerificationStatus {
        /// Recorded verdict.
        status: String,
        /// Number of pre-result checks.
        count: u64,
        /// Process exit code.
        exit_code: i64,
    },
    /// Expandable formatted capture metadata.
    PreformattedDisclosure {
        /// Disclosure title.
        title: String,
        /// Saved metadata, formatted as text.
        text: String,
    },
    /// Heading within a diagnostic.
    Subheading(String),
    /// Explanatory text.
    Paragraph(String),
    /// Explanatory text rendered as a caption.
    Caption(String),
    /// A prominent diagnostic result.
    Emphasis(String),
    /// A captioned table of diagnostic values.
    Table(ReportTable),
    /// A diagnostic evidence table with an optional explanatory note.
    EvidenceTable(ReportTable),
    /// A curve summary table, with newline-separated extrema inside cells.
    CurveTable {
        /// Summary table.
        table: ReportTable,
        /// Dim unavailable values in metric columns.
        dim_unavailable: bool,
    },
    /// An evidence panel with context and multiple groups.
    EvidencePanel(EvidencePanel),
    /// Expandable parameter rows associated with a saved DSP diagram.
    SignalFlowStages(Vec<SignalFlowStage>),
    /// Per-channel psychoacoustic scores and changes.
    EpaScores(Vec<EpaScoreRow>),
    /// Mixed-phase and FIR timing measurements with a trailing summary.
    MixedPhaseTable(ReportTable),
    /// Acoustic results, with status and evidence details per metric.
    ResultsSummary(Vec<ResultsRow>),
    /// A table of recorded decisions, reasons, or output loss.
    DecisionTable(ReportTable),
    /// Recorded evidence listed in display order.
    EvidenceList(Vec<String>),
    /// Counts, identifiers, and reasons for the final acceptance sets.
    AcceptanceSets(Vec<String>),
    /// Expandable recorded constraints and stage history.
    StageHistory {
        /// Disclosure title.
        title: String,
        /// Explanation displayed above the history.
        note: String,
        /// Saved diagnostic history in display order.
        items: Vec<String>,
    },
    /// Captured arrival evidence for one channel.
    CaptureArrival {
        /// Arrival candidate rows and channel notes.
        table: ReportTable,
        /// Direction plot when the evidence permits one.
        chart: Option<Value>,
    },
    /// The all-speaker room resonance summary.
    ResonanceSummary {
        /// Number of candidate columns across the report.
        columns: usize,
        /// Speaker and driver evidence.
        rows: Vec<ResonanceRow>,
    },
    /// A caption followed by a link to an external reference.
    Reference {
        /// Explanatory caption.
        text: String,
        /// Link label.
        label: String,
        /// Reference URL.
        url: String,
    },
    /// A table with row labels and a highlighted final summary row.
    Matrix(ReportTable),
    /// A plot encoded with the report chart schema.
    Chart(Value),
    /// A retained acceptance prediction, using the standalone encoding.
    AcceptanceChart(Value),
    /// A verified capture comparison chart.
    CaptureChart(Value),
}

/// Explanatory text following a recorded capture verification verdict.
pub fn verification_status_detail(count: u64, exit_code: i64) -> String {
    format!(
        "; {count} pre-result capture check(s); process exit code {exit_code}. This is the report’s recorded verdict, not a new validation by the viewer."
    )
}

pub(crate) fn diagnostic_html(blocks: &[DiagnosticBlock]) -> String {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let mut html = String::new();
    for block in blocks {
        match block {
            DiagnosticBlock::MajorHeading(text) => {
                let _ = write!(html, "<h2>{}</h2>", escape_html(text));
            }
            DiagnosticBlock::VerificationStatus {
                status,
                count,
                exit_code,
            } => {
                let _ = write!(
                    html,
                    "<p>Recorded capture verification status: <strong>{}</strong>{}</p>",
                    escape_html(status),
                    verification_status_detail(*count, *exit_code)
                );
            }
            DiagnosticBlock::PreformattedDisclosure { title, text } => {
                let _ = write!(
                    html,
                    "<details><summary>{}</summary><pre>{}</pre></details>",
                    escape_html(title),
                    escape_html(text)
                );
            }
            DiagnosticBlock::Heading(text) => {
                let _ = write!(html, "<h3>{}</h3>", escape_html(text));
            }
            DiagnosticBlock::Subheading(text) => {
                let _ = write!(html, "<h4>{}</h4>", escape_html(text));
            }
            DiagnosticBlock::Paragraph(text) => {
                let _ = write!(html, "<p>{}</p>", escape_html(text));
            }
            DiagnosticBlock::Caption(text) => {
                let _ = write!(html, "<p class=\"epa-footer\">{}</p>", escape_html(text));
            }
            DiagnosticBlock::Emphasis(text) => {
                let _ = write!(html, "<p><strong>{}</strong></p>", escape_html(text));
            }
            DiagnosticBlock::Reference { text, label, url } => {
                let _ = write!(
                    html,
                    "<p class=\"epa-footer\">{} <a href=\"{}\">{}</a>.</p>",
                    escape_html(text),
                    escape_html(url),
                    escape_html(label),
                );
            }
            DiagnosticBlock::EvidenceTable(table) => {
                if !table.note.is_empty() {
                    html.push_str("<div class=\"filters-section\">");
                }
                let _ = write!(html, "<h3>{}</h3>", escape_html(&table.title));
                if !table.note.is_empty() {
                    let _ = write!(
                        html,
                        "<p class=\"epa-footer\">{}</p>",
                        escape_html(&table.note)
                    );
                }
                crate::report_table::write_table_html(&mut html, table, "epa-table", "");
                if !table.note.is_empty() {
                    html.push_str("</div>");
                }
            }
            DiagnosticBlock::CurveTable {
                table,
                dim_unavailable,
            } => {
                write_curve_table(&mut html, table, *dim_unavailable);
            }
            DiagnosticBlock::EvidencePanel(panel) => {
                let _ = write!(
                    html,
                    "<div class=\"filters-section\"><h3>{}</h3>",
                    escape_html(&panel.title)
                );
                for paragraph in &panel.paragraphs {
                    let _ = write!(html, "<p>{}</p>", escape_html(paragraph));
                }
                for group in &panel.groups {
                    let _ = write!(html, "<h4>{}</h4>", escape_html(&group.title));
                    if group.rows.is_empty() {
                        let _ = write!(html, "<p>{}</p>", escape_html(&group.note));
                    } else {
                        crate::report_table::write_table_html(&mut html, group, "epa-table", "");
                    }
                }
                html.push_str("</div>");
            }
            DiagnosticBlock::SignalFlowStages(stages) => {
                html.push_str("<table><thead><tr><th>Node</th><th>Stage</th><th>Saved parameters</th></tr></thead><tbody>");
                for stage in stages {
                    let _ = write!(
                        html,
                        "<tr><td>{}</td><td>{}</td><td><details><summary>{}</summary><pre>{}</pre></details></td></tr>",
                        stage.number,
                        escape_html(&stage.label),
                        escape_html(&stage.heading),
                        escape_html(&stage.parameters),
                    );
                }
                html.push_str("</tbody></table>");
            }
            DiagnosticBlock::EpaScores(rows) => write_epa_scores(&mut html, rows),
            DiagnosticBlock::MixedPhaseTable(table) => write_mixed_phase(&mut html, table),
            DiagnosticBlock::ResonanceSummary { columns, rows } => {
                write_resonance_summary(&mut html, *columns, rows);
            }
            DiagnosticBlock::DecisionTable(table) => write_decision_table(&mut html, table),
            DiagnosticBlock::EvidenceList(items) => write_evidence_list(&mut html, items, ""),
            DiagnosticBlock::AcceptanceSets(items) => {
                write_evidence_list(&mut html, items, " class=\"acceptance-sets\"");
            }
            DiagnosticBlock::StageHistory { title, note, items } => {
                let _ = write!(
                    html,
                    "<details><summary>{}</summary><p>{}</p>",
                    escape_html(title),
                    escape_html(note),
                );
                write_evidence_list(&mut html, items, "");
                html.push_str("</details>");
            }
            DiagnosticBlock::ResultsSummary(rows) => write_results_summary(&mut html, rows),
            DiagnosticBlock::CaptureArrival { table, chart } => {
                let _ = write!(
                    html,
                    "<section class=\"capture-reflections\"><h3>{}</h3><p>{}</p>",
                    escape_html(&table.title),
                    escape_html(&table.note),
                );
                if let Some(chart) = chart {
                    html.push_str(&crate::evidence_sections::embedded_report_figure(chart));
                    let _ = write!(html, "<p>{ARRIVAL_PLOT_NOTE}</p>");
                }
                crate::report_table::write_table_html(&mut html, table, "summary-table", "");
                html.push_str("</section>");
            }
            DiagnosticBlock::Matrix(table) => write_diagnostic_matrix(&mut html, table),
            DiagnosticBlock::Table(table) => {
                html.push_str("<table>");
                if !table.title.is_empty() {
                    let _ = write!(html, "<caption>{}</caption>", escape_html(&table.title));
                }
                html.push_str("<tr>");
                for column in &table.columns {
                    let _ = write!(html, "<th>{}</th>", escape_html(column));
                }
                html.push_str("</tr>");
                for row in &table.rows {
                    html.push_str("<tr>");
                    for cell in row {
                        let _ = write!(html, "<td>{}</td>", escape_html(cell));
                    }
                    html.push_str("</tr>");
                }
                html.push_str("</table>");
            }
            DiagnosticBlock::CaptureChart(chart) => {
                html.push_str(&crate::evidence_sections::embedded_report_figure(chart));
            }
            DiagnosticBlock::AcceptanceChart(chart) => {
                let _ = write!(
                    html,
                    "<div class=\"report-figure\" data-section=\"{}\"></div>",
                    escape_html(&crate::evidence_sections::json_python_separators(chart)),
                );
            }
            DiagnosticBlock::Chart(chart) => {
                html.push_str(&crate::channel_evidence::embedded_figure(chart))
            }
        }
    }
    html
}

fn write_evidence_list(html: &mut String, items: &[String], attributes: &str) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = write!(html, "<ul{attributes}>");
    for item in items {
        let _ = write!(html, "<li>{}</li>", escape_html(item));
    }
    html.push_str("</ul>");
}

fn write_decision_table(html: &mut String, table: &ReportTable) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    if !table.title.is_empty() {
        let _ = write!(html, "<h3>{}</h3>", escape_html(&table.title));
    }
    html.push_str("<div style=\"overflow-x:auto\"><table style=\"width:100%;border-collapse:collapse;text-align:left\"><thead><tr>");
    for column in &table.columns {
        let _ = write!(html, "<th>{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead><tbody>");
    for row in &table.rows {
        html.push_str("<tr>");
        for cell in row {
            let _ = write!(
                html,
                "<td style=\"padding:8px;border-bottom:1px solid #ddd;vertical-align:top\">{}</td>",
                escape_html(cell),
            );
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div>");
}

fn write_results_summary(html: &mut String, rows: &[ResultsRow]) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = write!(
        html,
        "<div class=\"filters-section\">\n<h3>{RESULTS_TITLE}</h3>\n<p class=\"epa-footer\">{RESULTS_NOTE}</p>\n<table class=\"epa-table\"><thead><tr>"
    );
    for column in RESULTS_COLUMNS {
        let _ = write!(html, "<th>{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead><tbody>\n");
    for row in rows {
        let _ = write!(html, "<tr><td>{}</td>", escape_html(&row.speaker));
        for metric in &row.metrics {
            html.push_str(&metric.browser_html());
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</tbody></table></div>\n");
}

fn write_resonance_summary(html: &mut String, columns: usize, rows: &[ResonanceRow]) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = write!(
        html,
        "<h2>{RESONANCE_TITLE}</h2><p>{RESONANCE_NOTE}</p><table><tr><th>Speaker</th>"
    );
    for index in 1..=columns {
        let _ = write!(html, "<th>Mode {index}</th>");
    }
    html.push_str("</tr>");
    for row in rows {
        let _ = write!(html, "<tr><th>{}</th>", escape_html(&row.name));
        if let ResonanceEvidence::Candidates(cells) = &row.evidence {
            for cell in cells {
                let [frequency, level, decay] = cell.lines.each_ref().map(|line| escape_html(line));
                let color = cell.color;
                let _ = write!(
                    html,
                    "<td style=\"background:#{color:06x}33;border-top:4px solid #{color:06x}\">{frequency}<br>{level}<br>{decay}</td>"
                );
            }
            for _ in cells.len()..columns {
                html.push_str("<td>—</td>");
            }
        } else if let Some(status) = row.evidence.status() {
            let _ = write!(html, "<td colspan=\"{columns}\">{status}</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
}

fn write_epa_scores(html: &mut String, rows: &[EpaScoreRow]) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = write!(
        html,
        "                <div class=\"epa-section\">\n                    <h3>{EPA_TITLE}</h3>\n                    <table class=\"epa-table\">\n                        <thead><tr>",
    );
    for column in EPA_COLUMNS {
        let _ = write!(html, "<th>{column}</th>");
    }
    html.push_str("</tr></thead>\n                        <tbody>\n");
    for row in rows {
        let _ = writeln!(
            html,
            "                        <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&row.metric),
            escape_html(&row.before),
            escape_html(&row.after),
            row.delta.browser_html(),
        );
    }
    let _ = write!(
        html,
        "                        </tbody>\n                    </table>\n                    <p class=\"epa-footer\">{EPA_CAPTION}</p>\n                </div>\n",
    );
}

fn write_mixed_phase(html: &mut String, table: &ReportTable) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = writeln!(
        html,
        "<div class=\"filters-section mixed-phase-section\">\n<h3>{}</h3>",
        escape_html(&table.title),
    );
    if !table.rows.is_empty() {
        html.push_str("<table class=\"bm-table\">\n<thead><tr>");
        for column in &table.columns {
            let _ = write!(html, "<th>{}</th>", escape_html(column));
        }
        html.push_str("</tr></thead>\n<tbody>\n");
        for row in &table.rows {
            html.push_str("<tr>");
            for cell in row {
                let _ = write!(html, "<td>{}</td>", escape_html(cell));
            }
            html.push_str("</tr>\n");
        }
        html.push_str("</tbody>\n</table>\n");
    }
    if !table.note.is_empty() {
        let _ = writeln!(
            html,
            "<p class=\"epa-footer\">{}</p>",
            escape_html(&table.note)
        );
    }
    html.push_str("</div>\n");
}

fn write_curve_table(html: &mut String, table: &ReportTable, dim_unavailable: bool) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = writeln!(
        html,
        "<div class=\"filters-section\">\n<h3>{}</h3>\n<p class=\"epa-footer\">{}</p>",
        escape_html(&table.title),
        escape_html(&table.note),
    );
    html.push_str("<table class=\"epa-table\"><thead><tr>");
    for column in &table.columns {
        let _ = write!(html, "<th>{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead><tbody>\n");
    for row in &table.rows {
        html.push_str("<tr>");
        for (index, cell) in row.iter().enumerate() {
            let attrs = if dim_unavailable && index > 0 && cell == "n/a" {
                " style=\"color:#999\""
            } else {
                ""
            };
            let mut value = escape_html(cell);
            if index > 0 {
                value = value.replace('\n', "<br>");
            }
            let _ = write!(html, "<td{attrs}>{value}</td>");
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</tbody></table></div>\n");
}

fn write_diagnostic_matrix(html: &mut String, table: &ReportTable) {
    use crate::report_helpers::escape_html;
    use std::fmt::Write as _;
    let _ = write!(
        html,
        "<div style=\"overflow-x:auto\"><table class=\"summary-table\"><caption>{}</caption><thead><tr>",
        escape_html(&table.title)
    );
    for column in &table.columns {
        let _ = write!(html, "<th scope=\"col\">{}</th>", escape_html(column));
    }
    html.push_str("</tr></thead><tbody>");
    for (index, row) in table.rows.iter().enumerate() {
        let Some((label, cells)) = row.split_first() else {
            continue;
        };
        let _ = write!(html, "<tr><th scope=\"row\">{}</th>", escape_html(label));
        for cell in cells {
            let cell = escape_html(cell);
            if index + 1 == table.rows.len() {
                let _ = write!(html, "<td><strong>{cell}</strong></td>");
            } else {
                let _ = write!(html, "<td>{cell}</td>");
            }
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div>");
}

/// A group of diagnostic blocks with its placement in the report.
#[derive(Debug, Clone)]
pub struct DiagnosticSection {
    /// Diagnostic blocks in display order.
    pub blocks: Vec<DiagnosticBlock>,
    /// Optional section heading and introduction.
    pub introduction: Option<DiagnosticIntroduction>,
    placement: Map<String, Value>,
}

/// Presentation of the opening text and enclosing section in HTML.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum DiagnosticSectionStyle {
    /// Ordinary diagnostic section.
    #[default]
    Plain,
    /// Recorded rationale and delivery evidence.
    CorrectionExplanation,
    /// Saved acceptance diagnostics.
    AcceptanceViews,
    /// Capture comparison diagnostics rejected by the host's binding checks.
    CaptureViewsUnavailable,
    /// Saved capture comparison diagnostics.
    CaptureViews,
}

/// Opening text for a group of diagnostic blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticIntroduction {
    /// Group heading.
    pub title: String,
    /// Explanation shown before the blocks.
    pub text: String,
    /// Browser section presentation.
    #[serde(default)]
    pub style: DiagnosticSectionStyle,
}

impl DiagnosticSection {
    /// Returns the dashboard bucket.
    pub fn bucket(&self) -> Option<&str> {
        self.placement.get("bucket").and_then(Value::as_str)
    }

    /// Returns the dashboard heading.
    pub fn subhead(&self) -> Option<&str> {
        self.placement.get("subhead").and_then(Value::as_str)
    }

    /// Returns the channel tab.
    pub fn tab(&self) -> Option<&str> {
        self.placement.get("tab").and_then(Value::as_str)
    }

    fn browser_section(&self) -> Value {
        let mut output = Map::new();
        output.insert("kind".into(), json!("html"));
        let contents = diagnostic_html(&self.blocks);
        let html = match &self.introduction {
            Some(intro) => {
                let (attributes, heading, ending) = match intro.style {
                    DiagnosticSectionStyle::Plain => ("", "h2", ""),
                    DiagnosticSectionStyle::CorrectionExplanation => {
                        (" class=\"correction-explanation\"", "h2", "\n")
                    }
                    DiagnosticSectionStyle::AcceptanceViews => {
                        (" class=\"acceptance-views\"", "h2", "")
                    }
                    DiagnosticSectionStyle::CaptureViews => (" class=\"capture-views\"", "h1", ""),
                    DiagnosticSectionStyle::CaptureViewsUnavailable => {
                        (" class=\"capture-views\"", "h1", "\n")
                    }
                };
                let text = if intro.text.is_empty() {
                    String::new()
                } else {
                    format!("<p>{}</p>", crate::report_helpers::escape_html(&intro.text))
                };
                format!(
                    "<section{attributes}><{heading}>{}</{heading}>{text}{contents}</section>{ending}",
                    crate::report_helpers::escape_html(&intro.title),
                )
            }
            None => contents,
        };
        output.insert("html".into(), json!(html));
        output.extend(self.placement.clone());
        Value::Object(output)
    }
}

/// An EQ filter listing and its dashboard placement.
#[derive(Debug, Clone)]
pub struct FilterSection {
    /// Filter descriptions and their origins.
    pub listing: crate::eq_sections::EqFilterListing,
    placement: Map<String, Value>,
}

impl FilterSection {
    /// Returns the dashboard bucket.
    pub fn bucket(&self) -> Option<&str> {
        self.placement.get("bucket").and_then(Value::as_str)
    }

    /// Returns the dashboard heading.
    pub fn subhead(&self) -> Option<&str> {
        self.placement.get("subhead").and_then(Value::as_str)
    }

    /// Returns the channel tab.
    pub fn tab(&self) -> Option<&str> {
        self.placement.get("tab").and_then(Value::as_str)
    }

    fn browser_section(&self) -> Value {
        let mut output = Map::new();
        output.insert("kind".into(), json!("html"));
        output.insert("html".into(), json!(self.listing.html()));
        output.extend(self.placement.clone());
        Value::Object(output)
    }
}

/// A labelled statement about saved playback evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportNotice {
    /// Heading for the evidence.
    pub title: String,
    /// Display severity of the recorded evidence.
    pub tone: NoticeTone,
    /// Details in display order.
    pub paragraphs: Vec<String>,
}

/// Display severity of a report notice.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeTone {
    /// The recorded check passed.
    Success,
    /// Evidence is conditional or unavailable.
    Warning,
    /// The recorded check failed or approval is unavailable.
    Error,
}

fn playback_browser_section(notices: &[ReportNotice]) -> Value {
    json!({
        "kind":"html",
        "html":crate::evidence_sections::playback_notices_html(notices),
        "tab":null,
        "flat":true,
    })
}

impl ReportDocument {
    pub(crate) fn from_assembled(
        mut header: Value,
        preparation_status: Value,
    ) -> Result<Self, String> {
        let Value::Array(sections) = header["sections"].take() else {
            return Err("assembled report requires an array of sections".into());
        };
        let sections = sections
            .into_iter()
            .map(|section| match section["kind"].as_str() {
                Some("table") => decode_table_section(section).map(ReportSection::Table),
                Some("crossover_tables") => decode_table_group(section).map(ReportSection::Tables),
                Some("text") => decode_text_section(section).map(ReportSection::Text),
                Some("diagnostics") => decode_diagnostics(section).map(ReportSection::Diagnostics),
                Some("eq_filters") => decode_filter_section(section).map(ReportSection::Filters),
                Some("playback_notices") => {
                    decode_notices(section).map(ReportSection::PlaybackNotices)
                }
                _ => Ok(ReportSection::Legacy(section)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            header,
            sections,
            preparation_status,
        })
    }

    /// Returns the report title.
    pub fn title(&self) -> &str {
        self.header["title"].as_str().unwrap_or("Room EQ")
    }

    /// Adds validated capture evidence in the DSP dashboard.
    ///
    /// # Errors
    /// Returns an error if the evidence uses an unsupported schema or is invalid.
    pub fn add_capture_quality(
        &mut self,
        report: &CaptureQualityReport,
    ) -> Result<(), CaptureQualityReportError> {
        if let Some(section) = capture_quality_section(report)? {
            self.sections.push(ReportSection::CaptureQuality(section));
        }
        Ok(())
    }

    /// Projects the document to the existing standalone report payload.
    pub fn browser_payload(&self) -> Value {
        let mut output = self.header.clone();
        output["sections"] = Value::Array(
            self.sections
                .iter()
                .map(|section| match section {
                    ReportSection::Table(section) => section.browser_section(),
                    ReportSection::Tables(section) => section.browser_section(),
                    ReportSection::Text(section) => section.browser_section(),
                    ReportSection::Diagnostics(section) => section.browser_section(),
                    ReportSection::Filters(section) => section.browser_section(),
                    ReportSection::PlaybackNotices(notices) => playback_browser_section(notices),
                    ReportSection::CaptureQuality(section) => section.browser_section(),
                    ReportSection::Legacy(value) => value.clone(),
                })
                .collect(),
        );
        output
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TableStyle {
    Gain,
    Timing,
    Waveform,
    CaptureClock,
    BassSummary,
    BassTable,
}

impl TableStyle {
    fn render(self, table: &ReportTable) -> String {
        match self {
            Self::Gain => render_table_html(table),
            Self::Timing => render_timing_html(table),
            Self::Waveform => crate::evidence_sections::render_waveform_status(table),
            Self::CaptureClock => crate::evidence_sections::render_capture_clock_qa(table),
            Self::BassSummary => crate::bass_management::render_bass_summary(table),
            Self::BassTable => crate::bass_management::render_bass_table(table),
        }
    }
}

pub(crate) fn table_section(table: Option<ReportTable>, style: TableStyle) -> Value {
    table.map_or(
        Value::Null,
        |table| json!({"kind":"table","table":table,"style":style,"tab":null}),
    )
}

pub(crate) fn crossover_section(group: Option<ReportTableGroup>) -> Value {
    group.map_or(
        Value::Null,
        |group| json!({"kind":"crossover_tables","group":group,"tab":null}),
    )
}

pub(crate) fn section_fragment(mut section: Value) -> Result<String, String> {
    project_section(&mut section)?;
    Ok(section["html"].as_str().unwrap_or_default().to_owned())
}

pub(crate) fn paragraph_section(paragraphs: &[impl AsRef<str>]) -> Value {
    let paragraphs: Vec<&str> = paragraphs.iter().map(AsRef::as_ref).collect();
    json!({"kind":"text","paragraphs":paragraphs,"tab":null})
}

pub(crate) fn caption_section(text: impl AsRef<str>, newline: bool) -> Value {
    let format = if newline {
        TextFormat::CaptionLine
    } else {
        TextFormat::Caption
    };
    json!({"kind":"text","paragraphs":[text.as_ref()],"format":format,"tab":null})
}

pub(crate) fn filter_summary(heading: &str, text: String) -> Value {
    json!({"kind":"text","heading":heading,"paragraphs":[text],"format":TextFormat::FilterSummary,"tab":null})
}

pub(crate) fn channel_heading(label: &str, caption: &str) -> Value {
    let paragraphs: Vec<_> = (!caption.is_empty())
        .then_some(caption)
        .into_iter()
        .collect();
    json!({"kind":"text","heading":format!("Channel: {label}"),"paragraphs":paragraphs,"format":TextFormat::CaptionLine,"tab":null})
}

pub(crate) fn project_section(section: &mut Value) -> Result<(), String> {
    match section["kind"].as_str() {
        Some("table") => *section = decode_table_section(section.take())?.browser_section(),
        Some("crossover_tables") => {
            *section = decode_table_group(section.take())?.browser_section()
        }
        Some("text") => *section = decode_text_section(section.take())?.browser_section(),
        Some("diagnostics") => *section = decode_diagnostics(section.take())?.browser_section(),
        Some("eq_filters") => *section = decode_filter_section(section.take())?.browser_section(),
        Some("playback_notices") => {
            *section = playback_browser_section(&decode_notices(section.take())?)
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn project_sections(payload: &mut Value) -> Result<(), String> {
    if let Some(sections) = payload["sections"].as_array_mut() {
        for section in sections {
            project_section(section)?;
        }
    }
    Ok(())
}

fn decode_notices(mut section: Value) -> Result<Vec<ReportNotice>, String> {
    serde_json::from_value(section["notices"].take())
        .map_err(|error| format!("invalid report notices: {error}"))
}

pub(crate) fn eq_filter_section(listing: crate::eq_sections::EqFilterListing) -> Value {
    json!({"kind":"eq_filters","listing":listing,"tab":null})
}

fn decode_filter_section(section: Value) -> Result<FilterSection, String> {
    let Value::Object(mut placement) = section else {
        return Err("filter section requires an object".into());
    };
    placement.shift_remove("kind");
    let listing = serde_json::from_value(placement.shift_remove("listing").unwrap_or_default())
        .map_err(|error| format!("invalid filter listing: {error}"))?;
    Ok(FilterSection { listing, placement })
}

pub(crate) fn diagnostic_section(blocks: Vec<DiagnosticBlock>) -> Value {
    json!({"kind":"diagnostics","blocks":blocks,"tab":null})
}

fn decode_diagnostics(section: Value) -> Result<DiagnosticSection, String> {
    let Value::Object(mut placement) = section else {
        return Err("diagnostic section requires an object".into());
    };
    placement.shift_remove("kind");
    let blocks = serde_json::from_value(placement.shift_remove("blocks").unwrap_or_default())
        .map_err(|error| format!("invalid diagnostic blocks: {error}"))?;
    let introduction =
        serde_json::from_value(placement.shift_remove("introduction").unwrap_or_default())
            .map_err(|error| format!("invalid diagnostic introduction: {error}"))?;
    Ok(DiagnosticSection {
        blocks,
        introduction,
        placement,
    })
}

fn decode_text_section(mut section: Value) -> Result<TextSection, String> {
    let Value::Object(ref mut placement) = section else {
        return Err("report text section requires an object".into());
    };
    placement.shift_remove("kind");
    let paragraphs =
        serde_json::from_value(placement.shift_remove("paragraphs").unwrap_or_default())
            .map_err(|error| format!("invalid report paragraphs: {error}"))?;
    let heading = serde_json::from_value(placement.shift_remove("heading").unwrap_or_default())
        .map_err(|error| format!("invalid report heading: {error}"))?;
    let format = placement
        .shift_remove("format")
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| format!("invalid report text format: {error}"))?
        .unwrap_or_default();
    Ok(TextSection {
        heading,
        paragraphs,
        format,
        placement: std::mem::take(placement),
    })
}

fn decode_table_group(mut section: Value) -> Result<TableGroupSection, String> {
    let Value::Object(ref mut placement) = section else {
        return Err("report table group requires an object".into());
    };
    placement.shift_remove("kind");
    let group = serde_json::from_value(placement.shift_remove("group").unwrap_or_default())
        .map_err(|error| format!("invalid report table group: {error}"))?;
    Ok(TableGroupSection {
        group,
        placement: std::mem::take(placement),
    })
}

fn decode_table_section(mut section: Value) -> Result<TableSection, String> {
    let Value::Object(ref mut placement) = section else {
        return Err("report table section requires an object".into());
    };
    placement.shift_remove("kind");
    let table = serde_json::from_value(placement.shift_remove("table").unwrap_or_default())
        .map_err(|error| format!("invalid report table: {error}"))?;
    let style = serde_json::from_value(placement.shift_remove("style").unwrap_or_default())
        .map_err(|error| format!("invalid report table style: {error}"))?;
    Ok(TableSection {
        table,
        style,
        placement: std::mem::take(placement),
    })
}
