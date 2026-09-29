//! Versioned report payload schema (`autoeq-report-data-v1`).
//!
//! Both report producers (the Rust `autoeq-plot` emitter and the Python
//! `scripts/src/wasm_report.py` emitter) build this exact document; the WASM
//! renderers draw from it and never recompute curves.

use serde::{Deserialize, Serialize};

/// Schema discriminator embedded in every payload.
pub const SCHEMA_VERSION: &str = "autoeq-report-data-v1";

/// Top-level payload embedded in the HTML shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportPayload {
    /// Must equal [`SCHEMA_VERSION`].
    pub schema: String,
    /// Page `<title>` and header text.
    pub title: String,
    /// Ordered page content.
    pub sections: Vec<Section>,
}

impl ReportPayload {
    /// New empty payload with the current schema discriminator.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA_VERSION.to_string(),
            title: title.into(),
            sections: Vec::new(),
        }
    }
}

/// One block of page content.
///
/// The optional `tab` groups sections under a named report tab (the shell
/// renders a tab bar when any section carries one). It is purely a shell
/// concern: renderers ignore it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Section {
    /// Sampled heatmap or projected surface with shared figure controls.
    Grid {
        figure: Figure,
        grid: GridData,
        #[serde(default)]
        tab: Option<String>,
    },
    /// Raw HTML (tables, summaries, filter lists — already renderer-free).
    Html {
        html: String,
        #[serde(default)]
        tab: Option<String>,
    },
    /// Cartesian line chart (log or linear x; y always linear).
    Figure {
        figure: Figure,
        #[serde(default)]
        tab: Option<String>,
    },
    /// Grouped categorical bar chart.
    Bar {
        chart: BarChart,
        #[serde(default)]
        tab: Option<String>,
    },
    /// Flow (Sankey) diagram, e.g. bass-management routing.
    Sankey {
        chart: SankeyChart,
        #[serde(default)]
        tab: Option<String>,
    },
}

/// Renderer-independent time-frequency grid; rows follow the y/time axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridData {
    /// Positive frequency coordinates in Hz.
    pub x: Vec<f64>,
    /// Time coordinates in milliseconds.
    pub y: Vec<f64>,
    /// Levels indexed as `[time][frequency]`.
    pub z: Vec<Vec<f64>>,
    /// Surface when true, otherwise a heatmap.
    pub surface: bool,
    /// Display floor in dB; values are clipped, not renormalized.
    pub zmin: f64,
    /// Display ceiling in dB.
    pub zmax: f64,
    /// Frequency indexes for highlighted decays, matching figure series order.
    #[serde(default)]
    pub highlights: Vec<usize>,
    /// Optional surface elevation and azimuth in degrees; display-only camera state.
    #[serde(default)]
    pub rotation: Option<[f64; 2]>,
}

/// X-axis scale for a [`Figure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XScale {
    Log,
    Linear,
}

/// One axis specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisSpec {
    /// Axis title, e.g. `"Frequency (Hz)"`.
    #[serde(default)]
    pub label: String,
    /// X-only scale selector (ignored on y, always linear).
    #[serde(default = "default_x_scale")]
    pub scale: XScale,
    /// Explicit domain; defaults to the data extent with padding.
    #[serde(default)]
    pub min: Option<f64>,
    /// Explicit domain; defaults to the data extent with padding.
    #[serde(default)]
    pub max: Option<f64>,
}

fn default_x_scale() -> XScale {
    XScale::Linear
}

/// One named polyline series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Series {
    /// Legend/display name.
    pub name: String,
    /// X samples (same length as `y`).
    pub x: Vec<f64>,
    /// Y samples; `None` is a gap (line breaks, like plotly `connectgaps=false`).
    pub y: Vec<Option<f64>>,
    /// CSS color; defaults to the `d3rs` category10 slot for the index.
    #[serde(default)]
    pub color: Option<String>,
    /// Line width in CSS px.
    #[serde(default = "default_line_width")]
    pub width: f32,
    /// Dash pattern.
    #[serde(default)]
    pub dash: DashOption,
    /// Initial visibility (legend clicks toggle it client-side).
    #[serde(default = "default_true")]
    pub visible: bool,
    /// Y axis selector: 0 = primary (left), 1 = secondary (right, `Figure::y2`).
    /// Values other than 1 read as primary.
    #[serde(default)]
    pub y_axis: u8,
}

fn default_line_width() -> f32 {
    2.0
}

fn default_true() -> bool {
    true
}

/// Optional dash override (absent = solid).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DashOption {
    #[default]
    Solid,
    Dash,
    Dot,
    // Canonical wire form is the CSS-style `dashdot` (see SCHEMA.md); keep
    // accepting the snake_case form for payloads already in the wild.
    #[serde(rename = "dashdot", alias = "dash_dot")]
    DashDot,
}

impl DashOption {
    /// Canvas2D dash array for the pattern.
    pub fn pattern(self) -> &'static [f64] {
        match self {
            DashOption::Solid => &[],
            DashOption::Dash => &[8.0, 5.0],
            DashOption::Dot => &[2.0, 4.0],
            DashOption::DashDot => &[8.0, 4.0, 2.0, 4.0],
        }
    }
}

/// Horizontal or vertical reference line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineMark {
    /// Position in data units.
    pub at: f64,
    /// CSS color.
    pub color: String,
    /// Dash pattern.
    #[serde(default)]
    pub dash: DashOption,
    /// Line width in CSS px.
    #[serde(default = "default_mark_width")]
    pub width: f32,
    /// Optional end label (drawn inside the plot area).
    #[serde(default)]
    pub label: Option<String>,
}

fn default_mark_width() -> f32 {
    1.0
}

/// Shaded x-range (e.g. out-of-optimization-bounds highlight).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeMark {
    /// Range start in data units.
    pub x0: f64,
    /// Range end in data units.
    pub x1: f64,
    /// CSS fill color (usually translucent).
    pub color: String,
}

/// Text annotation pinned to data coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// X in data units.
    pub x: f64,
    /// Y in data units.
    pub y: f64,
    /// Label text.
    pub text: String,
}

/// Cartesian line chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Figure {
    /// Figure title (also used for the canvas accessibility label).
    #[serde(default)]
    pub title: String,
    /// X axis.
    pub x: AxisSpec,
    /// Y axis (always linear).
    pub y: AxisSpec,
    /// Optional secondary y axis (right side, always linear). Series with
    /// `y_axis == 1` map onto it; without it they read as primary.
    #[serde(default)]
    pub y2: Option<AxisSpec>,
    /// Named series in draw order.
    #[serde(default)]
    pub series: Vec<Series>,
    /// Horizontal reference lines (data y).
    #[serde(default)]
    pub hlines: Vec<LineMark>,
    /// Vertical reference lines (data x).
    #[serde(default)]
    pub vlines: Vec<LineMark>,
    /// Shaded x-ranges drawn under the series.
    #[serde(default)]
    pub xranges: Vec<RangeMark>,
    /// Text annotations.
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    /// Show the clickable legend (toggles series visibility).
    #[serde(default = "default_true")]
    pub legend: bool,
}

/// One group of bars sharing a legend entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarGroup {
    /// Legend/display name.
    pub name: String,
    /// One value per category.
    pub values: Vec<f64>,
    /// CSS color; defaults to the `d3rs` category10 slot for the index.
    #[serde(default)]
    pub color: Option<String>,
}

/// Grouped categorical bar chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarChart {
    /// Figure title.
    #[serde(default)]
    pub title: String,
    /// Category labels on the x axis.
    #[serde(default)]
    pub categories: Vec<String>,
    /// Bar groups in draw order.
    #[serde(default)]
    pub groups: Vec<BarGroup>,
    /// Y axis label.
    #[serde(default)]
    pub ylabel: String,
    /// Horizontal reference lines (data y), e.g. headroom limits.
    #[serde(default)]
    pub hlines: Vec<LineMark>,
    /// Explicit y domain; defaults to data extent (including zero).
    #[serde(default)]
    pub ymin: Option<f64>,
    /// Explicit y domain; defaults to data extent.
    #[serde(default)]
    pub ymax: Option<f64>,
    /// Show the clickable legend.
    #[serde(default = "default_true")]
    pub legend: bool,
}

/// One directed flow between node indexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SankeyLink {
    /// Source node index into [`SankeyChart::nodes`].
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Flow magnitude.
    pub value: f64,
    /// CSS link color; defaults to a translucent route color.
    #[serde(default)]
    pub color: Option<String>,
}

/// Flow (Sankey) diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SankeyChart {
    /// Figure title.
    #[serde(default)]
    pub title: String,
    /// Node labels in index order.
    #[serde(default)]
    pub nodes: Vec<String>,
    /// Flows between nodes.
    #[serde(default)]
    pub links: Vec<SankeyLink>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_round_trip() {
        let mut payload = ReportPayload::new("t");
        payload.sections.push(Section::Html {
            html: "<p>hi</p>".to_string(),
            tab: None,
        });
        payload.sections.push(Section::Figure {
            figure: Figure {
                title: "f".to_string(),
                x: AxisSpec {
                    label: "Frequency (Hz)".to_string(),
                    scale: XScale::Log,
                    min: Some(20.0),
                    max: Some(20000.0),
                },
                y: AxisSpec {
                    label: "SPL (dB)".to_string(),
                    scale: XScale::Linear,
                    min: None,
                    max: None,
                },
                y2: None,
                series: vec![Series {
                    name: "Input".to_string(),
                    x: vec![20.0, 20000.0],
                    y: vec![Some(0.0), Some(1.0)],
                    color: None,
                    width: 2.0,
                    dash: DashOption::Dash,
                    visible: true,
                    y_axis: 0,
                }],
                hlines: vec![],
                vlines: vec![],
                xranges: vec![],
                annotations: vec![],
                legend: true,
            },
            tab: None,
        });
        let json = serde_json::to_string(&payload).expect("serialize");
        let back: ReportPayload = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.schema, SCHEMA_VERSION);
        assert_eq!(back.sections.len(), 2);
    }

    #[test]
    fn dash_patterns_are_sane() {
        assert!(DashOption::Solid.pattern().is_empty());
        assert!(!DashOption::Dash.pattern().is_empty());
    }

    #[test]
    fn dashdot_wire_form_matches_emitters() {
        // Emitters (Python/JS/CSS) write `dashdot`; accept the legacy
        // snake_case form too.
        let json = serde_json::to_string(&DashOption::DashDot).expect("serialize");
        assert_eq!(json, "\"dashdot\"");
        assert_eq!(
            serde_json::from_str::<DashOption>("\"dashdot\"").expect("deserialize"),
            DashOption::DashDot
        );
        assert_eq!(
            serde_json::from_str::<DashOption>("\"dash_dot\"").expect("deserialize"),
            DashOption::DashDot
        );
    }
}
