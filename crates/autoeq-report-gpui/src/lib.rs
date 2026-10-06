//! Full-GPUI enhanced report viewer (WebGPU-gated).
//!
//! The 2D canvas renderer in `autoeq-report-wasm` is the default report path.
//! This crate hosts the GPUI viewer bundle the HTML shell loads when
//! `navigator.gpu` is present (plan decision D2): `mount_report` boots a
//! single-threaded GPUI app, moves its canvas into the shell container, and
//! renders an interactive explorer over the same versioned payload — tab
//! filter plus per-section series statistics (counts, ranges, means).
//!
//! The explorer model ([`ExplorerModel`]) is platform-independent and unit
//! tested; only the boot/view glue is wasm-gated.

pub use autoeq_report_wasm::schema;

use schema::{ReportPayload, Section};

// ---------------------------------------------------------------------------
// Explorer model (platform-independent)
// ---------------------------------------------------------------------------

/// Statistics for one line series.
#[derive(Debug, Clone, PartialEq)]
pub struct SeriesStat {
    pub name: String,
    /// Parsed CSS color as `[r, g, b, a]` floats, if the series names one.
    pub color: Option<[f32; 4]>,
    /// Number of x samples.
    pub points: usize,
    /// `(min, max, mean)` over finite y samples; `None` when empty.
    pub range: Option<(f64, f64, f64)>,
    /// Y axis selector (0 = primary, 1 = secondary).
    pub y_axis: u8,
}

/// One renderable payload section with precomputed headline details.
#[derive(Debug, Clone, PartialEq)]
pub struct SectionCard {
    pub tab: Option<String>,
    pub headline: String,
    pub details: Vec<String>,
    pub series: Vec<SeriesStat>,
}

/// Parse a `rgb(r, g, b)` / `rgba(r, g, b, a)` CSS color into floats.
pub fn parse_css_color(spec: &str) -> Option<[f32; 4]> {
    let spec = spec.trim();
    let (parts, alpha) = spec
        .strip_prefix("rgba(")
        .and_then(|inner| inner.strip_suffix(')'))
        .map(|inner| (inner, true))
        .or_else(|| {
            spec.strip_prefix("rgb(")
                .and_then(|inner| inner.strip_suffix(')'))
                .map(|inner| (inner, false))
        })?;
    let nums: Vec<f32> = parts
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    match (alpha, nums.as_slice()) {
        (true, [r, g, b, a]) => Some([*r / 255.0, *g / 255.0, *b / 255.0, *a]),
        (false, [r, g, b]) => Some([*r / 255.0, *g / 255.0, *b / 255.0, 1.0]),
        _ => None,
    }
}

fn series_stat(s: &schema::Series) -> SeriesStat {
    let finite: Vec<f64> =
        s.y.iter()
            .filter_map(|v| *v)
            .filter(|v| v.is_finite())
            .collect();
    let range = if finite.is_empty() {
        None
    } else {
        let min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        Some((min, max, mean))
    };
    SeriesStat {
        name: s.name.clone(),
        color: s.color.as_deref().and_then(parse_css_color),
        points: s.x.len(),
        range,
        y_axis: s.y_axis,
    }
}

/// Report explorer state: tab filter over section cards.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplorerModel {
    pub title: String,
    /// Tab labels in first-appearance order.
    pub tabs: Vec<String>,
    /// Active tab filter; `None` shows every section.
    pub active_tab: Option<String>,
    pub cards: Vec<SectionCard>,
}

impl ExplorerModel {
    pub fn from_report(report: &ReportPayload) -> Self {
        let mut tabs = Vec::new();
        let mut cards = Vec::new();
        for section in &report.sections {
            let (tab, headline, details, series) = match section {
                Section::Html { html: _, tab } => (
                    tab.clone(),
                    "Report block".to_string(),
                    vec!["tables and notes".to_string()],
                    vec![],
                ),
                Section::Figure { figure, tab } | Section::Grid { figure, tab, .. } => {
                    let stats: Vec<SeriesStat> = figure.series.iter().map(series_stat).collect();
                    let detail = format!("{} series", stats.len());
                    (tab.clone(), figure.title.clone(), vec![detail], stats)
                }
                Section::Bar { chart, tab } => {
                    let detail = format!("{} categories", chart.categories.len());
                    (tab.clone(), chart.title.clone(), vec![detail], vec![])
                }
                Section::Sankey { chart, tab } => {
                    let detail =
                        format!("{} nodes, {} links", chart.nodes.len(), chart.links.len());
                    (tab.clone(), chart.title.clone(), vec![detail], vec![])
                }
            };
            if let Some(name) = tab.as_ref()
                && !tabs.contains(name)
            {
                tabs.push(name.clone());
            }
            cards.push(SectionCard {
                tab,
                headline,
                details,
                series,
            });
        }
        ExplorerModel {
            title: report.title.clone(),
            tabs,
            active_tab: None,
            cards,
        }
    }

    /// Cards passing the active tab filter, in payload order.
    pub fn visible_cards(&self) -> impl Iterator<Item = &SectionCard> {
        self.cards
            .iter()
            .filter(|card| match (&self.active_tab, &card.tab) {
                (None, _) => true,
                (Some(active), Some(tab)) => active == tab,
                (Some(_), None) => false,
            })
    }

    pub fn set_active_tab(&mut self, tab: Option<String>) {
        if tab.as_ref().is_some_and(|name| !self.tabs.contains(name)) {
            return;
        }
        self.active_tab = tab;
    }
}

// ---------------------------------------------------------------------------
// GPUI view (wasm only)
// ---------------------------------------------------------------------------

#[cfg(target_family = "wasm")]
mod viewer {
    use super::*;
    use gpui::*;
    use wasm_bindgen::prelude::*;

    struct ReportView {
        model: ExplorerModel,
        scroll: ScrollHandle,
    }

    impl ReportView {
        fn tab_button(
            &self,
            cx: &mut Context<Self>,
            label: &str,
            active: bool,
            tab: Option<String>,
        ) -> impl IntoElement {
            let normal_bg = rgb(0xf0f4f8);
            let active_bg = rgb(0x4a90d9);
            div()
                .id(format!("gpui-tab-{label}"))
                .px_3()
                .py_1()
                .rounded_md()
                .border_1()
                .border_color(rgb(0xcccccc))
                .bg(if active { active_bg } else { normal_bg })
                .text_color(if active { rgb(0xffffff) } else { rgb(0x2a6496) })
                .text_sm()
                .child(label.to_string())
                .on_click(cx.listener(
                    move |view: &mut Self, _: &ClickEvent, _: &mut Window, cx| {
                        view.model.set_active_tab(tab.clone());
                        cx.notify();
                    },
                ))
        }
    }

    impl Render for ReportView {
        fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let tabs = self.model.tabs.clone();
            let active = self.model.active_tab.clone();
            let mut root = div()
                .size_full()
                .flex()
                .flex_col()
                .gap_3()
                .p_4()
                .bg(rgb(0xffffff))
                .text_color(rgb(0x1a1a1a));
            root = root.child(div().text_lg().child(self.model.title.clone()));
            let mut tab_row = div().flex().flex_row().flex_wrap().gap_2();
            {
                let all_active = active.is_none();
                tab_row = tab_row.child(self.tab_button(cx, "All", all_active, None));
            }
            for name in &tabs {
                let is_active = active.as_deref() == Some(name.as_str());
                tab_row = tab_row.child(self.tab_button(cx, name, is_active, Some(name.clone())));
            }
            root = root.child(tab_row);
            let mut list = div()
                .id("gpui-report-list")
                .flex_1()
                .flex()
                .flex_col()
                .gap_2()
                .overflow_y_scroll()
                .track_scroll(&self.scroll);
            for card in self.model.visible_cards() {
                let mut entry = div()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0xdddddd))
                    .bg(rgb(0xf8fafc))
                    .p_3()
                    .flex()
                    .flex_col()
                    .gap_1();
                let mut head = card.headline.clone();
                if let Some(tab) = card.tab.as_ref() {
                    head.push_str("  ·  ");
                    head.push_str(tab);
                }
                entry = entry.child(div().text_sm().child(head));
                for detail in &card.details {
                    entry = entry.child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x666666))
                            .child(detail.clone()),
                    );
                }
                for stat in &card.series {
                    let mut line = format!("{} — {} pts", stat.name, stat.points);
                    if let Some((min, max, mean)) = stat.range {
                        line.push_str(&format!("  [{min:.2}, {max:.2}] mean {mean:.2}"));
                    } else {
                        line.push_str("  (no finite samples)");
                    }
                    if stat.y_axis == 1 {
                        line.push_str("  [right axis]");
                    }
                    let mut row = div().flex().flex_row().items_center().gap_2().text_sm();
                    if let Some([r, g, b, _]) = stat.color {
                        let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
                        row = row.child(
                            div()
                                .w_3()
                                .h_3()
                                .rounded_sm()
                                .bg(rgb(byte(r) << 16 | byte(g) << 8 | byte(b))),
                        );
                    }
                    row = row.child(line);
                    entry = entry.child(row);
                }
                list = list.child(entry);
            }
            root = root.child(list);
            let _ = cx;
            root
        }
    }

    fn err_to_js(detail: String) -> JsValue {
        JsValue::from_str(&detail)
    }

    /// Read the embedded payload from the shell's script tag.
    fn read_payload() -> Result<ReportPayload, String> {
        let window = web_sys::window().ok_or_else(|| "no browser window".to_string())?;
        let document = window.document().ok_or_else(|| "no document".to_string())?;
        let tag = document
            .get_element_by_id("report-payload")
            .ok_or_else(|| "report-payload script tag missing".to_string())?;
        let text = tag
            .text_content()
            .ok_or_else(|| "report payload is empty".to_string())?;
        serde_json::from_str(&text).map_err(|e| format!("report payload is not valid JSON: {e}"))
    }

    /// Move the GPUI canvas (appended to `<body>` by the web platform) into
    /// the shell container so the 2D/GPUI toggle hides it with the container.
    /// The platform canvas is a direct `<body>` child; the 2D plot canvases
    /// nest inside the report divs, so parentage disambiguates them.
    fn relocate_canvas(container_id: &str) -> Result<(), String> {
        use wasm_bindgen::JsCast;
        let window = web_sys::window().ok_or_else(|| "no browser window".to_string())?;
        let document = window.document().ok_or_else(|| "no document".to_string())?;
        let container: web_sys::Element = document
            .get_element_by_id(container_id)
            .ok_or_else(|| format!("container #{container_id} missing"))?
            .dyn_into()
            .map_err(|_| format!("container #{container_id} is not an element"))?;
        let container_node: &web_sys::Node = container.unchecked_ref();
        let body = document.body().ok_or_else(|| "no body".to_string())?;
        let body_el: web_sys::Element =
            body.clone().dyn_into().map_err(|_| "no body".to_string())?;
        let canvases = document.get_elements_by_tag_name("canvas");
        for i in 0..canvases.length() {
            let Some(el) = canvases.item(i) else { continue };
            let node: &web_sys::Node = el.unchecked_ref();
            if container_node.contains(Some(node)) {
                return Ok(());
            }
            if el.parent_element().as_ref() == Some(&body_el) {
                container
                    .append_child(&el)
                    .map_err(|e| format!("canvas move failed: {e:?}"))?;
                el.set_attribute("id", "gpui-report-canvas").ok();
                return Ok(());
            }
        }
        Err("GPUI canvas not found under <body>".to_string())
    }

    /// Boot the viewer into the shell container. Idempotent: resolves
    /// immediately when the container already holds the canvas.
    #[wasm_bindgen]
    pub async fn mount_report(container_id: String) -> Result<(), JsValue> {
        console_error_panic_hook::set_once();
        let model = ExplorerModel::from_report(&read_payload().map_err(err_to_js)?);
        // Single-threaded: the report file may be opened without COOP/COEP
        // headers, where SharedArrayBuffer is unavailable.
        let platform = std::rc::Rc::new(gpui_web::WebPlatform::new(false));
        let handle =
            gpui::Application::with_platform(platform).run_embedded(move |cx: &mut gpui::App| {
                let bounds = gpui::Bounds::centered(None, gpui::size(gpui::px(1152.), gpui::px(800.)), cx);
                if let Err(error) = cx.open_window(
                    gpui::WindowOptions {
                        window_bounds: Some(gpui::WindowBounds::Windowed(bounds)),
                        ..Default::default()
                    },
                    |_, cx| {
                        cx.new(|_| ReportView { model, scroll: ScrollHandle::default() })
                    },
                ) {
                    web_sys::console::error_1(&JsValue::from_str(&format!(
                        "GPUI viewer could not open its window ({error:?}); showing 2D plots instead"
                    )));
                    return;
                }
                if let Err(detail) = relocate_canvas(&container_id) {
                    web_sys::console::error_1(&JsValue::from_str(&format!(
                        "GPUI viewer canvas placement failed ({detail}); showing 2D plots instead"
                    )));
                }
                cx.activate(true);
            });
        // Page-lifetime handle: the viewer lives as long as the report page
        // (same pattern as gpui-hello-web); the shell hides it via the
        // container when the 2D view is selected.
        std::mem::forget(handle);
        Ok(())
    }
}

#[cfg(target_family = "wasm")]
pub use viewer::mount_report;

// ---------------------------------------------------------------------------
// Native unit tests (model only; the wasm view needs a browser)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use schema::{AxisSpec, Figure, Section, XScale};

    fn figure_section(title: &str, tab: Option<&str>) -> Section {
        Section::Figure {
            figure: Figure {
                title: title.to_string(),
                x: AxisSpec {
                    label: "x".to_string(),
                    scale: XScale::Linear,
                    min: None,
                    max: None,
                },
                y: AxisSpec {
                    label: "y".to_string(),
                    scale: XScale::Linear,
                    min: None,
                    max: None,
                },
                y2: None,
                series: vec![schema::Series {
                    name: "s".to_string(),
                    x: vec![1.0, 2.0, 3.0],
                    y: vec![Some(1.0), None, Some(3.0)],
                    color: Some("rgba(255, 0, 0, 0.5)".to_string()),
                    width: 2.0,
                    dash: schema::DashOption::Solid,
                    visible: true,
                    y_axis: 0,
                }],
                hlines: vec![],
                vlines: vec![],
                xranges: vec![],
                annotations: vec![],
                legend: true,
            },
            tab: tab.map(str::to_string),
        }
    }

    fn sample_report() -> ReportPayload {
        ReportPayload {
            schema: schema::SCHEMA_VERSION.to_string(),
            title: "Demo".to_string(),
            sections: vec![
                Section::Html {
                    html: "<p>hi</p>".to_string(),
                    tab: None,
                },
                figure_section("A", Some("Left")),
                figure_section("B", Some("Right")),
                figure_section("C", None),
            ],
            provenance: None,
        }
    }

    #[test]
    fn tabs_follow_first_appearance_order() {
        let model = ExplorerModel::from_report(&sample_report());
        assert_eq!(model.tabs, vec!["Left".to_string(), "Right".to_string()]);
        assert_eq!(model.active_tab, None);
        assert_eq!(model.cards.len(), 4);
    }

    #[test]
    fn filter_shows_matching_tab_only() {
        let mut model = ExplorerModel::from_report(&sample_report());
        model.set_active_tab(Some("Left".to_string()));
        let heads: Vec<&str> = model.visible_cards().map(|c| c.headline.as_str()).collect();
        assert_eq!(heads, vec!["A"]);
        // Unknown tabs are ignored, not applied.
        model.set_active_tab(Some("Nope".to_string()));
        assert_eq!(model.active_tab, Some("Left".to_string()));
        model.set_active_tab(None);
        assert_eq!(model.visible_cards().count(), 4);
    }

    #[test]
    fn series_stats_skip_gaps() {
        let model = ExplorerModel::from_report(&sample_report());
        let stat = &model.cards[1].series[0];
        assert_eq!(stat.points, 3);
        assert_eq!(stat.range, Some((1.0, 3.0, 2.0)));
        assert_eq!(stat.color, Some([1.0, 0.0, 0.0, 0.5]));
    }

    #[test]
    fn css_color_parsing() {
        assert_eq!(
            parse_css_color("rgba(255, 0, 0, 0.5)"),
            Some([1.0, 0.0, 0.0, 0.5])
        );
        assert_eq!(
            parse_css_color("rgb(0, 128, 255)"),
            Some([0.0, 128.0 / 255.0, 1.0, 1.0])
        );
        assert_eq!(parse_css_color("#ff0000"), None);
        assert_eq!(parse_css_color("red"), None);
    }
}
