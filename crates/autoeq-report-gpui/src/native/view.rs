//! Native GPUI view of an assembled AutoEQ report.

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

use autoeq_report_wasm::{
    draw::{DrawMeta, PlotGeometry},
    grid::DEFAULT_SURFACE_ROTATION,
    report_document::{
        ARRIVAL_PLOT_NOTE, DiagnosticBlock, EPA_CAPTION, EPA_COLUMNS, EPA_TITLE, EpaDelta,
        EpaScoreRow, NoticeTone, RESONANCE_NOTE, RESONANCE_TITLE, RESULTS_COLUMNS, RESULTS_NOTE,
        RESULTS_TITLE, ReportDocument, ReportSection, ResonanceEvidence, ResonanceRow, ResultsRow,
        SignalFlowStage, verification_status_detail,
    },
    report_table::ReportTable,
    schema::{Figure, Section, XScale},
    viewer_smoothing::ViewerSmoothing,
};
use gpui::{
    AnyElement, AppContext, Bounds, ClickEvent, Context, Div, FocusHandle, InteractiveElement,
    IntoElement, KeyDownEvent, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    ParentElement, Pixels, Point, Render, ScrollHandle, ScrollWheelEvent,
    StatefulInteractiveElement, Styled, Window, canvas, div, px, relative, rgb, rgba,
};

use gpui::prelude::FluentBuilder;
use gpui_ui_kit::{Button, ButtonSize, ButtonTheme, ButtonVariant, ThemeExt};

use super::paint_chart;

/// Displays an immutable report with independent navigation and chart controls.
pub struct ReportView {
    document: Arc<ReportDocument>,
    sections: Vec<DisplaySection>,
    dashboards: Vec<String>,
    dashboard: usize,
    dark_appearance: Option<bool>,
    focus: Option<FocusHandle>,
    scroll: ScrollHandle,
    charts: HashMap<ChartKey, ChartView>,
    filter_origins: HashMap<usize, usize>,
    open_filters: HashSet<(usize, usize, usize, usize)>,
    open_stages: HashSet<(usize, usize, usize)>,
    open_diagnostics: HashSet<ChartKey>,
    flow_scope: usize,
}

/// Report data prepared for a native window without touching GPUI state.
pub struct NativeReport {
    document: Arc<ReportDocument>,
    sections: Vec<DisplaySection>,
    dashboards: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ChartKey(usize, usize);

impl std::fmt::Display for ChartKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.0, self.1)
    }
}

struct DisplaySection {
    index: usize,
    dashboard: usize,
    heading: Option<String>,
    channel: Option<String>,
    content: Content,
}

enum Content {
    Table,
    Text,
    Notices,
    Filters,
    Chart(Arc<Section>),
    Diagnostics(Vec<Option<Arc<Section>>>),
    BrowserContent,
}

fn is_chart(kind: Option<&str>) -> bool {
    matches!(kind, Some("figure" | "grid" | "bar" | "sankey"))
}

impl NativeReport {
    /// Prepares the document's charts and dashboard navigation.
    pub fn new(document: Arc<ReportDocument>) -> Self {
        let mut dashboards = Vec::<String>::new();
        let mut sections = Vec::new();
        for (index, section) in document.sections.iter().enumerate() {
            let (bucket, heading, channel, content) = match section {
                ReportSection::Table(table) => {
                    (table.bucket(), table.subhead(), table.tab(), Content::Table)
                }
                ReportSection::Tables(group) => {
                    (group.bucket(), group.subhead(), group.tab(), Content::Table)
                }
                ReportSection::Text(text) => {
                    (text.bucket(), text.subhead(), text.tab(), Content::Text)
                }
                ReportSection::Diagnostics(section) => {
                    let charts = section
                        .blocks
                        .iter()
                        .map(|block| match block {
                            DiagnosticBlock::Chart(value)
                            | DiagnosticBlock::AcceptanceChart(value)
                            | DiagnosticBlock::CaptureChart(value)
                            | DiagnosticBlock::CaptureArrival {
                                chart: Some(value), ..
                            } => {
                                let kind = value["kind"].as_str();
                                if !is_chart(kind) {
                                    return Err(());
                                }
                                serde_json::from_value(value.clone())
                                    .map(|chart| Some(Arc::new(chart)))
                                    .map_err(|_| ())
                            }
                            _ => Ok(None),
                        })
                        .collect::<Result<Vec<_>, _>>();
                    (
                        section.bucket(),
                        section.subhead(),
                        section.tab(),
                        charts
                            .map(Content::Diagnostics)
                            .unwrap_or(Content::BrowserContent),
                    )
                }
                ReportSection::Filters(section) => (
                    section.bucket(),
                    section.subhead(),
                    section.tab(),
                    Content::Filters,
                ),
                ReportSection::PlaybackNotices(_) => (None, None, None, Content::Notices),
                ReportSection::CaptureQuality(_) => (
                    Some("DSP analysis"),
                    Some("Capture quality"),
                    None,
                    Content::Table,
                ),
                ReportSection::Legacy(value) => {
                    let content = if is_chart(value["kind"].as_str()) {
                        serde_json::from_value(value.clone())
                            .map(|section| Content::Chart(Arc::new(section)))
                            .unwrap_or(Content::BrowserContent)
                    } else {
                        Content::BrowserContent
                    };
                    (
                        value["bucket"].as_str(),
                        value["subhead"].as_str(),
                        value["tab"].as_str(),
                        content,
                    )
                }
            };
            let bucket = bucket.unwrap_or("Report status");
            let dashboard = dashboards
                .iter()
                .position(|name| name == bucket)
                .unwrap_or_else(|| {
                    dashboards.push(bucket.into());
                    dashboards.len() - 1
                });
            sections.push(DisplaySection {
                index,
                dashboard,
                heading: heading.map(str::to_owned),
                channel: channel.map(str::to_owned),
                content,
            });
        }
        Self {
            document,
            sections,
            dashboards,
        }
    }

    /// Prepares a native report when every section can be displayed.
    ///
    /// Legacy HTML takes the browser route before chart data is decoded.
    pub fn try_complete(document: Arc<ReportDocument>) -> Option<Self> {
        if document.sections.iter().any(|section| {
            matches!(section, ReportSection::Legacy(value) if !is_chart(value["kind"].as_str()))
        }) {
            return None;
        }
        let report = Self::new(document);
        report.is_complete().then_some(report)
    }

    /// Returns whether all report sections have native renderers.
    pub fn is_complete(&self) -> bool {
        !self.sections.is_empty()
            && self
                .sections
                .iter()
                .all(|section| !matches!(section.content, Content::BrowserContent))
    }

    /// Returns the source document for standalone export.
    pub fn document(&self) -> &Arc<ReportDocument> {
        &self.document
    }
}

impl ReportView {
    /// Builds the native view from a composed report.
    pub fn new(document: Arc<ReportDocument>) -> Self {
        Self::from_prepared(NativeReport::new(document))
    }

    /// Mounts a report whose chart data is ready for display.
    pub fn from_prepared(report: NativeReport) -> Self {
        let NativeReport {
            document,
            sections,
            dashboards,
        } = report;
        let dashboard = sections
            .iter()
            .find(|section| {
                matches!(
                    section.content,
                    Content::Table | Content::Chart(_) | Content::Diagnostics(_)
                )
            })
            .or_else(|| {
                sections
                    .iter()
                    .find(|section| matches!(section.content, Content::Text))
            })
            .map_or(0, |section| section.dashboard);
        Self {
            document,
            sections,
            dashboards,
            dashboard,
            dark_appearance: None,
            focus: None,
            scroll: ScrollHandle::new(),
            charts: HashMap::new(),
            filter_origins: HashMap::new(),
            open_filters: HashSet::new(),
            open_stages: HashSet::new(),
            open_diagnostics: HashSet::new(),
            flow_scope: 0,
        }
    }

    /// Uses the host application's appearance for the report chrome.
    ///
    /// Without an override, the report follows its window appearance.
    #[must_use]
    pub fn with_dark_appearance(mut self, dark: bool) -> Self {
        self.dark_appearance = Some(dark);
        self
    }

    fn render_filters(
        &self,
        index: usize,
        listing: &autoeq_report_wasm::eq_sections::EqFilterListing,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let mut body = div()
            .flex()
            .flex_col()
            .gap_2()
            .child(div().text_lg().child(listing.title.clone()));
        if let Some(note) = &listing.note {
            body = body.child(div().text_sm().child(note.clone()));
        }
        let active = self.filter_origins.get(&index).copied().unwrap_or(0);
        if listing.origins.len() > 1 {
            let mut controls = div().flex().flex_wrap().gap_2();
            for (origin, listing) in listing.origins.iter().enumerate() {
                controls = controls.child(
                    button(
                        cx,
                        format!("report-filters-{index}-{origin}"),
                        listing.label.clone(),
                        active == origin,
                    )
                    .on_click(cx.listener(
                        move |view, _: &ClickEvent, _, cx| {
                            view.filter_origins.insert(index, origin);
                            cx.notify();
                        },
                    )),
                );
            }
            body = body.child(controls);
        }
        let Some(origin) = listing.origins.get(active) else {
            return body.into_any_element();
        };
        if listing.origins.len() == 1 && !origin.label.is_empty() {
            body = body.child(div().text_sm().child(origin.label.clone()));
        }
        for (pass_index, pass) in origin.passes.iter().enumerate() {
            if let Some((name, _)) = &pass.heading {
                body = body.child(
                    div()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(name.clone()),
                );
            }
            for (filter_index, filter) in pass.filters.iter().enumerate() {
                if filter.details.is_empty() {
                    body = body.child(div().text_sm().child(filter.summary.clone()));
                    continue;
                }
                let key = (index, active, pass_index, filter_index);
                let open = self.open_filters.contains(&key);
                body = body.child(
                    button(
                        cx,
                        format!(
                            "report-filter-details-{index}-{active}-{pass_index}-{filter_index}"
                        ),
                        filter.summary.clone(),
                        open,
                    )
                    .on_click(cx.listener(
                        move |view, _: &ClickEvent, _, cx| {
                            if !view.open_filters.insert(key) {
                                view.open_filters.remove(&key);
                            }
                            cx.notify();
                        },
                    )),
                );
                if open {
                    body = body.child(
                        div().pl_3().text_sm().children(
                            filter
                                .details
                                .iter()
                                .map(|detail| div().child(detail.clone())),
                        ),
                    );
                }
            }
        }
        body.into_any_element()
    }

    fn toggle(&mut self, index: ChartKey, series: usize) -> bool {
        let Some(chart) = self.charts.get_mut(&index) else {
            return false;
        };
        match Arc::make_mut(&mut chart.section) {
            Section::Figure { figure, .. } | Section::Grid { figure, .. } => {
                let Some(series) = figure.series.get_mut(series) else {
                    return false;
                };
                series.visible = !series.visible;
                true
            }
            Section::Bar { .. } => {
                let Some(visible) = Arc::make_mut(&mut chart.bars).get_mut(series) else {
                    return false;
                };
                *visible = !*visible;
                true
            }
            Section::Sankey { .. } | Section::Html { .. } => false,
        }
    }

    fn render_chart(
        &mut self,
        index: ChartKey,
        original: Arc<Section>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let chart = self
            .charts
            .entry(index)
            .or_insert_with(|| ChartView::new(original));
        let section = chart.section.clone();
        let magnification = chart.magnification;
        let bars = chart.bars.clone();
        let paint_geometry = chart.geometry.clone();
        let hit_geometry = chart.geometry.clone();
        let selection = match chart.drag {
            Some(ChartDrag::Selection(selection)) => chart.selection_overlay(selection),
            _ => None,
        };
        let has_viewport = matches!(&*section, Section::Figure { .. } | Section::Grid { .. });
        let surface_chart = matches!(&*section, Section::Grid { grid, .. } if grid.surface);
        let mut controls = div().flex().flex_wrap().gap_2();
        for (series, (name, visible)) in chart.legend().into_iter().enumerate() {
            controls = controls.child(
                button(cx, format!("report-series-{index}-{series}"), name, visible).on_click(
                    cx.listener(move |view, _: &ClickEvent, _, cx| {
                        if view.toggle(index, series) {
                            cx.notify();
                        }
                    }),
                ),
            );
        }
        if has_viewport {
            if !surface_chart {
                controls =
                    controls.child(chart_button(index, "Move", chart.move_mode, cx, |chart| {
                        chart.move_mode = !chart.move_mode;
                    }));
            }
            for (label, factor) in [("Zoom in", 1.0 / 1.5), ("Zoom out", 1.5)] {
                controls = controls.child(chart_button(index, label, false, cx, move |chart| {
                    chart.zoom(factor, None);
                }));
            }
        } else {
            for (label, factor) in [("Zoom in", 1.25), ("Zoom out", 1.0 / 1.25)] {
                controls = controls.child(chart_button(index, label, false, cx, move |chart| {
                    chart.magnification = (chart.magnification * factor).clamp(0.5, 4.0);
                }));
            }
        }
        let frequency_presets = match &*section {
            Section::Figure { figure, .. } => {
                matches!(figure.x.scale, XScale::Log)
                    && figure.x.label == "Frequency (Hz)"
                    && figure.y.label == "SPL (dB)"
            }
            Section::Grid { figure, .. } => figure.x.label == "Frequency (Hz)",
            _ => false,
        };
        if frequency_presets {
            for (label, range) in [
                ("20–200 Hz", [20.0, 200.0]),
                ("20–20k Hz", [20.0, 20_000.0]),
            ] {
                controls = controls.child(chart_button(index, label, false, cx, move |chart| {
                    chart.frequency_range(range);
                }));
            }
            if matches!(&*section, Section::Figure { .. }) {
                for (label, span_db) in [("20 dB span", 20.0), ("50 dB span", 50.0)] {
                    controls =
                        controls.child(chart_button(index, label, false, cx, move |chart| {
                            chart.db_span(span_db);
                        }));
                }
            }
        }
        if matches!(&*section, Section::Figure { figure, .. } if matches!(figure.x.scale, XScale::Log))
        {
            let (_, _, label) = ViewerSmoothing::OPTIONS
                .iter()
                .find(|(mode, _, _)| *mode == chart.smoothing)
                .copied()
                .unwrap_or(ViewerSmoothing::OPTIONS[0]);
            controls = controls.child(chart_menu_button(
                index,
                "Smoothing",
                label,
                ChartMenu::Smoothing,
                chart.menu,
                cx,
            ));
            if chart.menu == Some(ChartMenu::Smoothing) {
                for (mode, _, label) in ViewerSmoothing::OPTIONS {
                    controls = controls.child(chart_button(
                        index,
                        label,
                        chart.smoothing == mode,
                        cx,
                        move |chart| {
                            chart.set_smoothing(mode);
                        },
                    ));
                }
            }
        }
        if let Section::Grid { grid, .. } = &*section
            && grid.surface
        {
            let (_, label) = SURFACE_COLORMAPS
                .iter()
                .find(|(id, _)| *id == grid.colormap)
                .copied()
                .unwrap_or(SURFACE_COLORMAPS[0]);
            controls = controls.child(chart_menu_button(
                index,
                "Colormap",
                label,
                ChartMenu::Colormap,
                chart.menu,
                cx,
            ));
            if chart.menu == Some(ChartMenu::Colormap) {
                for (id, label) in SURFACE_COLORMAPS {
                    controls = controls.child(chart_button(
                        index,
                        label,
                        grid.colormap == id,
                        cx,
                        move |chart| {
                            chart.surface(SurfaceAction::Colormap(id));
                        },
                    ));
                }
            }
            for (label, selected, action) in [
                ("Surface fill", grid.show_surface, SurfaceAction::Fill),
                ("Contours", grid.show_contours, SurfaceAction::Contours),
                ("Rotate left", false, SurfaceAction::Turn(-15.0)),
                ("Rotate right", false, SurfaceAction::Turn(15.0)),
                ("Tilt up", false, SurfaceAction::Tilt(15.0)),
                ("Tilt down", false, SurfaceAction::Tilt(-15.0)),
            ] {
                controls = controls.child(chart_button(index, label, selected, cx, move |chart| {
                    chart.surface(action)
                }));
            }
        }
        controls = controls.child(chart_button(
            index,
            "Reset view",
            false,
            cx,
            ChartView::reset,
        ));
        div()
            .flex()
            .flex_col()
            .gap_2()
            .child(controls)
            .child(
                div()
                    .id(format!("report-chart-{index}"))
                    .h(px(340.0))
                    .overflow_hidden()
                    .when(!has_viewport, |element| element.overflow_scroll())
                    .relative()
                    .when(has_viewport, |mut element| {
                        for mouse_button in [MouseButton::Right, MouseButton::Middle] {
                            element = element
                                .on_mouse_down(
                                    mouse_button,
                                    cx.listener(move |view, event: &MouseDownEvent, _, _| {
                                        if let Some(chart) = view.charts.get_mut(&index) {
                                            chart.begin_drag(event.position, true);
                                        }
                                    }),
                                )
                                .on_mouse_up(
                                    mouse_button,
                                    cx.listener(move |view, event: &MouseUpEvent, _, cx| {
                                        if let Some(chart) = view.charts.get_mut(&index)
                                            && chart.drag.is_some()
                                        {
                                            chart.finish_drag(event.position);
                                            cx.notify();
                                        }
                                    }),
                                )
                                .on_mouse_up_out(
                                    mouse_button,
                                    cx.listener(move |view, event: &MouseUpEvent, _, cx| {
                                        if let Some(chart) = view.charts.get_mut(&index)
                                            && chart.drag.is_some()
                                        {
                                            chart.finish_drag(event.position);
                                            cx.notify();
                                        }
                                    }),
                                );
                        }
                        element
                            .on_scroll_wheel(cx.listener(
                                move |view, event: &ScrollWheelEvent, _, cx| {
                                    let delta = event.delta.pixel_delta(px(1.0)).y;
                                    if delta == px(0.0) {
                                        return;
                                    }
                                    let factor = if delta > px(0.0) { 1.0 / 1.2 } else { 1.2 };
                                    if let Some(chart) = view.charts.get_mut(&index)
                                        && chart.zoom(factor, Some(event.position))
                                    {
                                        cx.stop_propagation();
                                        cx.notify();
                                    }
                                },
                            ))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |view, event: &MouseDownEvent, _, cx| {
                                    if let Some(chart) = view.charts.get_mut(&index) {
                                        if event.click_count == 2 && !surface_chart {
                                            chart.reset();
                                            cx.notify();
                                        } else {
                                            chart.begin_drag(event.position, event.modifiers.shift);
                                        }
                                    }
                                }),
                            )
                            .on_mouse_move(cx.listener(
                                move |view, event: &MouseMoveEvent, _, cx| {
                                    if matches!(
                                        event.pressed_button,
                                        Some(
                                            MouseButton::Left
                                                | MouseButton::Right
                                                | MouseButton::Middle
                                        )
                                    ) && let Some(chart) = view.charts.get_mut(&index)
                                        && chart.update_drag(event.position)
                                    {
                                        cx.notify();
                                    }
                                },
                            ))
                            .on_mouse_up_out(
                                MouseButton::Left,
                                cx.listener(move |view, event: &MouseUpEvent, _, cx| {
                                    if let Some(chart) = view.charts.get_mut(&index)
                                        && chart.drag.is_some()
                                    {
                                        chart.finish_drag(event.position);
                                        cx.notify();
                                    }
                                }),
                            )
                    })
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(move |view, event: &gpui::MouseUpEvent, _, cx| {
                            if let Some(chart) = view.charts.get_mut(&index)
                                && chart.drag.is_some()
                            {
                                let applied = chart.finish_drag(event.position);
                                cx.notify();
                                if applied {
                                    return;
                                }
                            }
                            let hit = hit_geometry.borrow().as_ref().and_then(|(bounds, meta)| {
                                legend_at(*bounds, meta, event.position)
                            });
                            if let Some(series) = hit
                                && view.toggle(index, series)
                            {
                                cx.notify();
                            }
                        }),
                    )
                    .child(
                        canvas(
                            |_, _, _| (),
                            move |bounds, (), window, cx| {
                                let meta = paint_chart(&section, bounds, &bars, window, cx);
                                paint_geometry.replace(meta.map(|meta| (bounds, meta)));
                            },
                        )
                        .flex_none()
                        .w(relative(magnification))
                        .h(px(340.0 * magnification)),
                    )
                    .when_some(selection, |element, overlay| {
                        element.child(
                            div()
                                .absolute()
                                .left(overlay.origin.x)
                                .top(overlay.origin.y)
                                .w(overlay.size.width)
                                .h(overlay.size.height)
                                .bg(rgba(0x285c9e22))
                                .border_1()
                                .border_color(rgb(0x285c9e)),
                        )
                    }),
            )
            .into_any_element()
    }
}

#[derive(Clone, Copy)]
struct PlotSelection {
    start: Point<Pixels>,
    end: Point<Pixels>,
}

#[derive(Clone, Copy)]
enum ChartDrag {
    Selection(PlotSelection),
    Pan {
        start: Point<Pixels>,
        plot: PlotGeometry,
        secondary_y: Option<[f64; 2]>,
        bounds: Bounds<Pixels>,
    },
    Rotate {
        start: Point<Pixels>,
        rotation: [f64; 2],
        bounds: Bounds<Pixels>,
    },
}

struct ChartView {
    original: Arc<Section>,
    section: Arc<Section>,
    bars: Arc<Vec<bool>>,
    geometry: Rc<RefCell<Option<(Bounds<Pixels>, DrawMeta)>>>,
    drag: Option<ChartDrag>,
    move_mode: bool,
    smoothing: ViewerSmoothing,
    menu: Option<ChartMenu>,
    magnification: f32,
}

impl ChartView {
    fn new(section: Arc<Section>) -> Self {
        let bars = match &*section {
            Section::Bar { chart, .. } => vec![true; chart.groups.len()],
            _ => Vec::new(),
        };
        Self {
            original: section.clone(),
            section,
            bars: Arc::new(bars),
            geometry: Rc::default(),
            drag: None,
            move_mode: false,
            smoothing: ViewerSmoothing::None,
            menu: None,
            magnification: 1.0,
        }
    }

    fn set_smoothing(&mut self, mode: ViewerSmoothing) {
        if let (
            Section::Figure {
                figure: original, ..
            },
            Section::Figure { figure, .. },
        ) = (&*self.original, Arc::make_mut(&mut self.section))
        {
            mode.apply(original, figure);
            self.smoothing = mode;
        }
    }

    fn begin_drag(&mut self, position: Point<Pixels>, shift: bool) {
        let geometry = self.geometry.borrow();
        let Some((bounds, meta)) = geometry.as_ref() else {
            return;
        };
        if let Section::Grid { grid, .. } = &*self.section
            && grid.surface
        {
            if bounds.contains(&position) {
                self.drag = Some(ChartDrag::Rotate {
                    start: position,
                    rotation: grid.rotation.unwrap_or(DEFAULT_SURFACE_ROTATION),
                    bounds: *bounds,
                });
            }
            return;
        }
        let Some(plot) = meta.plot else { return };
        let [x, y] = point_coordinates(position - bounds.origin);
        if (plot.plot_x..=plot.plot_x + plot.plot_w).contains(&x)
            && (plot.plot_y..=plot.plot_y + plot.plot_h).contains(&y)
        {
            self.drag = Some(if shift || self.move_mode {
                ChartDrag::Pan {
                    start: position,
                    plot,
                    secondary_y: meta.secondary_y,
                    bounds: *bounds,
                }
            } else if matches!(&*self.section, Section::Figure { .. }) {
                ChartDrag::Selection(PlotSelection {
                    start: position,
                    end: position,
                })
            } else {
                return;
            });
        }
    }

    fn selection_overlay(&self, selection: PlotSelection) -> Option<Bounds<Pixels>> {
        let geometry = self.geometry.borrow();
        let (bounds, _) = geometry.as_ref()?;
        let a = selection.start - bounds.origin;
        let b = selection.end - bounds.origin;
        let width = (a.x - b.x).abs();
        let height = (a.y - b.y).abs();
        (width > px(0.0) && height > px(0.0)).then(|| Bounds {
            origin: gpui::point(a.x.min(b.x), a.y.min(b.y)),
            size: gpui::size(width, height),
        })
    }

    fn update_drag(&mut self, end: Point<Pixels>) -> bool {
        match self.drag {
            Some(ChartDrag::Selection(mut selection)) => {
                selection.end = end;
                self.drag = Some(ChartDrag::Selection(selection));
                true
            }
            Some(ChartDrag::Pan {
                start,
                plot,
                secondary_y,
                bounds,
            }) => {
                let [dx, dy] = drag_fraction(start, end, bounds);
                let Some((x, y)) = plot.panned(dx, dy) else {
                    return false;
                };
                let secondary_y = secondary_y.and_then(|[y_lo, y_hi]| {
                    PlotGeometry { y_lo, y_hi, ..plot }
                        .panned(0.0, dy)
                        .map(|(_, y)| y)
                });
                self.set_domains(x, y, secondary_y);
                true
            }
            Some(ChartDrag::Rotate {
                start,
                rotation,
                bounds,
            }) => {
                let [dx, dy] = drag_fraction(start, end, bounds);
                let Section::Grid { grid, .. } = Arc::make_mut(&mut self.section) else {
                    return false;
                };
                grid.rotation = Some([
                    (rotation[0] + dy * 160.0).clamp(10.0, 170.0),
                    (rotation[1] + dx * 180.0).rem_euclid(360.0),
                ]);
                true
            }
            None => false,
        }
    }

    fn finish_drag(&mut self, end: Point<Pixels>) -> bool {
        let Some(drag) = self.drag else { return false };
        let selection = match drag {
            ChartDrag::Selection(selection) => {
                self.drag = None;
                selection
            }
            ChartDrag::Pan { start, .. } | ChartDrag::Rotate { start, .. } => {
                let delta = end - start;
                let moved = delta.x.abs() > px(2.0) || delta.y.abs() > px(2.0);
                if moved {
                    self.update_drag(end);
                }
                self.drag = None;
                return moved;
            }
        };
        let delta = end - selection.start;
        if delta.x.abs() <= px(6.0) || delta.y.abs() <= px(6.0) {
            return false;
        }
        let domains = self.geometry.borrow().as_ref().and_then(|(bounds, meta)| {
            meta.plot?.selection(
                point_coordinates(selection.start - bounds.origin),
                point_coordinates(end - bounds.origin),
            )
        });
        let Some((x, y)) = domains else { return false };
        self.set_domains(x, y, None);
        true
    }

    fn frequency_range(&mut self, range: [f64; 2]) {
        if let Section::Figure { figure, .. } | Section::Grid { figure, .. } =
            Arc::make_mut(&mut self.section)
        {
            [figure.x.min, figure.x.max] = range.map(Some);
        }
    }

    fn db_span(&mut self, span_db: f64) {
        let center = self
            .geometry
            .borrow()
            .as_ref()
            .and_then(|(_, meta)| meta.plot)
            .map(|plot| plot.y_lo.midpoint(plot.y_hi));
        if let Some(center) = center
            && let Section::Figure { figure, .. } = Arc::make_mut(&mut self.section)
        {
            figure.y.min = Some(center - span_db / 2.0);
            figure.y.max = Some(center + span_db / 2.0);
        }
    }

    fn set_domains(&mut self, x: [f64; 2], y: [f64; 2], secondary_y: Option<[f64; 2]>) {
        if let Section::Figure { figure, .. } | Section::Grid { figure, .. } =
            Arc::make_mut(&mut self.section)
        {
            [figure.x.min, figure.x.max] = x.map(Some);
            [figure.y.min, figure.y.max] = y.map(Some);
            if let Some(axis) = &mut figure.y2
                && let Some(domain) = secondary_y
            {
                [axis.min, axis.max] = domain.map(Some);
            }
        }
    }

    fn zoom(&mut self, factor: f64, pointer: Option<Point<Pixels>>) -> bool {
        let axes = self.geometry.borrow().as_ref().and_then(|(bounds, meta)| {
            let plot = meta.plot?;
            let anchor = pointer.map_or([0.5, 0.5], |pointer| {
                drag_fraction(bounds.origin, pointer, *bounds)
            });
            let (x, y) = plot.zoomed(factor, anchor)?;
            let secondary_y = meta.secondary_y.and_then(|[y_lo, y_hi]| {
                PlotGeometry { y_lo, y_hi, ..plot }
                    .zoomed(factor, anchor)
                    .map(|(_, y)| y)
            });
            Some((x, y, secondary_y))
        });
        if let Some((x, y, secondary_y)) = axes {
            self.set_domains(x, y, secondary_y);
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.drag = None;
        self.magnification = 1.0;
        let (original, figure) = match (&*self.original, Arc::make_mut(&mut self.section)) {
            (
                Section::Figure {
                    figure: original, ..
                },
                Section::Figure { figure, .. },
            ) => (original, figure),
            (
                Section::Grid {
                    figure: original,
                    grid: original_grid,
                    ..
                },
                Section::Grid { figure, grid, .. },
            ) => {
                grid.colormap.clone_from(&original_grid.colormap);
                grid.rotation = original_grid.rotation;
                grid.show_surface = original_grid.show_surface;
                grid.show_contours = original_grid.show_contours;
                (original, figure)
            }
            _ => return,
        };
        figure.x = original.x.clone();
        figure.y = original.y.clone();
        figure.y2 = original.y2.clone();
    }

    fn surface(&mut self, action: SurfaceAction) {
        let Section::Grid { grid, .. } = Arc::make_mut(&mut self.section) else {
            return;
        };
        match action {
            SurfaceAction::Colormap(id) => grid.colormap = id.into(),
            SurfaceAction::Fill => grid.show_surface = !grid.show_surface,
            SurfaceAction::Contours => grid.show_contours = !grid.show_contours,
            SurfaceAction::Turn(delta) => {
                let rotation = grid.rotation.get_or_insert(DEFAULT_SURFACE_ROTATION);
                rotation[1] = (rotation[1] + delta).rem_euclid(360.0);
            }
            SurfaceAction::Tilt(delta) => {
                let rotation = grid.rotation.get_or_insert(DEFAULT_SURFACE_ROTATION);
                rotation[0] = (rotation[0] + delta).clamp(10.0, 170.0);
            }
        }
    }

    fn legend(&self) -> Vec<(String, bool)> {
        match &*self.section {
            Section::Figure { figure, .. } | Section::Grid { figure, .. } => figure_legend(figure),
            Section::Bar { chart, .. } if chart.legend => chart
                .groups
                .iter()
                .zip(self.bars.iter())
                .map(|(group, visible)| (group.name.clone(), *visible))
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
enum SurfaceAction {
    Colormap(&'static str),
    Fill,
    Contours,
    Turn(f64),
    Tilt(f64),
}

const SURFACE_COLORMAPS: [(&str, &str); 6] = [
    ("turbo", "Turbo"),
    ("viridis", "Viridis"),
    ("plasma", "Plasma"),
    ("inferno", "Inferno"),
    ("magma", "Magma"),
    ("rainbow", "Rainbow"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChartMenu {
    Smoothing,
    Colormap,
}

fn chart_menu_button(
    index: ChartKey,
    name: &'static str,
    value: &'static str,
    menu: ChartMenu,
    active: Option<ChartMenu>,
    cx: &mut Context<ReportView>,
) -> gpui::Stateful<Div> {
    button(
        cx,
        format!("report-chart-menu-{index}-{name}"),
        format!("{name}: {value}"),
        active == Some(menu),
    )
    .on_click(cx.listener(move |view, _: &ClickEvent, _, cx| {
        if let Some(chart) = view.charts.get_mut(&index) {
            chart.menu = (chart.menu != Some(menu)).then_some(menu);
            cx.notify();
        }
    }))
}

fn chart_button(
    index: ChartKey,
    label: &'static str,
    selected: bool,
    cx: &mut Context<ReportView>,
    action: impl Fn(&mut ChartView) + 'static,
) -> gpui::Stateful<Div> {
    button(
        cx,
        format!("report-chart-{index}-{label}"),
        label.into(),
        selected,
    )
    .on_click(cx.listener(move |view, _: &ClickEvent, _, cx| {
        if let Some(chart) = view.charts.get_mut(&index) {
            action(chart);
            cx.notify();
        }
    }))
}

fn figure_legend(figure: &Figure) -> Vec<(String, bool)> {
    if figure.legend {
        figure
            .series
            .iter()
            .map(|series| (series.name.clone(), series.visible))
            .collect()
    } else {
        Vec::new()
    }
}

fn drag_fraction(start: Point<Pixels>, end: Point<Pixels>, bounds: Bounds<Pixels>) -> [f64; 2] {
    let [dx, dy] = point_coordinates(end - start);
    let width = f64::from(f32::from(bounds.size.width)).max(1.0);
    let height = f64::from(f32::from(bounds.size.height)).max(1.0);
    [dx / width, dy / height]
}

fn point_coordinates(point: Point<Pixels>) -> [f64; 2] {
    [f64::from(f32::from(point.x)), f64::from(f32::from(point.y))]
}

fn legend_at(bounds: Bounds<Pixels>, meta: &DrawMeta, pointer: Point<Pixels>) -> Option<usize> {
    let point = pointer - bounds.origin;
    let x = f64::from(f32::from(point.x));
    let y = f64::from(f32::from(point.y));
    meta.legend
        .iter()
        .find(|entry| {
            (entry.x..=entry.x + entry.w).contains(&x) && (entry.y..=entry.y + entry.h).contains(&y)
        })
        .map(|entry| entry.series)
}

fn button(
    cx: &Context<ReportView>,
    id: String,
    label: String,
    selected: bool,
) -> gpui::Stateful<Div> {
    let theme = ButtonTheme::from(cx.theme().as_ref());
    let focus_color = theme.accent;
    Button::new(
        format!("report-view-{}-{id}", cx.entity_id().as_u64()),
        label.clone(),
    )
    .variant(ButtonVariant::Secondary)
    .size(ButtonSize::Sm)
    .selected(selected)
    .theme(theme)
    .build()
    .border_2()
    .tab_index(0)
    .role(gpui::Role::Button)
    .aria_label(label)
    .focus_visible(move |style| style.border_2().border_color(focus_color))
}

fn table_body(index: String, table: &ReportTable) -> impl IntoElement {
    let mut body = div()
        .id(format!("report-table-{index}"))
        .overflow_x_scroll()
        .w_full()
        .child(table_row(&table.columns, true));
    for row in &table.rows {
        body = body.child(table_row(row, false));
    }
    body
}

fn table_view(index: String, table: &ReportTable) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .when(!table.title.is_empty(), |element| {
            element.child(
                div()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(table.title.clone()),
            )
        })
        .when(!table.note.is_empty(), |element| {
            element.child(div().text_sm().child(table.note.clone()))
        })
        .when(!table.rows.is_empty(), |element| {
            element.child(table_body(index, table))
        })
}

fn table_row(values: &[String], heading: bool) -> impl IntoElement {
    div()
        .flex()
        .text_sm()
        .when(heading, |row| row.font_weight(gpui::FontWeight::SEMIBOLD))
        .children(
            values
                .iter()
                .map(|value| div().flex_none().w_48().px_2().py_1().child(value.clone())),
        )
}

struct MetricTooltip(String);

impl Render for MetricTooltip {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .max_w(px(420.0))
            .px_3()
            .py_2()
            .rounded_md()
            .bg(rgb(0x24272c))
            .text_color(rgb(0xf8f9fa))
            .text_sm()
            .child(self.0.clone())
    }
}

fn evidence_list(items: &[String]) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .text_sm()
        .children(items.iter().map(|item| {
            div()
                .flex()
                .gap_2()
                .child("•")
                .child(div().flex_1().min_w_0().child(item.clone()))
        }))
}

fn results_table(index: String, rows: &[ResultsRow]) -> impl IntoElement {
    let columns = RESULTS_COLUMNS.map(str::to_owned);
    let body = div()
        .id(format!("report-table-{index}"))
        .overflow_x_scroll()
        .w_full()
        .child(table_row(&columns, true))
        .children(rows.iter().enumerate().map(|(row_index, row)| {
            div()
                .flex()
                .text_sm()
                .child(
                    div()
                        .flex_none()
                        .w_48()
                        .px_2()
                        .py_1()
                        .child(row.speaker.clone()),
                )
                .children(row.metrics.iter().enumerate().map(|(column, metric)| {
                    div()
                        .id(format!("report-metric-{index}-{row_index}-{column}"))
                        .flex_none()
                        .w_48()
                        .px_2()
                        .py_1()
                        .bg(rgba(metric.tone.background_rgba()))
                        .when_some(metric.tone.foreground_rgb(), |cell, color| {
                            cell.text_color(rgb(color))
                        })
                        .when_some(metric.detail.clone(), |cell, detail| {
                            cell.tooltip(move |_, cx| {
                                cx.new(|_| MetricTooltip(detail.clone())).into()
                            })
                        })
                        .child(metric.value.clone())
                }))
        }));
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(RESULTS_TITLE),
        )
        .child(div().text_sm().child(RESULTS_NOTE))
        .child(body)
}

fn resonance_table(index: String, columns: usize, rows: &[ResonanceRow]) -> impl IntoElement {
    let headings = std::iter::once("Speaker".to_owned())
        .chain((1..=columns).map(|column| format!("Mode {column}")))
        .collect::<Vec<_>>();
    let body = div()
        .id(format!("report-table-{index}"))
        .overflow_x_scroll()
        .w_full()
        .child(table_row(&headings, true))
        .children(rows.iter().map(|row| {
            let mut cells = div().flex().text_sm().child(
                div()
                    .flex_none()
                    .w_48()
                    .px_2()
                    .py_1()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child(row.name.clone()),
            );
            if let ResonanceEvidence::Candidates(candidates) = &row.evidence {
                cells = cells.children(candidates.iter().map(|candidate| {
                    div()
                        .flex_none()
                        .w_48()
                        .px_2()
                        .py_1()
                        .border_t_2()
                        .border_color(rgb(candidate.color))
                        .child(candidate.lines.join("\n"))
                }));
                cells = cells.children(
                    (candidates.len()..columns)
                        .map(|_| div().flex_none().w_48().px_2().py_1().child("—")),
                );
            } else if let Some(status) = row.evidence.status() {
                cells = cells.child(div().px_2().py_1().child(status));
            }
            cells
        }));
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(RESONANCE_TITLE),
        )
        .child(div().text_sm().child(RESONANCE_NOTE))
        .child(body)
}

fn epa_table(index: String, rows: &[EpaScoreRow]) -> impl IntoElement {
    let columns = EPA_COLUMNS.map(str::to_owned);
    let body = div()
        .id(format!("report-table-{index}"))
        .overflow_x_scroll()
        .w_full()
        .child(table_row(&columns, true))
        .children(rows.iter().map(|row| {
            let (color, emphasized) = match &row.delta {
                EpaDelta::Improved(_) => (rgb(0x2ecc71), true),
                EpaDelta::Worsened(_) => (rgb(0xe74c3c), true),
                EpaDelta::Unavailable | EpaDelta::Unchanged => (rgb(0x999999), false),
            };
            div()
                .flex()
                .text_sm()
                .children(
                    [&row.metric, &row.before, &row.after]
                        .into_iter()
                        .map(|text| div().flex_none().w_48().px_2().py_1().child(text.clone())),
                )
                .child(
                    div()
                        .flex_none()
                        .w_48()
                        .px_2()
                        .py_1()
                        .text_color(color)
                        .when(emphasized, |cell| {
                            cell.font_weight(gpui::FontWeight::SEMIBOLD)
                        })
                        .child(row.delta.text().to_owned()),
                )
        }));
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(EPA_TITLE),
        )
        .child(body)
        .child(div().text_sm().child(EPA_CAPTION))
}

impl ReportView {
    fn render_flow_stages(
        &self,
        section: usize,
        position: usize,
        stages: &[SignalFlowStage],
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_2()
            .children(stages.iter().map(|stage| {
                let key = (section, position, stage.number);
                let open = self.open_stages.contains(&key);
                div()
                    .flex()
                    .flex_col()
                    .child(
                        button(
                            cx,
                            format!("report-stage-{section}-{position}-{}", stage.number),
                            format!("{} · {} · {}", stage.number, stage.label, stage.heading),
                            open,
                        )
                        .on_click(cx.listener(
                            move |view, _: &ClickEvent, _, cx| {
                                if !view.open_stages.insert(key) {
                                    view.open_stages.remove(&key);
                                }
                                cx.notify();
                            },
                        )),
                    )
                    .when(open, |row| {
                        row.child(
                            div()
                                .id(format!(
                                    "report-stage-parameters-{section}-{position}-{}",
                                    stage.number
                                ))
                                .overflow_x_scroll()
                                .text_sm()
                                .child(stage.parameters.clone()),
                        )
                    })
            }))
    }
}

impl Render for ReportView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let focus = self.focus.get_or_insert_with(|| cx.focus_handle()).clone();
        if window.focused(cx).is_none() {
            window.focus(&focus, cx);
        }
        let mut tabs = div().flex().flex_wrap().gap_2();
        for (dashboard, label) in self.dashboards.iter().enumerate() {
            tabs = tabs.child(
                button(
                    cx,
                    format!("report-dashboard-{dashboard}"),
                    label.clone(),
                    self.dashboard == dashboard,
                )
                .on_click(cx.listener(move |view, _: &ClickEvent, _, cx| {
                    view.dashboard = dashboard;
                    view.scroll = ScrollHandle::new();
                    cx.notify();
                })),
            );
        }

        let mut contents = div()
            .id("report-sections")
            .track_scroll(&self.scroll)
            .overflow_y_scroll()
            .flex_1()
            .min_h_0()
            .flex()
            .flex_col()
            .gap_3();
        let mut flow_scopes = Vec::new();
        for section in &self.sections {
            if section.dashboard == self.dashboard
                && section.heading.as_deref() == Some("DSP signal flow")
                && let Some(scope) = &section.channel
                && !flow_scopes.contains(scope)
            {
                flow_scopes.push(scope.clone());
            }
        }
        let selected_scope = flow_scopes
            .get(self.flow_scope)
            .or_else(|| flow_scopes.first());
        let mut heading = None;
        let mut channel = None;
        let mut omitted = false;
        for i in 0..self.sections.len() {
            let section = &self.sections[i];
            if section.dashboard != self.dashboard {
                continue;
            }
            if matches!(section.content, Content::BrowserContent) {
                omitted = true;
                continue;
            }
            if section.heading != heading {
                heading = section.heading.clone();
                channel = None;
                if let Some(label) = &heading {
                    contents = contents.child(div().text_lg().child(label.clone()));
                    if label == "DSP signal flow" && flow_scopes.len() > 1 {
                        contents = contents.child(div().flex().flex_wrap().gap_2().children(
                            flow_scopes.iter().enumerate().map(|(scope, label)| {
                                button(
                                    cx,
                                    format!("report-flow-scope-{scope}"),
                                    label.clone(),
                                    Some(label) == selected_scope,
                                )
                                .on_click(cx.listener(
                                    move |view, _: &ClickEvent, _, cx| {
                                        view.flow_scope = scope;
                                        cx.notify();
                                    },
                                ))
                            }),
                        ));
                    }
                }
            }
            if section.heading.as_deref() == Some("DSP signal flow")
                && section.channel.is_some()
                && section.channel.as_ref() != selected_scope
            {
                continue;
            }
            let index = section.index;
            if section.channel != channel {
                channel = section.channel.clone();
                let has_heading = matches!(&self.document.sections[index], ReportSection::Text(text) if text.heading.is_some());
                if let Some(label) = &channel
                    && !has_heading
                {
                    contents = contents.child(div().text_sm().child(label.clone()));
                }
            }
            let element = match &section.content {
                Content::Table => match &self.document.sections[index] {
                    ReportSection::Table(table) => {
                        table_view(index.to_string(), &table.table).into_any_element()
                    }
                    ReportSection::Tables(section) => div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(div().text_lg().child(section.group.title.clone()))
                        .children(section.group.tables.iter().enumerate().map(
                            |(table_index, table)| {
                                table_view(format!("{index}-{table_index}"), table)
                            },
                        ))
                        .into_any_element(),
                    ReportSection::CaptureQuality(quality) => {
                        table_view(index.to_string(), &quality.table).into_any_element()
                    }
                    _ => continue,
                },
                Content::Text => match &self.document.sections[index] {
                    ReportSection::Text(text) => div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .when_some(text.heading.as_ref(), |element, heading| {
                            element.child(div().text_lg().child(heading.clone()))
                        })
                        .children(
                            text.paragraphs
                                .iter()
                                .map(|paragraph| div().text_sm().child(paragraph.clone())),
                        )
                        .into_any_element(),
                    _ => continue,
                },
                Content::Notices => match &self.document.sections[index] {
                    ReportSection::PlaybackNotices(notices) => div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .children(notices.iter().map(|notice| {
                            let border = match notice.tone {
                                NoticeTone::Success => rgb(0x2ecc71),
                                NoticeTone::Warning => rgb(0xf1c40f),
                                NoticeTone::Error => rgb(0xe74c3c),
                            };
                            div()
                                .border_1()
                                .border_color(border)
                                .p_3()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(
                                    div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child(notice.title.clone()),
                                )
                                .children(
                                    notice
                                        .paragraphs
                                        .iter()
                                        .map(|paragraph| div().text_sm().child(paragraph.clone())),
                                )
                        }))
                        .into_any_element(),
                    _ => continue,
                },
                Content::Filters => match &self.document.sections[index] {
                    ReportSection::Filters(section) => {
                        self.render_filters(index, &section.listing, cx)
                    }
                    _ => continue,
                },
                Content::Chart(chart) => self.render_chart(ChartKey(index, 0), chart.clone(), cx),
                Content::Diagnostics(charts) => {
                    let document = self.document.clone();
                    let ReportSection::Diagnostics(section) = &document.sections[index] else {
                        continue;
                    };
                    let blocks = &section.blocks;
                    let charts = charts.clone();
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .when_some(section.introduction.as_ref(), |section, intro| {
                            section
                                .child(div().text_lg().child(intro.title.clone()))
                                .when(!intro.text.is_empty(), |section| {
                                    section.child(div().text_sm().child(intro.text.clone()))
                                })
                        })
                        .children(blocks.iter().zip(charts).enumerate().map(
                            |(position, (block, chart))| {
                                match block {
                                    DiagnosticBlock::MajorHeading(text) => {
                                        div().text_xl().child(text.clone()).into_any_element()
                                    }
                                    DiagnosticBlock::VerificationStatus {
                                        status,
                                        count,
                                        exit_code,
                                    } => div()
                                        .text_sm()
                                        .child(format!(
                                            "Recorded capture verification status: {status}{}",
                                            verification_status_detail(*count, *exit_code)
                                        ))
                                        .into_any_element(),
                                    DiagnosticBlock::PreformattedDisclosure { title, text } => {
                                        let key = ChartKey(index, position);
                                        let open = self.open_diagnostics.contains(&key);
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_2()
                                            .child(
                                                button(
                                                    cx,
                                                    format!("report-calibration-{key}"),
                                                    title.clone(),
                                                    open,
                                                )
                                                .on_click(cx.listener(
                                                    move |view, _: &ClickEvent, _, cx| {
                                                        if !view.open_diagnostics.insert(key) {
                                                            view.open_diagnostics.remove(&key);
                                                        }
                                                        cx.notify();
                                                    },
                                                )),
                                            )
                                            .when(open, |section| {
                                                section.child(
                                                    div()
                                                        .id(format!(
                                                            "report-calibration-text-{key}"
                                                        ))
                                                        .overflow_x_scroll()
                                                        .text_sm()
                                                        .child(text.clone()),
                                                )
                                            })
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::Heading(text) => {
                                        div().text_lg().child(text.clone()).into_any_element()
                                    }
                                    DiagnosticBlock::Subheading(text) => div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child(text.clone())
                                        .into_any_element(),
                                    DiagnosticBlock::Paragraph(text)
                                    | DiagnosticBlock::Caption(text) => {
                                        div().text_sm().child(text.clone()).into_any_element()
                                    }
                                    DiagnosticBlock::Table(table)
                                    | DiagnosticBlock::MixedPhaseTable(table)
                                    | DiagnosticBlock::EvidenceTable(table)
                                    | DiagnosticBlock::Matrix(table) => {
                                        table_view(format!("{index}-{position}"), table)
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::CurveTable { table, .. } => {
                                        table_view(format!("{index}-{position}"), table)
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::EvidencePanel(panel) => div()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .child(div().text_lg().child(panel.title.clone()))
                                        .children(
                                            panel
                                                .paragraphs
                                                .iter()
                                                .map(|text| div().text_sm().child(text.clone())),
                                        )
                                        .children(panel.groups.iter().enumerate().map(
                                            |(group_index, group)| {
                                                table_view(
                                                    format!("{index}-{position}-{group_index}"),
                                                    group,
                                                )
                                            },
                                        ))
                                        .into_any_element(),
                                    DiagnosticBlock::SignalFlowStages(stages) => self
                                        .render_flow_stages(index, position, stages, cx)
                                        .into_any_element(),
                                    DiagnosticBlock::CaptureArrival { table, .. } => div()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .child(div().text_lg().child(table.title.clone()))
                                        .when(!table.note.is_empty(), |section| {
                                            section.child(div().text_sm().child(table.note.clone()))
                                        })
                                        .when_some(chart, |section, chart| {
                                            section
                                                .child(self.render_chart(
                                                    ChartKey(index, position),
                                                    chart,
                                                    cx,
                                                ))
                                                .child(div().text_sm().child(ARRIVAL_PLOT_NOTE))
                                        })
                                        .child(table_body(format!("{index}-{position}"), table))
                                        .into_any_element(),
                                    DiagnosticBlock::DecisionTable(table) => {
                                        table_view(format!("{index}-{position}"), table)
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::EvidenceList(items)
                                    | DiagnosticBlock::AcceptanceSets(items) => {
                                        evidence_list(items).into_any_element()
                                    }
                                    DiagnosticBlock::StageHistory { title, note, items } => {
                                        let key = ChartKey(index, position);
                                        let open = self.open_diagnostics.contains(&key);
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_2()
                                            .child(
                                                button(
                                                    cx,
                                                    format!("report-history-{key}"),
                                                    title.clone(),
                                                    open,
                                                )
                                                .on_click(cx.listener(
                                                    move |view, _: &ClickEvent, _, cx| {
                                                        if !view.open_diagnostics.insert(key) {
                                                            view.open_diagnostics.remove(&key);
                                                        }
                                                        cx.notify();
                                                    },
                                                )),
                                            )
                                            .when(open, |section| {
                                                section
                                                    .child(div().text_sm().child(note.clone()))
                                                    .child(evidence_list(items))
                                            })
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::ResultsSummary(rows) => {
                                        results_table(format!("{index}-{position}"), rows)
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::ResonanceSummary { columns, rows } => {
                                        resonance_table(
                                            format!("{index}-{position}"),
                                            *columns,
                                            rows,
                                        )
                                        .into_any_element()
                                    }
                                    DiagnosticBlock::EpaScores(rows) => {
                                        epa_table(format!("{index}-{position}"), rows)
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::Reference { text, label, url } => {
                                        let url = url.clone();
                                        div()
                                            .text_sm()
                                            .child(text.clone())
                                            .child(
                                                div()
                                                    .id(format!(
                                                        "report-reference-{index}-{position}"
                                                    ))
                                                    .tab_index(0)
                                                    .role(gpui::Role::Link)
                                                    .aria_label(label.clone())
                                                    .border_1()
                                                    .border_color(rgba(0x00000000))
                                                    .focus_visible(|style| {
                                                        style.border_color(rgb(0x438de6))
                                                    })
                                                    .cursor_pointer()
                                                    .underline()
                                                    .child(label.clone())
                                                    .on_click(move |_, _, cx| cx.open_url(&url)),
                                            )
                                            .into_any_element()
                                    }
                                    DiagnosticBlock::Emphasis(text) => div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child(text.clone())
                                        .into_any_element(),
                                    DiagnosticBlock::Chart(_)
                                    | DiagnosticBlock::AcceptanceChart(_)
                                    | DiagnosticBlock::CaptureChart(_) => chart
                                        .map(|chart| {
                                            self.render_chart(ChartKey(index, position), chart, cx)
                                        })
                                        .unwrap_or_else(|| div().into_any_element()),
                                }
                            },
                        ))
                        .into_any_element()
                }
                Content::BrowserContent => continue,
            };
            contents = contents.child(div().flex_none().child(element));
        }
        if omitted {
            contents = contents.child(
                div()
                    .text_sm()
                    .child("This dashboard includes sections available in the HTML report."),
            );
        }
        let dark = self.dark_appearance.unwrap_or_else(|| {
            matches!(
                window.appearance(),
                gpui::WindowAppearance::Dark | gpui::WindowAppearance::VibrantDark
            )
        });
        let (background, foreground) = if dark {
            (rgb(0x20242a), rgb(0xe9edf3))
        } else {
            (rgb(0xffffff), rgb(0x172238))
        };
        div()
            .id("native-report")
            .tab_group()
            .track_focus(&focus)
            .on_key_down(|event: &KeyDownEvent, window, cx| {
                let modifiers = event.keystroke.modifiers;
                if event.keystroke.key != "tab"
                    || modifiers.control
                    || modifiers.alt
                    || modifiers.platform
                {
                    return;
                }
                if modifiers.shift {
                    window.focus_prev(cx);
                } else {
                    window.focus_next(cx);
                }
                cx.stop_propagation();
            })
            .size_full()
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .bg(background)
            .text_color(foreground)
            .child(div().text_xl().child(self.document.title().to_owned()))
            .child(tabs)
            .child(contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_route_requires_every_section_to_be_renderable() {
        use autoeq_report_wasm::{
            PreparedReportData, report_dataset_payload::build_report_document,
        };
        let input = PreparedReportData::new(
            json!({"channels":{}}),
            None,
            None,
            None,
            json!({"schema_version":1,"state":"not_configured","target":null}),
            None,
        )
        .unwrap();
        let mut document = build_report_document(&input, &Default::default()).unwrap();
        document
            .sections
            .retain(|section| !matches!(section, ReportSection::Legacy(_)));
        let structured = Arc::new(document.clone());
        assert!(NativeReport::try_complete(structured).is_some());
        document.sections.push(ReportSection::Legacy(json!({
            "kind": "html", "html": "<p>Legacy evidence</p>"
        })));
        assert!(NativeReport::try_complete(Arc::new(document)).is_none());
    }

    #[test]
    fn zoom_preserves_the_log_center_and_reset_preserves_visibility() {
        let section = serde_json::from_value(json!({
            "kind": "figure",
            "figure": {
                "x": {"scale": "log"},
                "y": {},
                "y2": {},
                "series": [{"name": "Measured", "x": [100.0], "y": [0.0], "visible": false}]
            }
        }))
        .unwrap();
        let mut chart = ChartView::new(Arc::new(section));
        chart.geometry.replace(Some((
            Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: gpui::size(px(100.0), px(100.0)),
            },
            DrawMeta {
                plot: Some(PlotGeometry {
                    plot_x: 0.0,
                    plot_y: 0.0,
                    plot_w: 100.0,
                    plot_h: 100.0,
                    x_lo: 10.0,
                    x_hi: 1000.0,
                    y_lo: -10.0,
                    y_hi: 10.0,
                    log_x: true,
                }),
                secondary_y: Some([100.0, 300.0]),
                ..DrawMeta::default()
            },
        )));
        assert!(chart.zoom(0.5, None));
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert!((figure.x.min.unwrap() - 10_f64.powf(1.5)).abs() < 1e-10);
        assert!((figure.x.max.unwrap() - 10_f64.powf(2.5)).abs() < 1e-10);
        assert_eq!([figure.y.min, figure.y.max], [Some(-5.0), Some(5.0)]);
        let secondary = figure.y2.as_ref().unwrap();
        assert_eq!([secondary.min, secondary.max], [Some(150.0), Some(250.0)]);

        assert!(chart.zoom(0.5, Some(gpui::point(px(25.0), px(25.0)))));
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert!((figure.x.min.unwrap() - 10_f64.powf(1.25)).abs() < 1e-10);
        assert!((figure.x.max.unwrap() - 10_f64.powf(2.25)).abs() < 1e-10);
        assert_eq!([figure.y.min, figure.y.max], [Some(-2.5), Some(7.5)]);
        let secondary = figure.y2.as_ref().unwrap();
        assert_eq!([secondary.min, secondary.max], [Some(175.0), Some(275.0)]);

        chart.begin_drag(gpui::point(px(75.0), px(75.0)), false);
        assert!(chart.finish_drag(gpui::point(px(25.0), px(25.0))));
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert!((figure.x.min.unwrap() - 10_f64.powf(1.5)).abs() < 1e-10);
        assert!((figure.x.max.unwrap() - 10_f64.powf(2.5)).abs() < 1e-10);
        assert_eq!([figure.y.min, figure.y.max], [Some(-5.0), Some(5.0)]);
        assert!(!figure.series[0].visible);

        chart.frequency_range([20.0, 200.0]);
        chart.db_span(50.0);
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert_eq!([figure.x.min, figure.x.max], [Some(20.0), Some(200.0)]);
        assert_eq!([figure.y.min, figure.y.max], [Some(-25.0), Some(25.0)]);
        let secondary = figure.y2.as_ref().unwrap();
        assert_eq!([secondary.min, secondary.max], [Some(175.0), Some(275.0)]);
        assert!(!figure.series[0].visible);

        chart.begin_drag(gpui::point(px(25.0), px(25.0)), false);
        assert!(!chart.finish_drag(gpui::point(px(29.0), px(29.0))));
        assert!(chart.drag.is_none());

        chart.begin_drag(gpui::point(px(25.0), px(25.0)), true);
        assert!(chart.finish_drag(gpui::point(px(75.0), px(50.0))));
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert!((figure.x.min.unwrap() - 1.0).abs() < 1e-10);
        assert!((figure.x.max.unwrap() - 100.0).abs() < 1e-10);
        assert_eq!([figure.y.min, figure.y.max], [Some(-5.0), Some(15.0)]);
        let secondary = figure.y2.as_ref().unwrap();
        assert_eq!([secondary.min, secondary.max], [Some(150.0), Some(350.0)]);
        assert!(!figure.series[0].visible);

        chart.reset();
        let Section::Figure { figure, .. } = &*chart.section else {
            panic!("expected a figure");
        };
        assert_eq!([figure.x.min, figure.y.min], [None, None]);
        let secondary = figure.y2.as_ref().unwrap();
        assert_eq!([secondary.min, secondary.max], [None, None]);
        assert!(!figure.series[0].visible);
    }

    #[test]
    fn surface_controls_change_only_view_settings_and_reset_restores_them() {
        let section = serde_json::from_value(json!({
            "kind": "grid",
            "figure": {"x": {}, "y": {}},
            "grid": {
                "x": [10.0, 100.0], "y": [0.0, 1.0],
                "z": [[-2.0, -1.0], [-4.0, -3.0]],
                "surface": true, "show_surface": false, "zmin": -10.0, "zmax": 0.0
            }
        }))
        .unwrap();
        let mut chart = ChartView::new(Arc::new(section));
        chart.surface(SurfaceAction::Colormap("viridis"));
        chart.surface(SurfaceAction::Fill);
        chart.surface(SurfaceAction::Contours);
        chart.surface(SurfaceAction::Tilt(200.0));
        let Section::Grid { grid, .. } = &*chart.section else {
            panic!("expected a grid");
        };
        assert_eq!(grid.colormap, "viridis");
        assert!(grid.show_surface);
        assert!(!grid.show_contours);
        assert_eq!(grid.rotation.unwrap()[0], 170.0);
        assert_eq!(grid.z, vec![vec![-2.0, -1.0], vec![-4.0, -3.0]]);
        chart.geometry.replace(Some((
            Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: gpui::size(px(100.0), px(100.0)),
            },
            DrawMeta::default(),
        )));
        chart.begin_drag(gpui::point(px(50.0), px(50.0)), false);
        assert!(chart.finish_drag(gpui::point(px(75.0), px(25.0))));
        let Section::Grid { grid, .. } = &*chart.section else {
            panic!("expected a grid");
        };
        assert_eq!(
            grid.rotation.unwrap(),
            [
                130.0,
                (DEFAULT_SURFACE_ROTATION[1] + 45.0).rem_euclid(360.0)
            ]
        );
        chart.reset();
        let Section::Grid { grid, .. } = &*chart.section else {
            panic!("expected a grid");
        };
        assert_eq!(grid.colormap, "turbo");
        assert!(!grid.show_surface);
        assert!(grid.show_contours);
        assert_eq!(grid.rotation, None);
    }
}
