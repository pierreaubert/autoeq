//! WASM export shell for the default 2D report renderer.
//!
//! The HTML shell inlines the generated glue plus the base64 `.wasm` and
//! drives these functions: [`render_section`], [`legend_json`],
//! [`toggle_series`], plus [`last_error`] and [`schema_version`]. All layout
//! math lives in the shared [`autoeq_report_wasm::draw`] core (d3rs
//! scales/geometry).
//!
//! This crate is `cdylib`-only on purpose: the release profile sets
//! `panic = "abort"` while test targets use `unwind`, so a `cdylib` + `rlib`
//! crate that host code links is built twice with identical output filenames
//! and parallel rustc invocations race on them (cargo#6313), flaking
//! `cargo test --release`. The Rust API stays in `autoeq-report-wasm`
//! (rlib); only the `wasm-bindgen` exports live here, and nothing may
//! depend on this crate (see `Cargo.toml`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use autoeq_report_wasm::draw::{
    Ctx, DrawMeta, LegendEntry, PaintTriangle, TextAlign, draw_bar, draw_figure, draw_sankey,
    paint_triangles,
};
use autoeq_report_wasm::grid::draw_grid;
use autoeq_report_wasm::schema::{SCHEMA_VERSION, Section};

/// Retained per-canvas document for legend toggles.
struct Retained {
    section: Section,
    legend: Vec<LegendEntry>,
    w: f64,
    h: f64,
    dpr: f64,
}

fn state() -> &'static Mutex<HashMap<String, Retained>> {
    static STATE: OnceLock<Mutex<HashMap<String, Retained>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Canvas 2D backend for the shared draw core (CSS pixel units).
struct CanvasCtx {
    ctx: CanvasRenderingContext2d,
}

impl CanvasCtx {
    fn new(canvas: &HtmlCanvasElement, w: f64, h: f64, dpr: f64) -> Option<Self> {
        canvas.set_width((w * dpr).max(1.0) as u32);
        canvas.set_height((h * dpr).max(1.0) as u32);
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")
            .ok()?
            .and_then(|c| c.dyn_into().ok())?;
        let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
        Some(Self { ctx })
    }
}

impl Ctx for CanvasCtx {
    fn triangles(&mut self, triangles: &[PaintTriangle]) {
        // The shell installs this synchronous compositor only after GPU setup.
        // Keep d3rs projection, colors, and painter order identical in both paths.
        let hook = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("__reportTriangles"))
            .ok()
            .and_then(|value| value.dyn_into::<js_sys::Function>().ok());
        if let Some(hook) = hook {
            let mut vertices = Vec::with_capacity(triangles.len() * 18);
            for triangle in triangles {
                for &(x, y) in &triangle.points {
                    vertices.extend_from_slice(&[
                        x as f32,
                        y as f32,
                        triangle.color[0] as f32 / 255.0,
                        triangle.color[1] as f32 / 255.0,
                        triangle.color[2] as f32 / 255.0,
                        1.0,
                    ]);
                }
            }
            let data = js_sys::Float32Array::from(vertices.as_slice());
            if hook
                .call2(&JsValue::NULL, self.ctx.as_ref(), data.as_ref())
                .ok()
                .and_then(|value| value.as_bool())
                == Some(true)
            {
                return;
            }
        }
        paint_triangles(self, triangles);
    }
    fn set_fill(&mut self, css: &str) {
        self.ctx.set_fill_style_str(css);
    }
    fn set_stroke(&mut self, css: &str) {
        self.ctx.set_stroke_style_str(css);
    }
    fn set_line_width(&mut self, w: f64) {
        self.ctx.set_line_width(w);
    }
    fn set_dash(&mut self, pattern: &[f64]) {
        let arr = js_sys::Array::new();
        for v in pattern {
            arr.push(&JsValue::from(*v));
        }
        let _ = self.ctx.set_line_dash(&arr);
    }
    fn set_font(&mut self, css_font: &str) {
        self.ctx.set_font(css_font);
    }
    fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
        self.ctx.fill_rect(x, y, w, h);
    }
    fn begin_path(&mut self) {
        self.ctx.begin_path();
    }
    fn move_to(&mut self, x: f64, y: f64) {
        self.ctx.move_to(x, y);
    }
    fn line_to(&mut self, x: f64, y: f64) {
        self.ctx.line_to(x, y);
    }
    fn bezier_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64) {
        self.ctx.bezier_curve_to(c1x, c1y, c2x, c2y, x, y);
    }
    fn close_path(&mut self) {
        self.ctx.close_path();
    }
    fn stroke(&mut self) {
        self.ctx.stroke();
    }
    fn fill(&mut self) {
        self.ctx.fill();
    }
    fn fill_text(&mut self, text: &str, x: f64, y: f64, align: TextAlign) {
        self.ctx.set_text_align(match align {
            TextAlign::Left => "left",
            TextAlign::Center => "center",
            TextAlign::Right => "right",
        });
        let _ = self.ctx.fill_text(text, x, y);
    }
    fn fill_text_rotated(&mut self, text: &str, x: f64, y: f64, angle_deg: f64, align: TextAlign) {
        self.ctx.save();
        let _ = self.ctx.translate(x, y);
        let _ = self.ctx.rotate(angle_deg * std::f64::consts::PI / 180.0);
        self.fill_text(text, 0.0, 0.0, align);
        self.ctx.restore();
    }
    fn text_width(&mut self, text: &str) -> f64 {
        self.ctx
            .measure_text(text)
            .map(|m| m.width())
            .unwrap_or(text.chars().count() as f64 * 6.5)
    }
}

fn canvas_by_id(id: &str) -> Option<HtmlCanvasElement> {
    let window = web_sys::window()?;
    let document = window.document()?;
    document.get_element_by_id(id)?.dyn_into().ok()
}

fn draw_error(canvas_id: &str, msg: &str) {
    if let Some(canvas) = canvas_by_id(canvas_id)
        && let Some(mut backend) = CanvasCtx::new(&canvas, 600.0, 80.0, 1.0)
    {
        backend.set_fill("#ffffff");
        backend.fill_rect(0.0, 0.0, 600.0, 80.0);
        backend.set_fill("#b00020");
        backend.set_font("13px system-ui, sans-serif");
        backend.fill_text(msg, 12.0, 30.0, TextAlign::Left);
    }
    web_sys::console::error_1(&JsValue::from_str(&format!("report2d/{canvas_id}: {msg}")));
}

/// Render one payload section into a canvas.
///
/// Returns 0 on success, 1 when the canvas is missing, 2 on bad JSON.
#[wasm_bindgen]
pub fn render_section(canvas_id: &str, section_json: &str, w: f64, h: f64, dpr: f64) -> i32 {
    let Some(canvas) = canvas_by_id(canvas_id) else {
        return 1;
    };
    let section: Section = match serde_json::from_str(section_json) {
        Ok(s) => s,
        Err(e) => {
            draw_error(canvas_id, &format!("bad section JSON: {e}"));
            return 2;
        }
    };
    let Some(mut backend) = CanvasCtx::new(&canvas, w, h, dpr) else {
        return 1;
    };
    let meta = draw_section(&mut backend, &section, w, h);
    if let Ok(mut map) = state().lock() {
        map.insert(
            canvas_id.to_string(),
            Retained {
                section,
                legend: meta.legend,
                w,
                h,
                dpr,
            },
        );
    }
    0
}

/// Legend hit rectangles for a rendered canvas as JSON
/// (`[{"series":i,"x":..,"y":..,"w":..,"h":..}]`, `"[]"` when unknown).
#[wasm_bindgen]
pub fn legend_json(canvas_id: &str) -> String {
    let entries = state()
        .lock()
        .map(|map| {
            map.get(canvas_id)
                .map(|r| r.legend.clone())
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let mut out = String::from("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"series\":{},\"x\":{:.1},\"y\":{:.1},\"w\":{:.1},\"h\":{:.1}}}",
            e.series, e.x, e.y, e.w, e.h
        ));
    }
    out.push(']');
    out
}

/// Flip one series' visibility and re-render. Returns 1 when re-rendered,
/// 0 when the canvas or series is unknown, -99 on an internal panic
/// (see [`last_error`]).
#[wasm_bindgen]
pub fn toggle_series(canvas_id: &str, idx: usize) -> i32 {
    // Figures flip the flag in the retained document; bar charts keep
    // visibility in a side table (schema v1 has no per-group flag).
    // Guards are never held across the `bar_toggle` call (separate lock).
    let step = state().lock().map(|map| {
        map.get(canvas_id).map(|r| {
            let is_bar = matches!(r.section, Section::Bar { .. });
            let is_fig = matches!(r.section, Section::Figure { .. } | Section::Grid { .. });
            (is_fig, is_bar, r.w, r.h, r.dpr)
        })
    });
    let Ok(Some((is_fig, is_bar, w, h, dpr))) = step else {
        return 0;
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if is_fig {
            let flipped = state()
                .lock()
                .map(|mut map| {
                    map.get_mut(canvas_id)
                        .and_then(|r| match &mut r.section {
                            Section::Figure { figure: f, .. } | Section::Grid { figure: f, .. } => {
                                f.series.get_mut(idx).map(|s| {
                                    s.visible = !s.visible;
                                })
                            }
                            _ => None,
                        })
                        .is_some()
                })
                .unwrap_or(false);
            if !flipped {
                return 0;
            }
        } else if is_bar {
            if !bar_toggle(canvas_id, idx) {
                return 0;
            }
        } else {
            return 0;
        }
        rerender(canvas_id, w, h, dpr)
    }));
    match outcome {
        Ok(rc) => rc,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(format!("toggle_series({canvas_id},{idx}): {msg}"));
            -99
        }
    }
}

/// Last internal error captured by [`toggle_series`] (empty when none).
#[wasm_bindgen]
pub fn last_error() -> String {
    last_error_slot()
        .lock()
        .map(|slot| slot.clone())
        .unwrap_or_default()
}

fn last_error_slot() -> &'static Mutex<String> {
    static SLOT: OnceLock<Mutex<String>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(msg: String) {
    if let Ok(mut slot) = last_error_slot().lock() {
        *slot = msg;
    }
}

/// Out-of-band visibility flags for bar groups (schema v1 has no flag).
fn bar_state() -> &'static Mutex<HashMap<String, Vec<bool>>> {
    static BARS: OnceLock<Mutex<HashMap<String, Vec<bool>>>> = OnceLock::new();
    BARS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bar_toggle(canvas_id: &str, idx: usize) -> bool {
    let is_bar = state()
        .lock()
        .map(|map| {
            matches!(
                map.get(canvas_id).map(|r| &r.section),
                Some(Section::Bar { .. })
            )
        })
        .unwrap_or(false);
    if !is_bar {
        return false;
    }
    if let Ok(mut map) = bar_state().lock() {
        let flags = map.entry(canvas_id.to_string()).or_default();
        while flags.len() <= idx {
            flags.push(true);
        }
        flags[idx] = !flags[idx];
    }
    true
}

fn rerender(canvas_id: &str, w: f64, h: f64, dpr: f64) -> i32 {
    let Some(canvas) = canvas_by_id(canvas_id) else {
        return 0;
    };
    let doc = state().lock().map(|map| {
        map.get(canvas_id).map(|r| {
            let bars = bar_state()
                .lock()
                .map(|m| m.get(canvas_id).cloned().unwrap_or_default())
                .unwrap_or_default();
            (r.section.clone(), bars)
        })
    });
    let Ok(Some((section, bars))) = doc else {
        return 0;
    };
    let Some(mut backend) = CanvasCtx::new(&canvas, w, h, dpr) else {
        return 0;
    };
    let meta = match &section {
        Section::Grid { figure, grid, .. } => draw_grid(&mut backend, figure, grid, w, h),
        Section::Figure { figure: f, .. } => draw_figure(&mut backend, f, w, h),
        Section::Bar { chart: b, .. } => draw_bar(&mut backend, b, w, h, &bars),
        Section::Sankey { chart: s, .. } => {
            draw_sankey(&mut backend, s, w, h);
            DrawMeta::default()
        }
        Section::Html { .. } => DrawMeta::default(),
    };
    if let Ok(mut map) = state().lock()
        && let Some(r) = map.get_mut(canvas_id)
    {
        r.legend = meta.legend;
    }
    1
}

fn draw_section(backend: &mut CanvasCtx, section: &Section, w: f64, h: f64) -> DrawMeta {
    match section {
        Section::Grid { figure, grid, .. } => draw_grid(backend, figure, grid, w, h),
        Section::Figure { figure: f, .. } => draw_figure(backend, f, w, h),
        Section::Bar { chart: b, .. } => draw_bar(backend, b, w, h, &[]),
        Section::Sankey { chart: s, .. } => {
            draw_sankey(backend, s, w, h);
            DrawMeta::default()
        }
        Section::Html { .. } => DrawMeta::default(),
    }
}

/// Payload schema discriminator (lets the shell refuse stale documents).
#[wasm_bindgen]
pub fn schema_version() -> String {
    SCHEMA_VERSION.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_report_wasm::schema::SCHEMA_VERSION;

    // Host-runnable exports: miss paths never touch the DOM.
    #[test]
    fn exports_match_schema() {
        assert_eq!(schema_version(), SCHEMA_VERSION);
    }

    #[test]
    fn unknown_canvas_answers_empty() {
        assert_eq!(legend_json("no-such-canvas"), "[]");
        assert_eq!(toggle_series("no-such-canvas", 0), 0);
        assert_eq!(last_error(), "");
    }
}
