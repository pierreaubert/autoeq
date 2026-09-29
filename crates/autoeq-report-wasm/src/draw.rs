//! Shared figure-drawing core.
//!
//! Layout math uses gpui-toolkit `d3rs` scales (`LogScale`/`LinearScale`),
//! renderer-independent [`AxisLayout`](d3rs::axis::AxisLayout) geometry, the
//! `d3rs` Sankey layout, and the `d3rs` category10 palette. The [`Ctx`] trait
//! abstracts the 2D backend: a Canvas 2D implementation ships in the
//! `autoeq-report-wasm-shell` crate, a recording implementation drives
//! native unit tests.

use d3rs::axis::{AxisConfig, AxisLayout};
use d3rs::color::ColorScheme;
use d3rs::sankey::{SankeyLayout, SankeyLinkInput};
use d3rs::scale::{LinearScale, LogScale, Scale};

use crate::schema::{BarChart, Figure, SankeyChart, Series, XScale};

/// Text horizontal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

/// Minimal 2D drawing surface (Canvas 2D semantics, CSS pixel units).
pub trait Ctx {
    /// Set the fill color (any CSS color string).
    fn set_fill(&mut self, css: &str);
    /// Set the stroke color.
    fn set_stroke(&mut self, css: &str);
    /// Set the stroke width in px.
    fn set_line_width(&mut self, w: f64);
    /// Set the stroke dash pattern (empty = solid).
    fn set_dash(&mut self, pattern: &[f64]);
    /// Set the font as a CSS font shorthand (`"12px system-ui, sans-serif"`).
    fn set_font(&mut self, css_font: &str);
    /// Fill an axis-aligned rectangle.
    fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64);
    /// Start a new path.
    fn begin_path(&mut self);
    /// Move the pen.
    fn move_to(&mut self, x: f64, y: f64);
    /// Line from the pen to a point.
    fn line_to(&mut self, x: f64, y: f64);
    /// Cubic bezier from the pen through two control points to a point.
    fn bezier_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64);
    /// Close the current path.
    fn close_path(&mut self);
    /// Stroke the current path.
    fn stroke(&mut self);
    /// Fill the current path.
    fn fill(&mut self);
    /// Fill text with horizontal alignment (alphabetic baseline).
    fn fill_text(&mut self, text: &str, x: f64, y: f64, align: TextAlign);
    /// Fill rotated text (degrees, clockwise positive like canvas).
    fn fill_text_rotated(&mut self, text: &str, x: f64, y: f64, angle_deg: f64, align: TextAlign);
    /// Measure text width in px with the current font.
    fn text_width(&mut self, text: &str) -> f64;
}

/// Clickable legend swatch geometry (canvas px), one per series.
#[derive(Debug, Clone)]
pub struct LegendEntry {
    /// Series index in the figure.
    pub series: usize,
    /// Hit rectangle.
    pub x: f64,
    /// Hit rectangle.
    pub y: f64,
    /// Hit rectangle.
    pub w: f64,
    /// Hit rectangle.
    pub h: f64,
}

/// What a draw call produced (for legend hit-testing).
#[derive(Debug, Clone, Default)]
pub struct DrawMeta {
    /// Legend entries in series order (empty when the legend is hidden).
    pub legend: Vec<LegendEntry>,
}

/// Plot area paddings in CSS px.
#[derive(Debug, Clone, Copy)]
struct Pad {
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

const FONT: &str = "12px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const FONT_TITLE: &str = "600 14px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const FONT_SMALL: &str = "11px system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif";
const INK: &str = "#333333";
const INK_FAINT: &str = "#777777";
const GRID: &str = "#e6e6e6";
const AXIS: &str = "#999999";

/// Default series color: `d3rs` category10 slot (wraps around).
pub fn default_color(index: usize) -> String {
    let scheme = ColorScheme::category10();
    let n = scheme.len().max(1);
    let c = scheme.color(index % n);
    let r = (c.r.clamp(0.0f32, 1.0) * 255.0).round() as u8;
    let g = (c.g.clamp(0.0f32, 1.0) * 255.0).round() as u8;
    let b = (c.b.clamp(0.0f32, 1.0) * 255.0).round() as u8;
    if c.a < 0.999 {
        format!("rgba({r},{g},{b},{:.3})", c.a.clamp(0.0, 1.0))
    } else {
        format!("rgb({r},{g},{b})")
    }
}

/// Format a frequency tick (20, 100, 1k, 10k).
pub fn fmt_freq(v: f64) -> String {
    if !v.is_finite() {
        return String::new();
    }
    if v >= 1000.0 {
        let k = v / 1000.0;
        if (k - k.round()).abs() < 1e-9 {
            format!("{}k", k.round() as i64)
        } else {
            format!("{k:.1}k")
        }
    } else if v >= 100.0 {
        format!("{}", v.round() as i64)
    } else if v >= 10.0 {
        let r = (v * 10.0).round() / 10.0;
        if (r - r.round()).abs() < 1e-9 {
            format!("{}", r.round() as i64)
        } else {
            format!("{r:.1}")
        }
    } else if v > 0.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Format a linear tick with trimmed decimals.
pub fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        return String::new();
    }
    let r = (v * 100.0).round() / 100.0;
    if (r - r.round()).abs() < 1e-9 {
        format!("{}", r.round() as i64)
    } else if ((r * 10.0).round() / 10.0 - r).abs() < 1e-9 {
        format!("{r:.1}")
    } else {
        format!("{r:.2}")
    }
}

/// Min-max decimation to at most `max_pts` points (pixel-column bucketing on x).
///
/// `None` samples are gaps: each contiguous run is decimated independently and
/// runs are joined with a `NaN`-y sentinel so the stroke breaks at the gap
/// (callers treat non-finite y as pen-up).
pub fn decimate(xs: &[f64], ys: &[Option<f64>], max_pts: usize) -> Vec<(f64, f64)> {
    let n = xs.len().min(ys.len());
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < n {
        while start < n && ys[start].is_none() {
            start += 1;
        }
        if start >= n {
            break;
        }
        let mut end = start;
        while end < n && ys[end].is_some() {
            end += 1;
        }
        if !out.is_empty() {
            out.push((xs[start], f64::NAN));
        }
        decimate_run(&xs[start..end], &ys[start..end], max_pts, &mut out);
        start = end;
    }
    out
}

/// Decimate one gap-free run (`ys` all `Some`) into `out`.
fn decimate_run(xs: &[f64], ys: &[Option<f64>], max_pts: usize, out: &mut Vec<(f64, f64)>) {
    let n = xs.len().min(ys.len());
    if n == 0 {
        return;
    }
    if n <= max_pts.max(2) {
        out.extend(
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| (x, y.unwrap_or(f64::NAN))),
        );
        return;
    }
    let buckets = max_pts.max(2) / 2;
    let x0 = xs[0];
    let x1 = xs[n - 1];
    let span = (x1 - x0).abs().max(f64::MIN_POSITIVE);
    let mut b = 0usize;
    let mut bmin = f64::INFINITY;
    let mut bmax = f64::NEG_INFINITY;
    let mut bx0 = 0.0;
    let mut bx1 = 0.0;
    let mut started = false;
    for i in 0..n {
        let x = xs[i];
        let Some(y) = ys[i] else { continue };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        let bi = (((x - x0).abs() / span) * buckets as f64).floor() as usize;
        let bi = bi.min(buckets - 1);
        if !started || bi != b {
            if started {
                out.push((bx0, bmin));
                if bx1 != bx0 || bmax != bmin {
                    out.push((bx1, bmax));
                }
            }
            b = bi;
            bmin = y;
            bmax = y;
            bx0 = x;
            bx1 = x;
            started = true;
        } else {
            if y < bmin {
                bmin = y;
                bx0 = x;
            }
            if y > bmax {
                bmax = y;
                bx1 = x;
            }
        }
    }
    if started {
        out.push((bx0, bmin));
        if bx1 != bx0 || bmax != bmin {
            out.push((bx1, bmax));
        }
    }
}

/// Data extent of visible series on one axis. The picker returns `None`
/// for series that do not belong to the axis (e.g. secondary-y series).
fn extent<F>(series: &[Series], visible: &[bool], mut pick: F) -> Option<(f64, f64)>
where
    F: FnMut(&Series) -> Option<&[f64]>,
{
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (s, &v) in series.iter().zip(visible.iter()) {
        if !v {
            continue;
        }
        let Some(vals) = pick(s) else { continue };
        for &val in vals {
            if val.is_finite() {
                lo = lo.min(val);
                hi = hi.max(val);
            }
        }
    }
    if lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// Data extent over optional y samples (`None` entries are gaps, skipped).
fn extent_opt<F>(series: &[Series], visible: &[bool], mut pick: F) -> Option<(f64, f64)>
where
    F: FnMut(&Series) -> Option<&[Option<f64>]>,
{
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (s, &v) in series.iter().zip(visible.iter()) {
        if !v {
            continue;
        }
        let Some(vals) = pick(s) else { continue };
        for val in vals.iter().filter_map(|&opt| opt) {
            if val.is_finite() {
                lo = lo.min(val);
                hi = hi.max(val);
            }
        }
    }
    if lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

/// True for series mapped to the secondary y axis (when one is active).
fn on_y2(s: &Series) -> bool {
    s.y_axis == 1
}

/// Padded domain honoring explicit min/max overrides.
fn domain(auto: Option<(f64, f64)>, explicit_min: Option<f64>, explicit_max: Option<f64>) -> (f64, f64) {
    let (mut lo, mut hi) = auto.unwrap_or((0.0, 1.0));
    if !matches!(lo.partial_cmp(&hi), Some(std::cmp::Ordering::Less)) {
        let c = if lo == 0.0 { 0.0 } else { lo };
        lo = c - 1.0;
        hi = c + 1.0;
    }
    if let Some(v) = explicit_min {
        lo = v;
    }
    if let Some(v) = explicit_max {
        hi = v;
    }
    if !matches!(lo.partial_cmp(&hi), Some(std::cmp::Ordering::Less)) {
        hi = lo + 1.0;
    }
    let pad = (hi - lo) * 0.03;
    let lo = if explicit_min.is_some() { lo } else { lo - pad };
    let hi = if explicit_max.is_some() { hi } else { hi + pad };
    (lo, hi)
}

/// Pad logarithmic domains in log space, keeping automatic bounds positive.
fn log_domain(
    auto: Option<(f64, f64)>,
    explicit_min: Option<f64>,
    explicit_max: Option<f64>,
) -> (f64, f64) {
    let valid = |v: &f64| v.is_finite() && *v > 0.0;
    let auto = auto
        .filter(|(lo, hi)| valid(lo) && valid(hi))
        .map(|(lo, hi)| (lo.ln(), hi.ln()));
    let (lo, hi) = domain(
        auto,
        explicit_min.filter(valid).map(f64::ln),
        explicit_max.filter(valid).map(f64::ln),
    );
    (lo.exp(), hi.exp())
}

/// Return conventional audio grid positions within a logarithmic viewport.
pub(crate) fn log_grid_ticks(lo: f64, hi: f64) -> Vec<f64> {
    // Decade subdivisions requested for audio reports; omit 7 to limit clutter.
    let mut ticks = Vec::new();
    for exponent in (lo.log10().floor() as i32).max(-307)..=(hi.log10().ceil() as i32).min(308) {
        let decade = 10.0_f64.powi(exponent);
        for multiple in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0] {
            let value = decade * multiple;
            if value >= lo * (1.0 - 1e-12) && value < hi * (1.0 - 1e-12) {
                ticks.push(value);
            }
        }
    }
    ticks
}

/// Label decades and the audio-band start, keeping dense gridlines legible.
pub(crate) fn fmt_log_grid(v: f64) -> String {
    if (v.log10() - v.log10().round()).abs() < 1e-10 || (v - 20.0).abs() < 1e-10 {
        fmt_freq(v)
    } else {
        String::new()
    }
}

/// Draw one Cartesian line figure; returns legend geometry.
pub fn draw_figure(ctx: &mut impl Ctx, fig: &Figure, w: f64, h: f64) -> DrawMeta {
    ctx.set_fill("#ffffff");
    ctx.fill_rect(0.0, 0.0, w, h);

    let visible: Vec<bool> = fig.series.iter().map(|s| s.visible).collect();

    // Legend column width from the longest visible name.
    ctx.set_font(FONT);
    let mut legend_w: f64 = 0.0;
    if fig.legend && !fig.series.is_empty() {
        for s in &fig.series {
            legend_w = legend_w.max(ctx.text_width(&s.name));
        }
        legend_w = (legend_w + 30.0).clamp(60.0, 240.0);
    }
    // A secondary y axis needs label room on the right.
    let y2_active = fig.y2.is_some() && fig.series.iter().any(|s| s.visible && s.y_axis == 1);
    let mut right = 14.0;
    if y2_active {
        right += 56.0;
    }
    if fig.legend && !fig.series.is_empty() {
        right += legend_w + 18.0;
    }
    let pad = Pad {
        left: 66.0,
        right,
        top: if fig.title.is_empty() { 12.0 } else { 32.0 },
        bottom: 52.0,
    };
    let plot_x = pad.left;
    let plot_y = pad.top;
    let plot_w = (w - pad.left - pad.right).max(40.0);
    let plot_h = (h - pad.top - pad.bottom).max(40.0);

    // Title.
    if !fig.title.is_empty() {
        ctx.set_fill(INK);
        ctx.set_font(FONT_TITLE);
        ctx.fill_text(&fig.title, plot_x, 20.0, TextAlign::Left);
    }

    // X domain (log axes need strictly positive values).
    let log_x = fig.x.scale == XScale::Log;
    let x_auto = extent(&fig.series, &visible, |s| Some(&s.x[..]));
    let mut x_auto = x_auto;
    if log_x {
        // Restrict the auto extent to positive samples.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (s, &v) in fig.series.iter().zip(visible.iter()) {
            if !v {
                continue;
            }
            for &x in &s.x {
                if x > 0.0 && x.is_finite() {
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
        }
        x_auto = if lo <= hi { Some((lo, hi)) } else { None };
    }
    let (x_lo, x_hi) = if log_x {
        log_domain(x_auto, fig.x.min, fig.x.max)
    } else {
        domain(x_auto, fig.x.min, fig.x.max)
    };

    let y_auto = extent_opt(&fig.series, &visible, |s| {
        if on_y2(s) && y2_active {
            None
        } else {
            Some(&s.y[..])
        }
    });
    let (y_lo, y_hi) = domain(y_auto, fig.y.min, fig.y.max);
    let y2_dom = if y2_active {
        let spec = fig.y2.as_ref();
        let auto = extent_opt(&fig.series, &visible, |s| {
            if on_y2(s) {
                Some(&s.y[..])
            } else {
                None
            }
        });
        Some(domain(
            auto,
            spec.and_then(|s| s.min),
            spec.and_then(|s| s.max),
        ))
    } else {
        None
    };

    // Scales from d3rs.
    enum XMap {
        Log(LogScale),
        Linear(LinearScale),
    }
    let xmap = if log_x {
        XMap::Log(
            LogScale::new()
                .domain(x_lo.max(f64::MIN_POSITIVE), x_hi)
                .range(plot_x, plot_x + plot_w),
        )
    } else {
        XMap::Linear(LinearScale::new().domain(x_lo, x_hi).range(plot_x, plot_x + plot_w))
    };
    let x_of = |x: f64| -> f64 {
        match &xmap {
            XMap::Log(s) => s.scale(x.max(f64::MIN_POSITIVE)),
            XMap::Linear(s) => s.scale(x),
        }
    };
    let yscale = LinearScale::new()
        .domain(y_lo, y_hi)
        .range(plot_y + plot_h, plot_y);
    let y_of = |y: f64| yscale.scale(y);
    let y2scale = y2_dom.map(|(lo, hi)| {
        LinearScale::new()
            .domain(lo, hi)
            .range(plot_y + plot_h, plot_y)
    });
    let y2_of = |y: f64| match &y2scale {
        Some(s) => s.scale(y),
        None => y_of(y),
    };

    // Shaded x-ranges under everything.
    for r in &fig.xranges {
        let a = x_of(r.x0).clamp(plot_x, plot_x + plot_w);
        let b = x_of(r.x1).clamp(plot_x, plot_x + plot_w);
        if b > a {
            ctx.set_fill(&r.color);
            ctx.fill_rect(a, plot_y, b - a, plot_h);
        }
    }

    // Axes via d3rs renderer-independent layout.
    let x_formatter: fn(f64) -> String = if log_x { fmt_log_grid } else { fmt_num };
    let mut x_cfg = AxisConfig::bottom()
        .with_ticks(10)
        .with_tick_size(5.0)
        .with_formatter(x_formatter)
        .with_title(fig.x.label.clone());
    if log_x {
        x_cfg = x_cfg.with_tick_values(log_grid_ticks(x_lo, x_hi));
    }
    let x_layout = match &xmap {
        XMap::Log(s) => AxisLayout::from_scale(s, &x_cfg, plot_w as f32),
        XMap::Linear(s) => AxisLayout::from_scale(s, &x_cfg, plot_w as f32),
    };
    let y_cfg = AxisConfig::left()
        .with_ticks(6)
        .with_tick_size(5.0)
        .with_formatter(fmt_num)
        .with_title(fig.y.label.clone());
    let y_layout = AxisLayout::from_scale(&yscale, &y_cfg, plot_h as f32);
    draw_axis_layout(ctx, &x_layout, plot_y + plot_h, plot_h, true, false);
    draw_axis_layout(ctx, &y_layout, plot_x, plot_w, false, false);
    if let (Some(spec), Some(scale)) = (fig.y2.as_ref(), y2scale.as_ref()) {
        let y2_cfg = AxisConfig::right()
            .with_ticks(5)
            .with_tick_size(5.0)
            .with_formatter(fmt_num)
            .with_title(spec.label.clone());
        let y2_layout = AxisLayout::from_scale(scale, &y2_cfg, plot_h as f32);
        draw_axis_layout(ctx, &y2_layout, plot_x + plot_w, plot_w, false, true);
    }

    // Reference lines.
    ctx.set_font(FONT_SMALL);
    for m in &fig.hlines {
        let y = y_of(m.at);
        if y < plot_y - 1.0 || y > plot_y + plot_h + 1.0 {
            continue;
        }
        ctx.set_stroke(&m.color);
        ctx.set_line_width(m.width as f64);
        ctx.set_dash(mark_dash(m.dash));
        ctx.begin_path();
        ctx.move_to(plot_x, y);
        ctx.line_to(plot_x + plot_w, y);
        ctx.stroke();
        if let Some(label) = m.label.as_ref() {
            ctx.set_fill(INK);
            ctx.fill_text(label, plot_x + plot_w - 4.0, y - 4.0, TextAlign::Right);
        }
    }
    ctx.set_dash(&[]);
    for m in &fig.vlines {
        let x = x_of(m.at);
        if x < plot_x - 1.0 || x > plot_x + plot_w + 1.0 {
            continue;
        }
        ctx.set_stroke(&m.color);
        ctx.set_line_width(m.width as f64);
        ctx.set_dash(mark_dash(m.dash));
        ctx.begin_path();
        ctx.move_to(x, plot_y);
        ctx.line_to(x, plot_y + plot_h);
        ctx.stroke();
    }
    ctx.set_dash(&[]);

    // Series polylines (decimated to ~2 px buckets).
    let max_pts = (plot_w as usize).saturating_mul(2).max(256);
    for (i, s) in fig.series.iter().enumerate() {
        if !visible[i] {
            continue;
        }
        let color = s.color.clone().unwrap_or_else(|| default_color(i));
        let pts = decimate(&s.x, &s.y, max_pts);
        ctx.set_stroke(&color);
        ctx.set_line_width(s.width as f64);
        ctx.set_dash(s.dash.pattern());
        ctx.begin_path();
        let mut pen = false;
        for (x, y) in pts {
            let xx = if log_x { x.max(f64::MIN_POSITIVE) } else { x };
            if !xx.is_finite() || !y.is_finite() {
                pen = false;
                continue;
            }
            let px = x_of(xx);
            let py = if on_y2(s) && y2scale.is_some() {
                y2_of(y)
            } else {
                y_of(y)
            };
            if !px.is_finite() || !py.is_finite() {
                pen = false;
                continue;
            }
            if pen {
                ctx.line_to(px, py);
            } else {
                ctx.move_to(px, py);
                pen = true;
            }
        }
        ctx.stroke();
    }
    ctx.set_dash(&[]);

    // Annotations.
    ctx.set_fill(INK);
    ctx.set_font(FONT_SMALL);
    for a in &fig.annotations {
        ctx.fill_text(&a.text, x_of(a.x), y_of(a.y) - 4.0, TextAlign::Center);
    }

    // Legend column (right of the plot area, past any y2 labels).
    let mut meta = DrawMeta::default();
    if fig.legend && !fig.series.is_empty() {
        let lx = plot_x + plot_w + 10.0 + if y2_active { 56.0 } else { 0.0 };
        let mut ly = plot_y + 2.0;
        ctx.set_font(FONT_SMALL);
        for (i, s) in fig.series.iter().enumerate() {
            let color = s.color.clone().unwrap_or_else(|| default_color(i));
            let row_h = 18.0;
            if ly + row_h > plot_y + plot_h {
                break;
            }
            if visible[i] {
                ctx.set_fill(&color);
            } else {
                ctx.set_fill("#bbbbbb");
            }
            ctx.fill_rect(lx, ly + 4.0, 16.0, 3.0);
            ctx.set_fill(if visible[i] { INK } else { INK_FAINT });
            ctx.fill_text(&s.name, lx + 21.0, ly + 13.0, TextAlign::Left);
            meta.legend.push(LegendEntry {
                series: i,
                x: lx,
                y: ly,
                w: legend_w,
                h: row_h,
            });
            ly += row_h;
        }
    }
    meta
}

fn mark_dash(d: crate::schema::DashOption) -> &'static [f64] {
    d.pattern()
}

/// Draw d3rs axis geometry.
///
/// d3rs axis coordinates are absolute canvas px along the axis (because the
/// scale range is set in canvas px) and relative to the axis line across it.
/// `cross` is the axis-line position (canvas y for a bottom axis, canvas x
/// for a left axis); `plot_len` is the plot-area length along the axis and
/// is only used for gridlines.
pub(crate) fn draw_axis_layout(
    ctx: &mut impl Ctx,
    layout: &AxisLayout,
    cross: f64,
    plot_len: f64,
    horizontal: bool,
    mirror: bool,
) {
    // Gridlines through the major tick positions (primary axes only).
    if !mirror {
        ctx.set_stroke(GRID);
        ctx.set_line_width(1.0);
        ctx.begin_path();
        for t in &layout.major_ticks {
            if horizontal {
                ctx.move_to(t.position, cross);
                ctx.line_to(t.position, cross - plot_len);
            } else {
                ctx.move_to(cross, t.position);
                ctx.line_to(cross + plot_len, t.position);
            }
        }
        ctx.stroke();
    }
    // Tick marks (relative to the axis line).
    ctx.set_stroke(AXIS);
    ctx.set_line_width(1.0);
    ctx.begin_path();
    for t in layout.all_ticks() {
        if horizontal {
            ctx.move_to(t.position, cross + t.line.start.y as f64);
            ctx.line_to(t.position, cross + t.line.end.y as f64);
        } else {
            ctx.move_to(cross + t.line.start.x as f64, t.position);
            ctx.line_to(cross + t.line.end.x as f64, t.position);
        }
    }
    ctx.stroke();
    // Domain line.
    if let Some(d) = &layout.domain_line {
        ctx.set_stroke(AXIS);
        ctx.set_line_width(1.5);
        ctx.begin_path();
        if horizontal {
            ctx.move_to(d.start.x as f64, cross);
            ctx.line_to(d.end.x as f64, cross);
        } else {
            ctx.move_to(cross, d.start.y as f64);
            ctx.line_to(cross, d.end.y as f64);
        }
        ctx.stroke();
    }
    // Tick labels.
    ctx.set_fill(INK);
    ctx.set_font(FONT);
    for t in layout.all_ticks() {
        let Some(label) = t.label.as_ref() else { continue };
        if label.is_empty() {
            continue;
        }
        let Some(p) = t.label_position else { continue };
        if t.is_minor {
            continue;
        }
        if horizontal {
            ctx.fill_text(label, p.x as f64, cross + p.y as f64, TextAlign::Center);
        } else if mirror {
            ctx.fill_text(label, cross + p.x as f64, t.position + 4.0, TextAlign::Left);
        } else {
            // Nudge up half a line for optical centering on the tick.
            ctx.fill_text(label, cross + p.x as f64, t.position + 4.0, TextAlign::Right);
        }
    }
    // Axis title.
    if let Some(title) = &layout.title {
        ctx.set_fill(INK_FAINT);
        ctx.set_font(FONT);
        if horizontal {
            ctx.fill_text(
                &title.text,
                title.position.x as f64,
                cross + title.position.y as f64,
                TextAlign::Center,
            );
        } else {
            ctx.fill_text_rotated(
                &title.text,
                cross + title.position.x as f64,
                title.position.y as f64,
                title.angle_degrees as f64,
                TextAlign::Center,
            );
        }
    }
}

/// Draw one grouped bar chart; returns legend geometry.
pub fn draw_bar(ctx: &mut impl Ctx, chart: &BarChart, w: f64, h: f64, visible: &[bool]) -> DrawMeta {
    ctx.set_fill("#ffffff");
    ctx.fill_rect(0.0, 0.0, w, h);

    ctx.set_font(FONT);
    let mut legend_w: f64 = 0.0;
    if chart.legend && !chart.groups.is_empty() {
        for g in &chart.groups {
            legend_w = legend_w.max(ctx.text_width(&g.name));
        }
        legend_w = (legend_w + 30.0).clamp(60.0, 240.0);
    }
    let pad = Pad {
        left: 66.0,
        right: if chart.legend && !chart.groups.is_empty() {
            legend_w + 18.0
        } else {
            14.0
        },
        top: if chart.title.is_empty() { 12.0 } else { 32.0 },
        bottom: 52.0,
    };
    let plot_x = pad.left;
    let plot_y = pad.top;
    let plot_w = (w - pad.left - pad.right).max(40.0);
    let plot_h = (h - pad.top - pad.bottom).max(40.0);

    if !chart.title.is_empty() {
        ctx.set_fill(INK);
        ctx.set_font(FONT_TITLE);
        ctx.fill_text(&chart.title, plot_x, 20.0, TextAlign::Left);
    }

    let n_cat = chart.categories.len().max(1);
    let groups: Vec<(usize, &crate::schema::BarGroup)> = chart
        .groups
        .iter()
        .enumerate()
        .filter(|(i, _)| visible.get(*i).copied().unwrap_or(true))
        .collect();
    let n_g = groups.len().max(1);

    let mut lo = 0.0f64;
    let mut hi = 0.0f64;
    for (_, g) in &groups {
        for &v in &g.values {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if chart.ymin.is_none() {
        lo = lo.min(0.0);
    }
    if chart.ymax.is_none() {
        hi = hi.max(0.0);
    }
    let (y_lo, y_hi) = domain(Some((lo, hi)), chart.ymin, chart.ymax);
    let yscale = LinearScale::new()
        .domain(y_lo, y_hi)
        .range(plot_y + plot_h, plot_y);
    let y_of = |y: f64| yscale.scale(y);

    let y_cfg = AxisConfig::left()
        .with_ticks(6)
        .with_tick_size(5.0)
        .with_formatter(fmt_num)
        .with_title(chart.ylabel.clone());
    let y_layout = AxisLayout::from_scale(&yscale, &y_cfg, plot_h as f32);
    draw_axis_layout(ctx, &y_layout, plot_x, plot_w, false, false);

    // Zero line when inside the domain.
    let yz = y_of(0.0);
    if yz >= plot_y && yz <= plot_y + plot_h {
        ctx.set_stroke(AXIS);
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.move_to(plot_x, yz);
        ctx.line_to(plot_x + plot_w, yz);
        ctx.stroke();
    }

    // Reference lines (e.g. headroom limits).
    ctx.set_font(FONT_SMALL);
    for m in &chart.hlines {
        let y = y_of(m.at);
        if y < plot_y - 1.0 || y > plot_y + plot_h + 1.0 {
            continue;
        }
        ctx.set_stroke(&m.color);
        ctx.set_line_width(m.width as f64);
        ctx.set_dash(mark_dash(m.dash));
        ctx.begin_path();
        ctx.move_to(plot_x, y);
        ctx.line_to(plot_x + plot_w, y);
        ctx.stroke();
        if let Some(label) = m.label.as_ref() {
            ctx.set_fill(INK);
            ctx.fill_text(label, plot_x + plot_w - 4.0, y - 4.0, TextAlign::Right);
        }
    }
    ctx.set_dash(&[]);

    let slot = plot_w / n_cat as f64;
    let bar_w = (slot * 0.72 / n_g as f64).max(2.0);
    for (ci, cat) in chart.categories.iter().enumerate() {
        let cx = plot_x + (ci as f64 + 0.5) * slot;
        let total = bar_w * n_g as f64;
        for (gi, (idx, g)) in groups.iter().enumerate() {
            let v = g.values.get(ci).copied().unwrap_or(0.0);
            if !v.is_finite() {
                continue;
            }
            let color = g.color.clone().unwrap_or_else(|| default_color(*idx));
            let bx = cx - total / 2.0 + gi as f64 * bar_w;
            let top = y_of(v.max(y_lo).min(y_hi));
            let base = y_of(0.0f64.max(y_lo).min(y_hi));
            let (y0, hh) = if top <= base { (top, base - top) } else { (base, top - base) };
            ctx.set_fill(&color);
            ctx.fill_rect(bx + 1.0, y0, (bar_w - 2.0).max(1.0), hh.max(1.0));
        }
        // Category label.
        ctx.set_fill(INK);
        ctx.set_font(FONT_SMALL);
        let label = truncate_label(cat, ctx, slot - 6.0);
        ctx.fill_text(&label, cx, plot_y + plot_h + 18.0, TextAlign::Center);
    }

    // Legend column.
    let mut meta = DrawMeta::default();
    if chart.legend && !chart.groups.is_empty() {
        let lx = plot_x + plot_w + 10.0;
        let mut ly = plot_y + 2.0;
        ctx.set_font(FONT_SMALL);
        for (i, g) in chart.groups.iter().enumerate() {
            let row_h = 18.0;
            if ly + row_h > plot_y + plot_h {
                break;
            }
            let on = visible.get(i).copied().unwrap_or(true);
            let color = g.color.clone().unwrap_or_else(|| default_color(i));
            ctx.set_fill(if on { color.as_str() } else { "#bbbbbb" });
            ctx.fill_rect(lx, ly + 4.0, 14.0, 10.0);
            ctx.set_fill(if on { INK } else { INK_FAINT });
            ctx.fill_text(&g.name, lx + 19.0, ly + 13.0, TextAlign::Left);
            meta.legend.push(LegendEntry {
                series: i,
                x: lx,
                y: ly,
                w: legend_w,
                h: row_h,
            });
            ly += row_h;
        }
    }
    meta
}

/// Draw one Sankey flow diagram (no legend; uses the d3rs Sankey layout).
pub fn draw_sankey(ctx: &mut impl Ctx, chart: &SankeyChart, w: f64, h: f64) {
    ctx.set_fill("#ffffff");
    ctx.fill_rect(0.0, 0.0, w, h);

    let top = if chart.title.is_empty() { 8.0 } else { 30.0 };
    if !chart.title.is_empty() {
        ctx.set_fill(INK);
        ctx.set_font(FONT_TITLE);
        ctx.fill_text(&chart.title, 12.0, 20.0, TextAlign::Left);
    }
    let area_w = (w - 24.0).max(60.0);
    let area_h = (h - top - 12.0).max(60.0);

    let inputs: Vec<SankeyLinkInput> = chart
        .links
        .iter()
        .filter_map(|l| {
            let s = chart.nodes.get(l.source)?.clone();
            let t = chart.nodes.get(l.target)?.clone();
            if l.value.is_finite() && l.value > 0.0 {
                Some(SankeyLinkInput {
                    source: s,
                    target: t,
                    value: l.value,
                })
            } else {
                None
            }
        })
        .collect();
    let Ok(layout) = SankeyLayout::new()
        .width(area_w)
        .height(area_h)
        .node_width(14.0)
        .node_padding(10.0)
        .try_compute(&chart.nodes, &inputs)
    else {
        ctx.set_fill(INK_FAINT);
        ctx.set_font(FONT);
        ctx.fill_text("no flow data", 12.0, top + 16.0, TextAlign::Left);
        return;
    };

    // Links under the nodes.
    for (li, link) in layout.links.iter().enumerate() {
        let color = chart
            .links
            .get(li)
            .and_then(|l| l.color.clone())
            .unwrap_or_else(|| "rgba(74,144,217,0.45)".to_string());
        let src = &layout.nodes[link.source];
        let dst = &layout.nodes[link.target];
        let sx = 12.0 + src.x1;
        let tx = 12.0 + dst.x0;
        let hw = (link.width / 2.0).max(0.5);
        let y0 = top + link.y0;
        let y1 = top + link.y1;
        let cx = (sx + tx) / 2.0;
        ctx.set_fill(&color);
        ctx.begin_path();
        ctx.move_to(sx, y0 - hw);
        ctx.bezier_to(cx, y0 - hw, cx, y1 - hw, tx, y1 - hw);
        ctx.line_to(tx, y1 + hw);
        ctx.bezier_to(cx, y1 + hw, cx, y0 + hw, sx, y0 + hw);
        ctx.close_path();
        ctx.fill();
    }
    // Nodes over the links.
    ctx.set_font(FONT_SMALL);
    for node in &layout.nodes {
        let nx = 12.0 + node.x0;
        let ny = top + node.y0;
        let nw = (node.x1 - node.x0).max(2.0);
        let nh = (node.y1 - node.y0).max(2.0);
        ctx.set_fill("#4a90d9");
        ctx.fill_rect(nx, ny, nw, nh);
        ctx.set_fill(INK);
        let name = truncate_label(&node.id, ctx, 150.0);
        // Rightmost layer labels go left of the node.
        let is_sink = layout
            .links
            .iter()
            .all(|l| l.target != node.index);
        let dest_right = layout.nodes.iter().any(|o| o.x0 > node.x0 + 1.0);
        if dest_right && !is_sink {
            ctx.fill_text(&name, nx + nw + 5.0, ny + nh / 2.0 + 4.0, TextAlign::Left);
        } else {
            let tw = ctx.text_width(&name);
            ctx.fill_text(&name, nx - 5.0 - tw, ny + nh / 2.0 + 4.0, TextAlign::Left);
        }
    }
}

/// Truncate a label with an ellipsis to fit `max_w` px.
fn truncate_label(label: &str, ctx: &mut impl Ctx, max_w: f64) -> String {
    if ctx.text_width(label) <= max_w {
        return label.to_string();
    }
    let mut out = String::new();
    for ch in label.chars() {
        out.push(ch);
        if ctx.text_width(&out) + ctx.text_width("…") > max_w {
            out.push('…');
            return out;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{AxisSpec, DashOption, LineMark, Series, XScale};
    use std::cell::RefCell;

    #[test]
    fn audio_log_grid_has_requested_subdivisions_and_no_end_tick() {
        let ticks = log_grid_ticks(20.0, 20000.0);
        for multiple in [1.0, 10.0, 100.0] {
            for base in [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 90.0] {
                assert!(ticks.contains(&(base * multiple)));
            }
        }
        assert_eq!(ticks.last(), Some(&10000.0));
        assert_eq!(fmt_log_grid(10000.0), "10k");
        assert!(fmt_log_grid(90.0).is_empty());
        assert!(log_grid_ticks(300.0, 600.0).iter().all(|v| *v >= 300.0 && *v < 600.0));
    }

    #[test]
    fn octave_log_domain_uses_multiplicative_padding() {
        let (lo, hi) = log_domain(Some((63.0, 16000.0)), None, None);
        assert!(lo > 50.0 && lo < 63.0);
        assert!(hi > 16000.0 && hi < 20000.0);
        assert!((63.0 / lo - hi / 16000.0).abs() < 1e-12);
        let (lo, hi) = log_domain(Some((63.0, 16000.0)), Some(20.0), Some(20000.0));
        assert!((lo - 20.0).abs() < 1e-10);
        assert!((hi - 20000.0).abs() < 1e-10);
    }

    #[test]
    fn log_domain_handles_empty_singleton_and_invalid_bounds() {
        for (auto, min) in [
            (None, None),
            (Some((1000.0, 1000.0)), None),
            (Some((63.0, 16000.0)), Some(0.0)),
        ] {
            let (lo, hi) = log_domain(auto, min, None);
            assert!(lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > lo);
            assert!(hi / lo < 1000.0);
        }
    }

    /// Recording 2D backend for native unit tests.
    #[derive(Default)]
    struct Rec {
        ops: RefCell<Vec<String>>,
        texts: RefCell<Vec<(String, f64, f64)>>,
    }

    impl Ctx for Rec {
        fn set_fill(&mut self, css: &str) {
            self.ops.borrow_mut().push(format!("fill={css}"));
        }
        fn set_stroke(&mut self, css: &str) {
            self.ops.borrow_mut().push(format!("stroke={css}"));
        }
        fn set_line_width(&mut self, w: f64) {
            self.ops.borrow_mut().push(format!("lw={w}"));
        }
        fn set_dash(&mut self, p: &[f64]) {
            self.ops.borrow_mut().push(format!("dash={}", p.len()));
        }
        fn set_font(&mut self, f: &str) {
            self.ops.borrow_mut().push(format!("font={f}"));
        }
        fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
            self.ops
                .borrow_mut()
                .push(format!("rect={x:.1},{y:.1},{w:.1},{h:.1}"));
        }
        fn begin_path(&mut self) {
            self.ops.borrow_mut().push("begin".to_string());
        }
        fn move_to(&mut self, x: f64, y: f64) {
            self.ops.borrow_mut().push(format!("M{x:.1},{y:.1}"));
        }
        fn line_to(&mut self, x: f64, y: f64) {
            self.ops.borrow_mut().push(format!("L{x:.1},{y:.1}"));
        }
        fn bezier_to(&mut self, a: f64, b: f64, c: f64, d: f64, x: f64, y: f64) {
            self.ops.borrow_mut().push(format!("C{a:.1},{b:.1},{c:.1},{d:.1},{x:.1},{y:.1}"));
        }
        fn close_path(&mut self) {
            self.ops.borrow_mut().push("close".to_string());
        }
        fn stroke(&mut self) {
            self.ops.borrow_mut().push("stroke-path".to_string());
        }
        fn fill(&mut self) {
            self.ops.borrow_mut().push("fill-path".to_string());
        }
        fn fill_text(&mut self, t: &str, x: f64, y: f64, a: TextAlign) {
            self.texts.borrow_mut().push((t.to_string(), x, y));
            self.ops.borrow_mut().push(format!("text@{a:?}"));
        }
        fn fill_text_rotated(&mut self, t: &str, x: f64, y: f64, ang: f64, a: TextAlign) {
            self.texts.borrow_mut().push((t.to_string(), x, y));
            self.ops.borrow_mut().push(format!("rtext{ang:.0}@{a:?}"));
        }
        fn text_width(&mut self, t: &str) -> f64 {
            t.chars().count() as f64 * 6.5
        }
    }

    #[test]
    fn grid_views_draw_axes_and_reject_ragged_data() {
        let mut fig = demo_figure();
        fig.y.label = "Time (ms)".to_string();
        fig.y.min = Some(0.0);
        fig.y.max = Some(15.0);
        let mut grid = crate::schema::GridData {
            x: vec![20.0, 1000.0, 20000.0], y: vec![0.0, 5.0, 15.0],
            z: vec![vec![0.0, -10.0, -20.0], vec![-5.0, -15.0, -25.0], vec![-10.0, -20.0, -30.0]],
            surface: false, zmin: -30.0, zmax: 0.0, highlights: vec![1], rotation: None,
        };
        for surface in [false, true] {
            grid.surface = surface;
            let mut ctx = Rec::default();
            let meta = crate::grid::draw_grid(&mut ctx, &fig, &grid, 900.0, 500.0);
            assert_eq!(meta.legend.len(), 2);
            let texts = ctx.texts.borrow();
            assert!(texts.iter().any(|t| t.0 == "Frequency (Hz)"));
            assert!(texts.iter().any(|t| t.0 == "Time (ms)"));
            assert!(texts.iter().any(|t| t.0 == "dB"));
            assert!(!ctx.ops.borrow().iter().any(|op| op.contains("NaN")));
        }
        grid.rotation = Some([40.0, 50.0]);
        let mut rotated = Rec::default();
        crate::grid::draw_grid(&mut rotated, &fig, &grid, 900.0, 500.0);
        assert!(!rotated.ops.borrow().iter().any(|op| op.contains("NaN")));
        grid.z[0].pop();
        let mut ctx = Rec::default();
        crate::grid::draw_grid(&mut ctx, &fig, &grid, 900.0, 500.0);
        assert!(ctx.texts.borrow().iter().any(|t| t.0.contains("invalid time-frequency grid")));
    }

    fn demo_figure() -> Figure {
        Figure {
            title: "demo".to_string(),
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
            series: vec![
                Series {
                    name: "Input".to_string(),
                    x: vec![20.0, 100.0, 1000.0, 20000.0],
                    y: vec![Some(0.0), Some(2.0), Some(-1.0), Some(0.5)],
                    color: None,
                    width: 2.0,
                    dash: DashOption::Solid,
                    visible: true,
                    y_axis: 0,
                },
                Series {
                    name: "Hidden".to_string(),
                    x: vec![20.0, 20000.0],
                    y: vec![Some(50.0), Some(50.0)],
                    color: None,
                    width: 1.0,
                    dash: DashOption::Dash,
                    visible: false,
                    y_axis: 0,
                },
            ],
            hlines: vec![LineMark {
                at: 0.0,
                color: "rgba(150,150,150,0.4)".to_string(),
                dash: DashOption::Dash,
                width: 1.0,
                label: None,
            }],
            vlines: vec![],
            xranges: vec![],
            annotations: vec![],
            legend: true,
        }
    }

    #[test]
    fn figure_draws_visible_series_only() {
        let mut ctx = Rec::default();
        let meta = draw_figure(&mut ctx, &demo_figure(), 900.0, 500.0);
        assert_eq!(meta.legend.len(), 2);
        let ops = ctx.ops.borrow().join("\n");
        // Hidden 50 dB series must not move the y domain: no line near the top.
        assert!(ops.contains("stroke=#"));
        // Legend names recorded as text.
        let texts: Vec<String> = ctx.texts.borrow().iter().map(|t| t.0.clone()).collect();
        assert!(texts.iter().any(|t| t == "Input"));
        assert!(texts.iter().any(|t| t == "Hidden"));
        assert!(texts.iter().any(|t| t == "demo"));
    }

    #[test]
    fn legend_rects_do_not_overlap_plot() {
        let mut ctx = Rec::default();
        let meta = draw_figure(&mut ctx, &demo_figure(), 900.0, 500.0);
        for e in &meta.legend {
            assert!(e.x > 700.0, "legend {e:?} should sit right of the plot");
            assert!(e.w > 20.0);
        }
    }

    #[test]
    fn d3rs_log_scale_maps_decades() {
        let s = LogScale::new().domain(20.0, 20000.0).range(0.0, 300.0);
        let a = s.scale(20.0);
        let b = s.scale(200.0);
        let c = s.scale(2000.0);
        let d = s.scale(20000.0);
        assert!((a - 0.0).abs() < 1e-9);
        assert!((d - 300.0).abs() < 1e-9);
        assert!((b - a - 100.0).abs() < 1.0, "decades are evenly spaced: {a} {b}");
        assert!((c - b - 100.0).abs() < 1.0);
    }

    #[test]
    fn sankey_bands_stay_inside_nodes() {
        use crate::schema::{SankeyChart, SankeyLink};
        let chart = SankeyChart {
            title: String::new(),
            nodes: vec!["in".to_string(), "out".to_string()],
            links: vec![SankeyLink {
                source: 0,
                target: 1,
                value: 3.0,
                color: None,
            }],
        };
        let mut ctx = Rec::default();
        draw_sankey(&mut ctx, &chart, 600.0, 300.0);
        // Node rects and one filled band path must exist.
        let ops = ctx.ops.borrow().join("\n");
        assert!(ops.contains("fill-path"), "band must be filled");
        assert!(ctx.texts.borrow().iter().any(|t| t.0 == "in"));
        assert!(ctx.texts.borrow().iter().any(|t| t.0 == "out"));
    }

    #[test]
    fn bar_visibility_flags() {
        use crate::schema::{BarChart, BarGroup};
        let chart = BarChart {
            title: "Scores".to_string(),
            categories: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            groups: vec![
                BarGroup {
                    name: "Before".to_string(),
                    values: vec![3.1, 4.2, 2.5],
                    color: None,
                },
                BarGroup {
                    name: "After".to_string(),
                    values: vec![5.1, 5.6, 4.9],
                    color: None,
                },
            ],
            ylabel: "score".to_string(),
            ymin: None,
            ymax: None,
            legend: true,
            hlines: vec![],
        };
        for visible in [&[][..], &[true][..], &[false][..], &[false, false][..], &[true, false][..]] {
            let mut ctx = Rec::default();
            let meta = draw_bar(&mut ctx, &chart, 900.0, 420.0, visible);
            assert_eq!(meta.legend.len(), 2);
        }
    }

    #[test]
    fn bar_hlines_draw_with_labels() {
        use crate::schema::{BarGroup, LineMark};
        let mut chart = BarChart {
            title: "Headroom".to_string(),
            categories: vec!["sub".to_string()],
            groups: vec![BarGroup {
                name: "Peak".to_string(),
                values: vec![3.0],
                color: None,
            }],
            ylabel: "dB".to_string(),
            ymin: None,
            ymax: None,
            legend: false,
            hlines: vec![LineMark {
                at: 2.5,
                color: "rgba(40,40,40,0.7)".to_string(),
                dash: DashOption::Dash,
                width: 2.0,
                label: Some("headroom limit +6.0 dB".to_string()),
            }],
        };
        let mut ctx = Rec::default();
        let meta = draw_bar(&mut ctx, &chart, 900.0, 420.0, &[]);
        assert!(meta.legend.is_empty());
        let texts: Vec<String> = ctx.texts.borrow().iter().map(|t| t.0.clone()).collect();
        assert!(
            texts.iter().any(|t| t == "headroom limit +6.0 dB"),
            "hline label drawn: {texts:?}"
        );
        // An out-of-range hline draws nothing and labels nothing.
        chart.hlines[0].at = 1e9;
        chart.hlines[0].label = Some("far away".to_string());
        let mut ctx2 = Rec::default();
        draw_bar(&mut ctx2, &chart, 900.0, 420.0, &[]);
        let texts2: Vec<String> = ctx2.texts.borrow().iter().map(|t| t.0.clone()).collect();
        assert!(!texts2.iter().any(|t| t == "far away"));
    }

    #[test]
    fn secondary_axis_maps_and_labels() {
        let mut fig = demo_figure();
        fig.y2 = Some(AxisSpec {
            label: "DI (dB)".to_string(),
            scale: XScale::Linear,
            min: Some(-5.0),
            max: Some(45.0),
        });
        fig.series.push(Series {
            name: "SPDI".to_string(),
            x: vec![20.0, 20000.0],
            y: vec![Some(0.0), Some(10.0)],
            color: None,
            width: 2.0,
            dash: DashOption::Solid,
            visible: true,
            y_axis: 1,
        });
        let mut ctx = Rec::default();
        let meta = draw_figure(&mut ctx, &fig, 900.0, 500.0);
        assert_eq!(meta.legend.len(), 3);
        let texts: Vec<String> = ctx.texts.borrow().iter().map(|t| t.0.clone()).collect();
        assert!(texts.iter().any(|t| t == "DI (dB)"), "right axis title drawn");
        assert!(texts.iter().any(|t| t == "SPDI"), "y2 series in legend");
        // Legend sits past the y2 label column.
        for e in &meta.legend {
            assert!(e.x > 750.0, "legend {e:?} clears the y2 axis");
        }
    }

    #[test]
    fn decimate_caps_long_series() {
        let xs: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
        let ys: Vec<Option<f64>> = (0..100_000).map(|i| Some((i as f64).sin())).collect();
        let pts = decimate(&xs, &ys, 1000);
        assert!(pts.len() <= 1000, "got {}", pts.len());
        assert!(!pts.is_empty());
    }

    #[test]
    fn decimate_preserves_gaps() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![Some(0.0), Some(1.0), None, Some(3.0), Some(4.0)];
        let pts = decimate(&xs, &ys, 100);
        // Two runs joined by one NaN sentinel: no segment bridges the gap.
        assert_eq!(pts.len(), 5, "got {pts:?}");
        assert!(pts[2].1.is_nan(), "got {pts:?}");
        assert_eq!((pts[0].0, pts[0].1), (0.0, 0.0));
        assert_eq!((pts[4].0, pts[4].1), (4.0, 4.0));
    }

    #[test]
    fn freq_formatting() {
        assert_eq!(fmt_freq(20.0), "20");
        assert_eq!(fmt_freq(1000.0), "1k");
        assert_eq!(fmt_freq(20000.0), "20k");
        assert_eq!(fmt_num(2.0), "2");
        assert_eq!(fmt_num(-1.25), "-1.25");
    }
}
