//! GPUI painting backend for AutoEQ's shared report charts.

mod view;

pub use view::{NativeReport, ReportView};

use autoeq_report_wasm::draw::{Ctx, DrawMeta, TextAlign, draw_bar, draw_figure, draw_sankey};
use autoeq_report_wasm::grid::draw_grid;
use autoeq_report_wasm::schema::Section;
use gpui::{
    App, Bounds, ContentMask, Font, FontWeight, Hsla, PathBuilder, Pixels, Point, Rgba, ShapedLine,
    TextRun, TransformationMatrix, Window, fill, point, px, size,
};

/// Paints a report chart and returns its plot and legend geometry.
///
/// Call this from a GPUI paint callback. Off-screen charts and HTML sections
/// return `None`.
pub fn paint_chart(
    section: &Section,
    bounds: Bounds<Pixels>,
    bar_visibility: &[bool],
    window: &mut Window,
    cx: &mut App,
) -> Option<DrawMeta> {
    let w = f64::from(f32::from(bounds.size.width));
    let h = f64::from(f32::from(bounds.size.height));
    if w <= 0.0 || h <= 0.0 || !window.content_mask().bounds.intersects(&bounds) {
        return None;
    }

    window.with_content_mask(Some(ContentMask { bounds }), |window| {
        let mut ctx = PaintContext::new(bounds.origin, window, cx);
        match section {
            Section::Figure { figure, .. } => Some(draw_figure(&mut ctx, figure, w, h)),
            Section::Grid { figure, grid, .. } => Some(draw_grid(&mut ctx, figure, grid, w, h)),
            Section::Bar { chart, .. } => Some(draw_bar(&mut ctx, chart, w, h, bar_visibility)),
            Section::Sankey { chart, .. } => {
                draw_sankey(&mut ctx, chart, w, h);
                Some(DrawMeta::default())
            }
            Section::Html { .. } => None,
        }
    })
}

enum Segment {
    Move(Point<Pixels>),
    Line(Point<Pixels>),
    Cubic {
        to: Point<Pixels>,
        control_a: Point<Pixels>,
        control_b: Point<Pixels>,
    },
    Close,
}

struct PaintContext<'a> {
    origin: Point<Pixels>,
    window: &'a mut Window,
    cx: &'a mut App,
    fill: Hsla,
    stroke: Hsla,
    width: Pixels,
    dash: Vec<Pixels>,
    font: Font,
    font_size: Pixels,
    path: Vec<Segment>,
}

impl<'a> PaintContext<'a> {
    fn new(origin: Point<Pixels>, window: &'a mut Window, cx: &'a mut App) -> Self {
        let style = window.text_style();
        let font_size = style.font_size.to_pixels(window.rem_size());
        Self {
            origin,
            window,
            cx,
            fill: style.color,
            stroke: style.color,
            width: px(1.0),
            dash: Vec::new(),
            font: style.font(),
            font_size,
            path: Vec::new(),
        }
    }

    fn point(&self, x: f64, y: f64) -> Point<Pixels> {
        self.origin + point(px(x as f32), px(y as f32))
    }

    fn paint_path(&mut self, mut builder: PathBuilder, color: Hsla) {
        for segment in &self.path {
            match *segment {
                Segment::Move(to) => builder.move_to(to),
                Segment::Line(to) => builder.line_to(to),
                Segment::Cubic {
                    to,
                    control_a,
                    control_b,
                } => {
                    builder.cubic_bezier_to(to, control_a, control_b);
                }
                Segment::Close => builder.close(),
            }
        }
        if let Ok(path) = builder.build() {
            self.window.paint_path(path, color);
        }
    }

    fn shape(&self, text: &str) -> ShapedLine {
        let text = text.replace(['\r', '\n'], " ");
        let run = TextRun {
            len: text.len(),
            font: self.font.clone(),
            color: self.fill,
            ..Default::default()
        };
        self.window
            .text_system()
            .shape_line(text.into(), self.font_size, &[run], None)
    }
}

fn color(css: &str) -> Option<Hsla> {
    Rgba::try_from(css)
        .ok()
        .or_else(|| {
            let [r, g, b, a] = crate::parse_css_color(css)?;
            Some(Rgba { r, g, b, a })
        })
        .map(Into::into)
}

impl Ctx for PaintContext<'_> {
    fn set_fill(&mut self, css: &str) {
        if let Some(color) = color(css) {
            self.fill = color;
        }
    }

    fn set_stroke(&mut self, css: &str) {
        if let Some(color) = color(css) {
            self.stroke = color;
        }
    }

    fn set_line_width(&mut self, width: f64) {
        self.width = px(width as f32);
    }

    fn set_dash(&mut self, pattern: &[f64]) {
        self.dash.clear();
        self.dash
            .extend(pattern.iter().map(|&value| px(value as f32)));
    }

    fn set_font(&mut self, css: &str) {
        // Shared chart fonts use CSS pixel sizes and an optional weight.
        let Some((prefix, _)) = css.split_once("px") else {
            return;
        };
        let mut tokens = prefix.split_ascii_whitespace().rev();
        if let Some(size) = tokens.next().and_then(|size| size.parse::<f32>().ok()) {
            self.font_size = px(size);
        }
        self.font.weight = if tokens.next() == Some("600") {
            FontWeight::SEMIBOLD
        } else {
            FontWeight::NORMAL
        };
    }

    fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
        let bounds = Bounds::new(self.point(x, y), size(px(w as f32), px(h as f32)));
        self.window.paint_quad(fill(bounds, self.fill));
    }

    fn begin_path(&mut self) {
        self.path.clear();
    }

    fn move_to(&mut self, x: f64, y: f64) {
        self.path.push(Segment::Move(self.point(x, y)));
    }

    fn line_to(&mut self, x: f64, y: f64) {
        self.path.push(Segment::Line(self.point(x, y)));
    }

    fn bezier_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64) {
        self.path.push(Segment::Cubic {
            to: self.point(x, y),
            control_a: self.point(c1x, c1y),
            control_b: self.point(c2x, c2y),
        });
    }

    fn close_path(&mut self) {
        self.path.push(Segment::Close);
    }

    fn stroke(&mut self) {
        let mut builder = PathBuilder::stroke(self.width);
        if !self.dash.is_empty() {
            builder = builder.dash_array(&self.dash);
        }
        self.paint_path(builder, self.stroke);
    }

    fn fill(&mut self) {
        self.paint_path(PathBuilder::fill(), self.fill);
    }

    fn fill_text(&mut self, text: &str, x: f64, y: f64, align: TextAlign) {
        let line = self.shape(text);
        let mut origin = self.point(x, y);
        origin.x -= match align {
            TextAlign::Left => px(0.0),
            TextAlign::Center => line.width() / 2.0,
            TextAlign::Right => line.width(),
        };
        origin.y -= line.ascent;
        let _ = line.paint(
            origin,
            line.ascent + line.descent,
            gpui::TextAlign::Left,
            None,
            self.window,
            self.cx,
        );
    }

    fn fill_text_rotated(&mut self, text: &str, x: f64, y: f64, angle_deg: f64, align: TextAlign) {
        if !angle_deg.is_finite() {
            return;
        }

        // GPUI shapes and paints ordinary text directly. Its SVG painter
        // supplies the rotation that the shaped-line painter does not expose.
        let font_size = f32::from(self.font_size);
        let advance = f32::from(self.shape(text).width());
        let anchor = match align {
            TextAlign::Left => "start",
            TextAlign::Center => "middle",
            TextAlign::Right => "end",
        };
        let (sin, cos) = angle_deg.to_radians().sin_cos();
        // Symmetric padding around the baseline origin keeps the GPUI sprite
        // centered on the same anchor as the browser canvas.
        let horizontal = f64::from(advance + font_size);
        let vertical = f64::from(font_size * 2.0);
        let half_w = (cos.abs() * horizontal + sin.abs() * vertical).ceil();
        let half_h = (sin.abs() * horizontal + cos.abs() * vertical).ceil();
        let width = half_w * 2.0;
        let height = half_h * 2.0;
        let text = text
            .replace(['\r', '\n'], " ")
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        let weight = self.font.weight.0;
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{} {} {width} {height}"><text transform="rotate({angle_deg})" text-anchor="{anchor}" font-family="sans-serif" font-size="{font_size}" font-weight="{weight}">{text}</text></svg>"#,
            -half_w, -half_h,
        );
        let bounds = Bounds::new(
            self.point(x - half_w, y - half_h),
            size(px(width as f32), px(height as f32)),
        );
        let _ = self.window.paint_svg(
            bounds,
            svg.clone().into(),
            Some(svg.as_bytes()),
            TransformationMatrix::unit(),
            self.fill,
            self.cx,
        );
    }

    fn text_width(&mut self, text: &str) -> f64 {
        f64::from(f32::from(self.shape(text).width()))
    }
}
