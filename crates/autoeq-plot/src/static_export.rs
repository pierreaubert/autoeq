use anyhow::{Context, anyhow};
use plotly::Plot;
use serde_json::Value;
use std::path::Path;

const PALETTE: [u32; 10] = [
    0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b, 0xe377c2, 0x7f7f7f, 0xbcbd22,
    0x17becf,
];

#[derive(Debug)]
struct StaticTrace {
    x: Vec<f64>,
    y: Vec<f64>,
    name: String,
    color: u32,
    axis_index: usize,
}

/// Render Plotly's serializable line-trace model as deterministic SVG, then
/// rasterize it in-process with resvg.
pub(super) fn write_plot_png(
    plot: &Plot,
    output_path: &Path,
    width: usize,
    height: usize,
) -> anyhow::Result<()> {
    let document = serde_json::to_value(plot).context("failed to serialize Plotly chart")?;
    let traces = parse_traces(&document);
    if traces.is_empty() {
        return Err(anyhow!("static chart has no finite line traces"));
    }
    let layout = document.get("layout").unwrap_or(&Value::Null);
    let max_axis = traces
        .iter()
        .map(|trace| trace.axis_index)
        .max()
        .unwrap_or(0);
    let (rows, columns) = grid_dimensions(layout, max_axis + 1);
    let cell_width = width as f32 / columns as f32;
    let cell_height = height as f32 / rows as f32;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#ffffff"/>"##);

    for axis_index in 0..(rows * columns) {
        let axis_traces: Vec<_> = traces
            .iter()
            .filter(|trace| trace.axis_index == axis_index)
            .collect();
        if axis_traces.is_empty() {
            continue;
        }
        let axis = axis_config(layout, axis_index);
        let column = axis_index % columns;
        let row = axis_index / columns;
        svg.push_str(&render_axis_panel(
            &axis_traces,
            &axis,
            column as f32 * cell_width,
            row as f32 * cell_height,
            cell_width,
            cell_height,
            axis_index,
        )?);
    }
    svg.push_str("</svg>");

    rasterize_svg(&svg, output_path, width, height)
}

fn render_axis_panel(
    traces: &[&StaticTrace],
    axis: &AxisConfig,
    offset_x: f32,
    offset_y: f32,
    width: f32,
    height: f32,
    panel_index: usize,
) -> anyhow::Result<String> {
    let margin_left = 58.0;
    let margin_right = 18.0;
    let margin_top = 34.0;
    let margin_bottom = 42.0;
    let plot_width = (width - margin_left - margin_right).max(1.0);
    let plot_height = (height - margin_top - margin_bottom).max(1.0);

    let all_x = traces
        .iter()
        .flat_map(|trace| trace.x.iter().copied())
        .filter(|value| value.is_finite() && (!axis.log_x || *value > 0.0));
    let all_y = traces
        .iter()
        .flat_map(|trace| trace.y.iter().copied())
        .filter(|value| value.is_finite());
    let x_range = axis
        .x_range
        .or_else(|| finite_extent(all_x, axis.log_x))
        .ok_or_else(|| anyhow!("static chart has no valid x range"))?;
    let y_range = axis
        .y_range
        .or_else(|| finite_extent(all_y, false).map(pad_linear_range))
        .ok_or_else(|| anyhow!("static chart has no valid y range"))?;
    let x_domain = if axis.log_x {
        (x_range.0.log10(), x_range.1.log10())
    } else {
        x_range
    };

    let map_x = |value: f64| {
        let value = if axis.log_x { value.log10() } else { value };
        margin_left + ((value - x_domain.0) / (x_domain.1 - x_domain.0)) as f32 * plot_width
    };
    let map_y = |value: f64| {
        margin_top + (1.0 - (value - y_range.0) / (y_range.1 - y_range.0)) as f32 * plot_height
    };

    let mut out = format!(
        "<g transform=\"translate({offset_x},{offset_y})\"><rect width=\"{width}\" height=\"{height}\" fill=\"#fff\"/><clipPath id=\"panel-{panel_index}\"><rect x=\"{margin_left}\" y=\"{margin_top}\" width=\"{plot_width}\" height=\"{plot_height}\"/></clipPath>"
    );
    if let Some(title) = &axis.title {
        out.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#222\">{}</text>",
            width / 2.0,
            escape_xml(title)
        ));
    }

    for step in 0..=5 {
        let fraction = step as f32 / 5.0;
        let y = margin_top + fraction * plot_height;
        let value = y_range.1 - fraction as f64 * (y_range.1 - y_range.0);
        out.push_str(&format!(
            "<line x1=\"{margin_left}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#e5e7eb\"/><text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#555\">{value:.1}</text>",
            margin_left + plot_width,
            margin_left - 6.0,
            y + 3.0
        ));
    }

    let x_ticks = if axis.log_x {
        log_ticks(x_range)
    } else {
        (0..=5)
            .map(|step| x_range.0 + step as f64 / 5.0 * (x_range.1 - x_range.0))
            .collect()
    };
    for value in x_ticks {
        let x = map_x(value);
        out.push_str(&format!(
            "<line x1=\"{x}\" y1=\"{margin_top}\" x2=\"{x}\" y2=\"{}\" stroke=\"#e5e7eb\"/><text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#555\">{}</text>",
            margin_top + plot_height,
            margin_top + plot_height + 16.0,
            format_tick(value)
        ));
    }
    out.push_str(&format!(
        "<rect x=\"{margin_left}\" y=\"{margin_top}\" width=\"{plot_width}\" height=\"{plot_height}\" fill=\"none\" stroke=\"#6b7280\"/>"
    ));

    for trace in traces {
        let mut path = String::new();
        for (&x, &y) in trace.x.iter().zip(&trace.y) {
            if !x.is_finite() || !y.is_finite() || (axis.log_x && x <= 0.0) {
                continue;
            }
            let command = if path.is_empty() { 'M' } else { 'L' };
            path.push_str(&format!("{command}{:.3},{:.3}", map_x(x), map_y(y)));
        }
        if !path.is_empty() {
            out.push_str(&format!(
                "<path d=\"{path}\" fill=\"none\" stroke=\"#{:06x}\" stroke-width=\"1.8\" clip-path=\"url(#panel-{panel_index})\"/>",
                trace.color
            ));
        }
    }

    let legend_x = margin_left + 8.0;
    for (index, trace) in traces.iter().take(8).enumerate() {
        let y = margin_top + 14.0 + index as f32 * 14.0;
        out.push_str(&format!(
            "<line x1=\"{legend_x}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#{:06x}\" stroke-width=\"2\"/><text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#222\">{}</text>",
            legend_x + 14.0,
            trace.color,
            legend_x + 18.0,
            y + 3.0,
            escape_xml(&trace.name)
        ));
    }
    out.push_str("</g>");
    Ok(out)
}

fn finite_extent(values: impl Iterator<Item = f64>, log_scale: bool) -> Option<(f64, f64)> {
    let (minimum, maximum) = values.fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    );
    if !minimum.is_finite() || !maximum.is_finite() {
        None
    } else if minimum < maximum {
        Some((minimum, maximum))
    } else if log_scale && minimum > 0.0 {
        Some((minimum / 2.0, maximum * 2.0))
    } else {
        Some((minimum - 1.0, maximum + 1.0))
    }
}

fn pad_linear_range((minimum, maximum): (f64, f64)) -> (f64, f64) {
    let padding = ((maximum - minimum) * 0.05).max(0.5);
    (minimum - padding, maximum + padding)
}

fn log_ticks(range: (f64, f64)) -> Vec<f64> {
    const MULTIPLIERS: [f64; 3] = [1.0, 2.0, 5.0];
    let first_decade = range.0.log10().floor() as i32;
    let last_decade = range.1.log10().ceil() as i32;
    (first_decade..=last_decade)
        .flat_map(|decade| MULTIPLIERS.map(move |multiplier| multiplier * 10f64.powi(decade)))
        .filter(|value| *value >= range.0 && *value <= range.1)
        .collect()
}

fn format_tick(value: f64) -> String {
    if value.abs() >= 1_000.0 {
        format!("{:.0}k", value / 1_000.0)
    } else if value.abs() >= 10.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn rasterize_svg(svg: &str, output_path: &Path, width: usize, height: usize) -> anyhow::Result<()> {
    let tree = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default())
        .context("failed to parse static chart SVG")?;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width as u32, height as u32)
        .ok_or_else(|| anyhow!("invalid static PNG dimensions {width}x{height}"))?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    pixmap
        .save_png(output_path)
        .with_context(|| format!("failed to write static PNG '{}'", output_path.display()))?;
    Ok(())
}

fn parse_traces(document: &Value) -> Vec<StaticTrace> {
    document
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, trace)| {
            if trace.get("visible").and_then(Value::as_bool) == Some(false) {
                return None;
            }
            let (x, y) = numeric_pairs(trace.get("x")?, trace.get("y")?)?;
            Some(StaticTrace {
                x,
                y,
                name: trace
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("Series")
                    .to_string(),
                color: trace_color(trace).unwrap_or(PALETTE[index % PALETTE.len()]),
                axis_index: trace
                    .get("xaxis")
                    .and_then(Value::as_str)
                    .map(axis_index)
                    .unwrap_or(0),
            })
        })
        .collect()
}

fn numeric_pairs(x: &Value, y: &Value) -> Option<(Vec<f64>, Vec<f64>)> {
    let x_values = x.as_array()?;
    let y_values = y.as_array()?;
    let mut paired_x = Vec::with_capacity(x_values.len().min(y_values.len()));
    let mut paired_y = Vec::with_capacity(paired_x.capacity());
    for (x, y) in x_values.iter().zip(y_values) {
        let (Some(x), Some(y)) = (x.as_f64(), y.as_f64()) else {
            continue;
        };
        if x.is_finite() && y.is_finite() {
            paired_x.push(x);
            paired_y.push(y);
        }
    }
    (!paired_x.is_empty()).then_some((paired_x, paired_y))
}

fn axis_index(name: &str) -> usize {
    name.strip_prefix('x')
        .and_then(|suffix| {
            if suffix.is_empty() {
                Some(1)
            } else {
                suffix.parse::<usize>().ok()
            }
        })
        .unwrap_or(1)
        .saturating_sub(1)
}

fn trace_color(trace: &Value) -> Option<u32> {
    let color = trace
        .pointer("/line/color")
        .or_else(|| trace.pointer("/marker/color"))?
        .as_str()?;
    let hex = color.strip_prefix('#')?;
    (hex.len() == 6)
        .then(|| u32::from_str_radix(hex, 16).ok())
        .flatten()
}

fn grid_dimensions(layout: &Value, required_cells: usize) -> (usize, usize) {
    let rows = layout
        .pointer("/grid/rows")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let rows = rows.max(1);
    let columns = layout
        .pointer("/grid/columns")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| required_cells.div_ceil(rows) as u64) as usize;
    let columns = columns.max(1);
    let rows = rows.max(required_cells.div_ceil(columns));
    (rows, columns)
}

#[derive(Default)]
struct AxisConfig {
    log_x: bool,
    x_range: Option<(f64, f64)>,
    y_range: Option<(f64, f64)>,
    title: Option<String>,
}

fn axis_config(layout: &Value, index: usize) -> AxisConfig {
    let suffix = if index == 0 {
        String::new()
    } else {
        (index + 1).to_string()
    };
    let x_axis = layout.get(format!("xaxis{suffix}")).unwrap_or(&Value::Null);
    let y_axis = layout.get(format!("yaxis{suffix}")).unwrap_or(&Value::Null);
    let log_x = x_axis.get("type").and_then(Value::as_str) == Some("log");
    let mut x_range = numeric_range(x_axis.get("range"));
    if log_x {
        x_range = x_range.map(|(minimum, maximum)| (10f64.powf(minimum), 10f64.powf(maximum)));
    }
    AxisConfig {
        log_x,
        x_range,
        y_range: numeric_range(y_axis.get("range")),
        title: axis_title(y_axis).or_else(|| axis_title(x_axis)),
    }
}

fn numeric_range(value: Option<&Value>) -> Option<(f64, f64)> {
    let range = value?.as_array()?;
    let minimum = range.first()?.as_f64()?;
    let maximum = range.get(1)?.as_f64()?;
    (minimum.is_finite() && maximum.is_finite() && minimum < maximum).then_some((minimum, maximum))
}

fn axis_title(axis: &Value) -> Option<String> {
    axis.pointer("/title/text")
        .and_then(Value::as_str)
        .or_else(|| axis.get("title").and_then(Value::as_str))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::{grid_dimensions, parse_traces};
    use serde_json::json;

    #[test]
    fn trace_parser_keeps_x_and_y_values_paired_across_gaps() {
        let traces = parse_traces(&json!({
            "data": [{
                "x": [20.0, null, 80.0, 160.0],
                "y": [1.0, 2.0, null, 4.0],
            }]
        }));

        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].x, vec![20.0, 160.0]);
        assert_eq!(traces[0].y, vec![1.0, 4.0]);
    }

    #[test]
    fn zero_sized_plotly_grid_dimensions_are_clamped() {
        assert_eq!(
            grid_dimensions(&json!({"grid": {"rows": 0, "columns": 0}}), 3),
            (3, 1)
        );
    }
}
