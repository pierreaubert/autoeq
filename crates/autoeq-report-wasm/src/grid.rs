//! Time-frequency heatmaps and surfaces using d3rs geometry on the shared canvas backend.
//!
//! Input levels retain their exported reference. Display clipping never changes
//! normalization, and no acoustic fitting or resampling is performed here.

use d3rs::axis::{AxisConfig, AxisLayout};
use d3rs::color::{D3Color, SequentialScheme, interpolate_colors};
use d3rs::scale::{LinearScale, LogScale, Scale};
use d3rs::surface::{OrthographicProjection, Projection, SurfaceData, SurfaceMesh, SurfacePoint3D};

use crate::draw::{
    Ctx, DrawMeta, LegendEntry, PaintTriangle, TextAlign, draw_axis_layout, fmt_log_grid, fmt_num,
    log_grid_ticks,
};
use crate::schema::{Figure, GridData};

fn valid(grid: &GridData) -> bool {
    grid.x.len() >= 2
        && grid.y.len() >= 2
        && grid.x.len() <= 4096
        && grid.y.len() <= 4096
        && grid.x.len().saturating_mul(grid.y.len()) <= 1_000_000
        && grid.x.iter().all(|v| v.is_finite() && *v > 0.0)
        && grid.y.iter().all(|v| v.is_finite())
        && grid.x.windows(2).all(|w| w[0] < w[1])
        && grid.y.windows(2).all(|w| w[0] < w[1])
        && grid.z.len() == grid.y.len()
        && grid
            .z
            .iter()
            .all(|r| r.len() == grid.x.len() && r.iter().all(|v| v.is_finite()))
        && grid.zmin.is_finite()
        && grid.zmax.is_finite()
        && grid.zmin < grid.zmax
}

/// Peak energy arrival time per frequency column: `(freq_hz, time_ms)`.
///
/// Each column's peak is the earliest time holding the column maximum —
/// ties resolve to first arrival. Levels keep their exported reference;
/// only the argmax index is used, so display clipping cannot move it.
pub(crate) fn peak_energy_times(x: &[f64], y: &[f64], z: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(x.len());
    for (fi, &freq) in x.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (ti, row) in z.iter().enumerate() {
            if let Some(&db) = row.get(fi) {
                let better = match best {
                    None => true,
                    Some((_, top)) => db > top,
                };
                if better {
                    best = Some((ti, db));
                }
            }
        }
        if let Some((ti, _)) = best
            && let Some(&time) = y.get(ti)
        {
            out.push((freq, time));
        }
    }
    out
}

fn level_color(t: f64) -> String {
    // Fixed violet/blue/cyan/green/yellow/red legend, shared by cells and colorbar.
    interpolate_colors(
        &[
            D3Color::rgb(26, 10, 80),
            D3Color::rgb(25, 90, 255),
            D3Color::rgb(90, 240, 240),
            D3Color::rgb(150, 255, 110),
            D3Color::rgb(255, 235, 40),
            D3Color::rgb(220, 70, 20),
            D3Color::rgb(110, 10, 5),
        ],
        t.clamp(0.0, 1.0) as f32,
    )
    .to_hex()
}

/// Surface fill color from a d3rs sequential map. Unknown names fall back
/// to turbo so a mistyped viewer option never blanks the surface.
fn surface_color(map: &str, t: f64) -> D3Color {
    let scale = match map {
        "viridis" => SequentialScheme::viridis(),
        "plasma" => SequentialScheme::plasma(),
        "inferno" => SequentialScheme::inferno(),
        "magma" => SequentialScheme::magma(),
        "rainbow" => SequentialScheme::rainbow(),
        _ => SequentialScheme::turbo(),
    };
    scale.get(t.clamp(0.0, 1.0))
}

#[allow(clippy::cast_possible_truncation)]
fn to_rgb(color: D3Color) -> [u8; 3] {
    [
        (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn path(ctx: &mut impl Ctx, points: &[(f64, f64)], fill: bool) {
    ctx.begin_path();
    for (i, &(x, y)) in points.iter().enumerate() {
        if i == 0 {
            ctx.move_to(x, y);
        } else {
            ctx.line_to(x, y);
        }
    }
    if fill {
        ctx.close_path();
        ctx.fill();
    } else {
        ctx.stroke();
    }
}

/// Draw a validated grid with axes, color legend, and optional resonance highlights.
///
/// Malformed grids display an error message rather than indexing unchecked arrays.
pub fn draw_grid(ctx: &mut impl Ctx, fig: &Figure, grid: &GridData, w: f64, h: f64) -> DrawMeta {
    ctx.set_fill("#ffffff");
    ctx.fill_rect(0.0, 0.0, w, h);
    ctx.set_font("12px system-ui");
    ctx.set_fill("#333333");
    if !valid(grid) {
        ctx.fill_text(
            "Unavailable: invalid time-frequency grid",
            20.0,
            30.0,
            TextAlign::Left,
        );
        return DrawMeta::default();
    }
    let xmin = fig.x.min.unwrap_or(grid.x[0]);
    let xmax = fig.x.max.unwrap_or(grid.x[grid.x.len() - 1]);
    let ymin = fig.y.min.unwrap_or(grid.y[0]);
    let ymax = fig.y.max.unwrap_or(grid.y[grid.y.len() - 1]);
    if !(xmin > 0.0 && xmax > xmin && ymax > ymin)
        || ![xmin, xmax, ymin, ymax].iter().all(|v| v.is_finite())
    {
        return DrawMeta::default();
    }
    let left = 76.0;
    let top = 68.0;
    let pw = (w - 172.0).max(60.0);
    let ph = (h - 140.0).max(60.0);
    ctx.fill_text(&fig.title, left, 22.0, TextAlign::Left);
    let mut meta = DrawMeta::default();
    let mut lx = left;
    for (i, series) in fig.series.iter().enumerate() {
        let width = ctx.text_width(&series.name) + 25.0;
        ctx.set_fill(if series.visible {
            series.color.as_deref().unwrap_or("#8855dd")
        } else {
            "#aaaaaa"
        });
        ctx.fill_rect(lx, 37.0, 14.0, 3.0);
        ctx.fill_text(&series.name, lx + 19.0, 43.0, TextAlign::Left);
        meta.legend.push(LegendEntry {
            series: i,
            x: lx,
            y: 28.0,
            w: width,
            h: 22.0,
        });
        lx += width;
    }
    let xs = LogScale::new().domain(xmin, xmax).range(0.0, 1.0);
    let ys = LinearScale::new().domain(ymin, ymax).range(0.0, 1.0);
    let zs = LinearScale::new()
        .domain(grid.zmin, grid.zmax)
        .range(0.0, 1.0);
    if grid.surface {
        // Center the box, rotate with d3rs, then fit all eight corners to the viewport.
        let [elevation, azimuth] = grid
            .rotation
            .filter(|r| r.iter().all(|v| v.is_finite()))
            .unwrap_or([65.0, -12.0]);
        let mut projection = OrthographicProjection::new()
            .scale(1.0)
            .rotation(-elevation, 0.0, azimuth)
            .origin(0.0, 0.0);
        let sx = |v: f64| (v - 0.5) * 2.4;
        let sy = |v: f64| 0.5 - v;
        let sz = |v: f64| (v - 0.5) * 0.9;
        let mut bounds = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for x in [0.0, 1.0] {
            for y in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    let p = projection.project(sx(x), sy(y), sz(z));
                    bounds[0] = bounds[0].min(p.x);
                    bounds[1] = bounds[1].max(p.x);
                    bounds[2] = bounds[2].min(p.y);
                    bounds[3] = bounds[3].max(p.y);
                }
            }
        }
        let scale = (pw / (bounds[1] - bounds[0])).min(ph / (bounds[3] - bounds[2]));
        projection = projection.scale(scale).origin(
            left + pw / 2.0 - scale * (bounds[0] + bounds[1]) / 2.0,
            top + ph / 2.0 - scale * (bounds[2] + bounds[3]) / 2.0,
        );
        let project = |f: f64, t: f64, db: f64| {
            let p = projection.project(
                sx(xs.scale(f)),
                sy(ys.scale(t)),
                sz(zs.scale(db.clamp(grid.zmin, grid.zmax))),
            );
            (p.x, p.y)
        };
        ctx.set_stroke("#cccccc");
        ctx.set_line_width(0.7);
        // Axis tick labels, axis titles and colorbar scale read at 14px;
        // the figure title and series legend above stay at 12px.
        ctx.set_font("14px system-ui");
        for f in log_grid_ticks(xmin, xmax) {
            path(
                ctx,
                &[
                    project(f, ymin, grid.zmax),
                    project(f, ymin, grid.zmin),
                    project(f, ymax, grid.zmin),
                ],
                false,
            );
            let (x, y) = project(f, ymax, grid.zmin);
            ctx.set_fill("#333333");
            ctx.fill_text(&fmt_log_grid(f), x, y + 20.0, TextAlign::Center);
        }
        for db in LinearScale::new().domain(grid.zmin, grid.zmax).ticks(6) {
            path(
                ctx,
                &[project(xmin, ymin, db), project(xmax, ymin, db)],
                false,
            );
            let (x, y) = project(xmin, ymin, db);
            ctx.fill_text(&fmt_num(db), x - 8.0, y + 5.0, TextAlign::Right);
        }
        for t in LinearScale::new().domain(ymin, ymax).ticks(5) {
            path(
                ctx,
                &[project(xmin, t, grid.zmin), project(xmax, t, grid.zmin)],
                false,
            );
            let (x, y) = project(xmin, t, grid.zmin);
            ctx.fill_text(&fmt_num(t), x - 8.0, y + 5.0, TextAlign::Right);
        }
        let fi: Vec<usize> = (0..grid.x.len())
            .filter(|i| grid.x[*i] >= xmin && grid.x[*i] <= xmax)
            .collect();
        let ti: Vec<usize> = (0..grid.y.len())
            .filter(|i| grid.y[*i] >= ymin && grid.y[*i] <= ymax)
            .collect();
        if grid.show_surface {
            let points: Vec<Vec<SurfacePoint3D>> = ti
                .iter()
                .map(|&t| {
                    fi.iter()
                        .map(|&f| {
                            SurfacePoint3D::new(
                                sx(xs.scale(grid.x[f])),
                                sy(ys.scale(grid.y[t])),
                                sz(zs.scale(grid.z[t][f].clamp(grid.zmin, grid.zmax))),
                                grid.z[t][f],
                            )
                        })
                        .collect()
                })
                .collect();
            let mut mesh = SurfaceMesh::from_surface_data(&SurfaceData::from_grid(points));
            mesh.depth_sort(&projection);
            let triangles: Vec<_> = mesh
                .triangles
                .iter()
                .map(|triangle| {
                    let points = triangle.vertices.map(|p| {
                        let q = projection.project_point(&p);
                        (q.x, q.y)
                    });
                    PaintTriangle {
                        points,
                        color: to_rgb(surface_color(&grid.colormap, zs.scale(triangle.avg_t))),
                    }
                })
                .collect();
            ctx.triangles(&triangles);
        }
        if grid.show_contours && fi.len() >= 2 {
            // REW-style wireframe: each time slice is a black polyline over a
            // white underfill, painted back-to-front for hidden-line removal.
            let mut order: Vec<usize> = ti.clone();
            order.sort_by(|&a, &b| {
                let depth = |t: usize| {
                    projection.point_depth(&SurfacePoint3D::new(
                        sx(0.5),
                        sy(ys.scale(grid.y[t])),
                        sz(0.0),
                        0.0,
                    ))
                };
                depth(b)
                    .partial_cmp(&depth(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let first = *fi.first().expect("checked non-empty");
            let last = *fi.last().expect("checked non-empty");
            for &t in &order {
                let mut poly: Vec<(f64, f64)> = fi
                    .iter()
                    .map(|&f| project(grid.x[f], grid.y[t], grid.z[t][f]))
                    .collect();
                poly.push(project(grid.x[last], grid.y[t], grid.zmin));
                poly.push(project(grid.x[first], grid.y[t], grid.zmin));
                ctx.set_fill("#ffffff");
                path(ctx, &poly, true);
                ctx.set_stroke("#111111");
                ctx.set_line_width(1.0);
                path(ctx, &poly[..poly.len() - 2], false);
            }
        }
        for (i, &f) in grid.highlights.iter().enumerate() {
            let Some(series) = fig.series.get(i).filter(|s| s.visible) else {
                continue;
            };
            if f >= grid.x.len() || grid.x[f] < xmin || grid.x[f] > xmax {
                continue;
            }
            ctx.set_stroke(series.color.as_deref().unwrap_or("#8855dd"));
            ctx.set_line_width(2.5);
            let points: Vec<_> = ti
                .iter()
                .map(|&t| project(grid.x[f], grid.y[t], grid.z[t][f]))
                .collect();
            path(ctx, &points, false);
        }
        ctx.set_fill("#333333");
        ctx.fill_text(
            "Frequency (Hz)",
            left + pw / 2.0,
            top + ph + 45.0,
            TextAlign::Center,
        );
        ctx.fill_text("Time (ms)", left, top + ph + 45.0, TextAlign::Center);
        ctx.fill_text_rotated(
            "Relative level (dB)",
            18.0,
            top + ph / 2.0,
            -90.0,
            TextAlign::Center,
        );
    } else {
        let x = LogScale::new().domain(xmin, xmax).range(left, left + pw);
        let y = LinearScale::new().domain(ymin, ymax).range(top + ph, top);
        // Cells use geometric frequency boundaries and arithmetic time boundaries.
        for (ti, row) in grid.z.iter().enumerate() {
            let low = if ti == 0 {
                grid.y[0]
            } else {
                (grid.y[ti - 1] + grid.y[ti]) / 2.0
            };
            let high = if ti + 1 == grid.y.len() {
                grid.y[ti]
            } else {
                (grid.y[ti] + grid.y[ti + 1]) / 2.0
            };
            if high < ymin || low > ymax {
                continue;
            }
            for (fi, &db) in row.iter().enumerate() {
                let low_f = if fi == 0 {
                    grid.x[0]
                } else {
                    (grid.x[fi - 1] * grid.x[fi]).sqrt()
                };
                let high_f = if fi + 1 == grid.x.len() {
                    grid.x[fi]
                } else {
                    (grid.x[fi] * grid.x[fi + 1]).sqrt()
                };
                if high_f < xmin || low_f > xmax {
                    continue;
                }
                let x0 = x.scale(low_f.max(xmin));
                let x1 = x.scale(high_f.min(xmax));
                let y0 = y.scale(high.min(ymax));
                let y1 = y.scale(low.max(ymin));
                ctx.set_fill(&level_color(zs.scale(db)));
                ctx.fill_rect(x0, y0, x1 - x0, y1 - y0);
            }
        }
        let xc = AxisConfig::bottom()
            .with_tick_values(log_grid_ticks(xmin, xmax))
            .with_formatter(fmt_log_grid)
            .with_title(fig.x.label.clone());
        let yc = AxisConfig::left()
            .with_ticks(5)
            .with_formatter(fmt_num)
            .with_title(fig.y.label.clone());
        draw_axis_layout(
            ctx,
            &AxisLayout::from_scale(&x, &xc, pw as f32),
            top + ph,
            ph,
            true,
            false,
        );
        draw_axis_layout(
            ctx,
            &AxisLayout::from_scale(&y, &yc, ph as f32),
            left,
            pw,
            false,
            false,
        );
        // Peak energy arrival per frequency (cyan dashed ridge). Drawn
        // after the axes so it stays crisp over cells and gridlines;
        // segments outside the viewport lift the pen instead of painting
        // the margins.
        ctx.set_stroke("#00d5ff");
        ctx.set_line_width(2.0);
        ctx.set_dash(&[6.0, 4.0]);
        ctx.begin_path();
        let mut pen = false;
        for (freq, time) in peak_energy_times(&grid.x, &grid.y, &grid.z) {
            if freq < xmin || freq > xmax || time < ymin || time > ymax {
                pen = false;
                continue;
            }
            let (px, py) = (x.scale(freq), y.scale(time));
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
        ctx.set_dash(&[]);
    }
    // Numeric colorbar is part of both views, with unchanged dB reference.
    // Surface views follow the selected fill map; heatmaps keep the legend.
    let bar = |t: f64| {
        if grid.surface {
            surface_color(&grid.colormap, t).to_hex()
        } else {
            level_color(t)
        }
    };
    for i in 0..100 {
        ctx.set_fill(&bar(1.0 - i as f64 / 99.0));
        ctx.fill_rect(
            left + pw + 24.0,
            top + ph * i as f64 / 100.0,
            14.0,
            ph / 100.0 + 0.5,
        );
    }
    ctx.set_fill("#333333");
    for i in 0..=6 {
        let db = grid.zmax - (grid.zmax - grid.zmin) * i as f64 / 6.0;
        ctx.fill_text(
            &fmt_num(db),
            left + pw + 43.0,
            top + ph * i as f64 / 6.0 + 4.0,
            TextAlign::Left,
        );
    }
    ctx.fill_text("dB", left + pw + 24.0, top - 12.0, TextAlign::Left);
    meta
}

#[cfg(test)]
mod tests {
    use super::{peak_energy_times, surface_color, to_rgb};

    #[test]
    fn peak_energy_picks_earliest_column_maximum() {
        let x = vec![20.0, 100.0, 1000.0];
        let y = vec![-1.0, 0.0, 2.0, 5.0];
        let z = vec![
            vec![-30.0, -30.0, -30.0],
            vec![-6.0, -3.0, -30.0],
            vec![-12.0, -3.0, -1.0],
            vec![-20.0, -9.0, -4.0],
        ];
        assert_eq!(
            peak_energy_times(&x, &y, &z),
            vec![(20.0, 0.0), (100.0, 0.0), (1000.0, 2.0)]
        );
    }

    #[test]
    fn peak_energy_skips_missing_cells_without_panicking() {
        let x = vec![20.0, 100.0];
        let y = vec![0.0];
        let z = vec![vec![-3.0]];
        assert_eq!(peak_energy_times(&x, &y, &z), vec![(20.0, 0.0)]);
        assert!(peak_energy_times(&[], &[], &[]).is_empty());
    }

    #[test]
    fn surface_maps_differ_and_unknown_falls_back_to_turbo() {
        let at = |map: &str| to_rgb(surface_color(map, 0.5));
        let turbo = at("turbo");
        assert_eq!(at("no-such-map"), turbo);
        let others = [
            at("viridis"),
            at("plasma"),
            at("inferno"),
            at("magma"),
            at("rainbow"),
        ];
        assert!(
            others.iter().any(|c| *c != turbo),
            "maps must differ from turbo at mid-scale"
        );
        // Clamping keeps out-of-range stops in gamut.
        for map in ["turbo", "viridis", "plasma", "inferno", "magma", "rainbow"] {
            let _ = to_rgb(surface_color(map, -1.0));
            let _ = to_rgb(surface_color(map, 2.0));
        }
    }
}
