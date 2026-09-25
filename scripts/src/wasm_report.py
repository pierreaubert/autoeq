"""HTML+WASM report emission for roomeq (Plotly replacement).

Builds ``autoeq-report-data-v1`` payloads (see
``crates/autoeq-report-wasm/SCHEMA.md``) and assembles self-contained HTML
reports from the checked-in shell template plus ``dist/`` bundles.

Series dicts intentionally mirror the old plotly trace fields
(name/x/y/color/width/dash) so the figure builders in ``figures.py`` port
mechanically.
"""

import base64
import json
from pathlib import Path

SCHEMA_VERSION = "autoeq-report-data-v1"

# Repo root resolved from this file (scripts/src/wasm_report.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "crates" / "autoeq-report-wasm" / "dist"
SHELL_TEMPLATE = (
    REPO_ROOT / "crates" / "autoeq-report-wasm" / "shell" / "template.html"
)


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def axis(label="", scale="linear", vmin=None, vmax=None):
    """Axis spec dict (x scale is 'log' or 'linear'; y is always linear)."""
    return {"label": label, "scale": scale, "min": vmin, "max": vmax}


def series(name, x, y, color=None, width=2.0, dash="solid", visible=True,
           y_axis=0):
    """Line series dict. ``dash`` is solid/dash/dot/dashdot.

    ``y`` entries may be ``None``: a null sample is a gap (line breaks).
    """
    return {
        "name": name,
        "x": [float(v) for v in x],
        "y": [None if v is None else float(v) for v in y],
        "color": color,
        "width": float(width),
        "dash": dash,
        "visible": bool(visible),
        "y_axis": int(y_axis),
    }


def hline(at, color, dash="solid", width=1.0, label=None):
    """Horizontal reference line dict."""
    return {"at": float(at), "color": color, "dash": dash,
            "width": float(width), "label": label}


def vline(at, color, dash="solid", width=1.0, label=None):
    """Vertical reference line dict."""
    return {"at": float(at), "color": color, "dash": dash,
            "width": float(width), "label": label}


def xrange(x0, x1, color):
    """Shaded x-range dict."""
    return {"x0": float(x0), "x1": float(x1), "color": color}


def annotation(x, y, text):
    """Text annotation dict (data coordinates, primary y)."""
    return {"x": float(x), "y": float(y), "text": str(text)}


def figure(title, x, y, series_list=None, hlines=None, vlines=None,
           xranges=None, annotations=None, legend=True, y2=None, tab=None):
    """Cartesian line figure section dict."""
    return {
        "kind": "figure",
        "figure": {
            "title": title,
            "x": x,
            "y": y,
            "y2": y2,
            "series": list(series_list or []),
            "hlines": list(hlines or []),
            "vlines": list(vlines or []),
            "xranges": list(xranges or []),
            "annotations": list(annotations or []),
            "legend": bool(legend),
        },
        "tab": tab,
    }


def bar_chart(title, categories, groups, ylabel="", ymin=None, ymax=None,
              legend=True, hlines=None, tab=None):
    """Grouped bar chart section dict.

    ``groups`` is a list of ``(name, values, color)`` triples.
    """
    return {
        "kind": "bar",
        "chart": {
            "title": title,
            "categories": list(categories),
            "groups": [
                {"name": name, "values": [float(v) for v in values],
                 "color": color}
                for name, values, color in groups
            ],
            "ylabel": ylabel,
            "ymin": ymin,
            "ymax": ymax,
            "legend": bool(legend),
            "hlines": list(hlines or []),
        },
        "tab": tab,
    }


def sankey_chart(title, nodes, links, tab=None):
    """Flow diagram section dict.

    ``links`` is a list of ``(source, target, value, color)`` tuples.
    """
    return {
        "kind": "sankey",
        "chart": {
            "title": title,
            "nodes": list(nodes),
            "links": [
                {"source": int(s), "target": int(t), "value": float(v),
                 "color": c}
                for s, t, v, c in links
            ],
        },
        "tab": tab,
    }


def html_section(html, tab=None):
    """Raw HTML section dict (tables, summaries, notes)."""
    return {"kind": "html", "html": html, "tab": tab}


def payload(title, sections):
    """Top-level payload dict."""
    return {"schema": SCHEMA_VERSION, "title": title, "sections": sections}


# ---------------------------------------------------------------------------
# Shell assembly
# ---------------------------------------------------------------------------

def load_assets(dist_dir=None, template_path=None):
    """Read the shell template plus base64 ``dist/`` bundles."""
    dist = Path(dist_dir) if dist_dir else DIST_DIR
    template_file = Path(template_path) if template_path else SHELL_TEMPLATE
    template = template_file.read_text(encoding="utf-8")
    assets = {}
    for key, filename, is_text in (
        ("wasm_2d", "report2d.wasm", False),
        ("glue_2d", "report2d.js", True),
        ("wasm_gpui", "reportgpui.wasm", False),
        ("glue_gpui", "reportgpui.js", True),
    ):
        raw = (dist / filename).read_bytes()
        assets[key] = base64.b64encode(raw).decode("ascii")
    return template, assets


def assemble_html(title, payload_dict, dist_dir=None, template_path=None):
    """Assemble a self-contained HTML report string."""
    template, assets = load_assets(dist_dir, template_path)
    # Raw UTF-8 (like the Rust serde output): escaped \uXXXX sequences would
    # break plain-text searches and diverge from the old inline-HTML reports.
    payload_json = json.dumps(payload_dict, ensure_ascii=False)
    safe_title = (
        title.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
    return (
        template
        .replace("{{PAGE_TITLE}}", safe_title)
        .replace("{{PAYLOAD_JSON}}", payload_json)
        .replace("{{WASM_2D_B64}}", assets["wasm_2d"])
        .replace("{{GLUE_2D_B64}}", assets["glue_2d"])
        .replace("{{WASM_GPUI_B64}}", assets["wasm_gpui"])
        .replace("{{GLUE_GPUI_B64}}", assets["glue_gpui"])
    )


def write_report(output_path, title, payload_dict, dist_dir=None,
                 template_path=None):
    """Write a self-contained HTML report file. Returns the path."""
    html = assemble_html(title, payload_dict, dist_dir, template_path)
    output = Path(output_path)
    if output.suffix.lower() != ".html":
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
