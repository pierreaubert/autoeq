//! HTML shell assembly (shared by the Rust and Python emitters).
//!
//! The shell template lives in `shell/template.html`; both emitters fill the
//! same `{{PLACEHOLDERS}}`. The `.wasm` binaries and wasm-pack glue live in
//! `dist/` (checked in; rebuild with `just report-wasm`).

use base64::Engine as _;

/// Filled shell pieces: base64 `.wasm` binaries plus base64 wasm-pack glue.
/// Glue travels base64 so an inline `<script>` can never be broken by a
/// `</script>` sequence inside generated code.
pub struct ShellAssets {
    /// Contents of `shell/template.html`.
    pub template: String,
    /// Base64 of `dist/report2d.wasm`.
    pub wasm_2d_b64: String,
    /// Base64 of the wasm-pack `--target web` JS glue for the 2D bundle.
    pub glue_2d_b64: String,
    /// Base64 of `dist/reportgpui.wasm`.
    pub wasm_gpui_b64: String,
    /// Base64 of the wasm-pack `--target web` JS glue for the GPUI bundle.
    pub glue_gpui_b64: String,
}

/// Assemble a self-contained HTML report.
pub fn assemble_html(title: &str, payload_json: &str, assets: &ShellAssets) -> String {
    assets
        .template
        .replace("{{PAGE_TITLE}}", &escape_html(title))
        .replace("{{PAYLOAD_JSON}}", payload_json)
        .replace("{{WASM_2D_B64}}", &assets.wasm_2d_b64)
        .replace("{{GLUE_2D_B64}}", &assets.glue_2d_b64)
        .replace("{{WASM_GPUI_B64}}", &assets.wasm_gpui_b64)
        .replace("{{GLUE_GPUI_B64}}", &assets.glue_gpui_b64)
}

/// Base64-encode raw bytes (standard alphabet, no line breaks).
pub fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Load the checked-in shell template plus `dist/` bundle bytes.
pub fn checked_in_assets() -> Result<ShellAssets, String> {
    let template = include_str!("../shell/template.html").to_string();
    let wasm_2d_b64 = b64(include_bytes!("../dist/report2d.wasm"));
    let glue_2d_b64 = b64(include_bytes!("../dist/report2d.js"));
    let wasm_gpui_b64 = b64(include_bytes!("../dist/reportgpui.wasm"));
    let glue_gpui_b64 = b64(include_bytes!("../dist/reportgpui.js"));
    Ok(ShellAssets {
        template,
        wasm_2d_b64,
        glue_2d_b64,
        wasm_gpui_b64,
        glue_gpui_b64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_fill() {
        let assets = ShellAssets {
            template: "<title>{{PAGE_TITLE}}</title><p>{{PAYLOAD_JSON}}</p>".to_string(),
            wasm_2d_b64: String::new(),
            glue_2d_b64: String::new(),
            wasm_gpui_b64: String::new(),
            glue_gpui_b64: String::new(),
        };
        let html = assemble_html("a<b", "{\"x\":1}", &assets);
        assert!(html.contains("<title>a&lt;b</title>"));
        assert!(html.contains("<p>{\"x\":1}</p>"));
    }

    #[test]
    fn b64_round_trip() {
        let raw = [0u8, 1, 2, 250, 255];
        let enc = b64(&raw);
        let back = base64::engine::general_purpose::STANDARD
            .decode(&enc)
            .expect("decode");
        assert_eq!(back, raw);
    }
}
