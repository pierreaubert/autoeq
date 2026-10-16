//! HTML shell assembly (shared by the Rust and Python emitters).
//!
//! The shell template lives in `shell/template.html`; both emitters fill the
//! same `{{PLACEHOLDERS}}`. The `.wasm` binaries and wasm-pack glue live in
//! `dist/` (checked in; rebuild with `just report-wasm`).

use base64::Engine as _;

/// Filled shell pieces: base64 `.wasm` binaries plus base64 wasm-pack glue.
/// Glue travels base64 so an inline `<script>` can never be broken by a
/// `</script>` sequence inside generated code.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
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
    // JSON is embedded in a script data block. HTML parsers still recognize
    // a closing script tag inside a JSON string, so escape those characters
    // without changing the decoded payload. Both native and WASM callers use
    // this boundary rather than maintaining host-specific escaping.
    let payload_json = payload_json
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    let title = escape_html(title);
    let replacements = [
        ("{{WASM_2D_B64}}", assets.wasm_2d_b64.as_str()),
        ("{{GLUE_2D_B64}}", assets.glue_2d_b64.as_str()),
        ("{{WASM_GPUI_B64}}", assets.wasm_gpui_b64.as_str()),
        ("{{GLUE_GPUI_B64}}", assets.glue_gpui_b64.as_str()),
        ("{{PAGE_TITLE}}", title.as_str()),
        ("{{PAYLOAD_JSON}}", payload_json.as_str()),
    ];
    // Only template tokens are interpreted. Inserted values may themselves
    // contain token-shaped text and must survive without another substitution.
    let mut remaining = assets.template.as_str();
    let mut output = String::with_capacity(remaining.len() + payload_json.len());
    while let Some(start) = remaining.find("{{") {
        output.push_str(&remaining[..start]);
        remaining = &remaining[start..];
        let Some(end) = remaining.find("}}") else {
            break;
        };
        let token = &remaining[..end + 2];
        output.push_str(
            replacements
                .iter()
                .find(|(key, _)| *key == token)
                .map_or(token, |(_, value)| *value),
        );
        remaining = &remaining[end + 2..];
    }
    output.push_str(remaining);
    output
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
///
/// The `dist/` bundles are generated (`just report-wasm`) and may be absent
/// from a fresh checkout, so they are read at runtime and a missing bundle is
/// reported as `Err` instead of failing compilation via `include_bytes!`.
/// Every caller already handles that `Err` by surfacing it to the user.
pub fn checked_in_assets() -> Result<ShellAssets, String> {
    let template = include_str!("../shell/template.html").to_string();
    // The WebAssembly producer must not embed its own checked-in WASM binary
    // (or the separate GPUI renderer) inside itself; `std::fs` is not
    // available on that target either.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = template;
        return Err("WebAssembly assembly requires explicit `shell_assets`".into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dist = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist");
        let load = |file: &str| {
            std::fs::read(dist.join(file))
                .map_err(|error| format!("could not read dist/{file}: {error}"))
        };
        Ok(ShellAssets {
            template,
            wasm_2d_b64: b64(&load("report2d.wasm")?),
            glue_2d_b64: b64(&load("report2d.js")?),
            wasm_gpui_b64: b64(&load("reportgpui.wasm")?),
            glue_gpui_b64: b64(&load("reportgpui.js")?),
        })
    }
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
    fn inserted_tokens_are_never_reinterpreted() {
        let assets = ShellAssets {
            template: "{{PAGE_TITLE}}|{{WASM_2D_B64}}|{{PAYLOAD_JSON}}|{{UNKNOWN}}|{{tail".into(),
            wasm_2d_b64: "{{PAGE_TITLE}}".into(),
            glue_2d_b64: String::new(),
            wasm_gpui_b64: String::new(),
            glue_gpui_b64: String::new(),
        };
        assert_eq!(
            assemble_html(
                "{{PAYLOAD_JSON}}",
                r#"{"label":"{{WASM_2D_B64}}"}"#,
                &assets
            ),
            r#"{{PAYLOAD_JSON}}|{{PAGE_TITLE}}|{"label":"{{WASM_2D_B64}}"}|{{UNKNOWN}}|{{tail"#,
        );
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

    #[test]
    fn embedded_payload_cannot_close_its_script_block() {
        let assets = ShellAssets {
            template: "<script type=\"application/json\">{{PAYLOAD_JSON}}</script>".into(),
            wasm_2d_b64: String::new(),
            glue_2d_b64: String::new(),
            wasm_gpui_b64: String::new(),
            glue_gpui_b64: String::new(),
        };
        let value =
            serde_json::json!({"label":"</script><script>alert(1)</script>&{{WASM_2D_B64}}"});
        let html = assemble_html("Report", &value.to_string(), &assets);
        assert_eq!(html.matches("</script>").count(), 1);
        let embedded = html
            .split_once('>')
            .unwrap()
            .1
            .strip_suffix("</script>")
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(embedded).unwrap(),
            value
        );
    }
}
