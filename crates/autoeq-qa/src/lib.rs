//! Wolfram Engine cross-validation harness for AutoEQ / RoomEQ.
//!
//! Each QA case pairs a closed-form `.wls` oracle (`wolfram/`) with a Rust
//! comparison test (`tests/`). References resolve live when the
//! `WOLFRAMSCRIPT` environment variable points at an activated engine, and
//! otherwise fall back to the checked-in goldens (`wolfram/goldens/`).
//! Regenerate goldens with `just qa-wolfram-goldens` (needs the engine).
//!
//! Conventions (mirroring `math-qa` in the neighboring `math-audio`
//! checkout, itself attributing them to `sonium-qa`):
//! - Every `.wls` script prints exactly one compact RawJSON payload as its
//!   last stdout line; the harness parses that line.
//! - Comparisons use [`rel_error`] (zero-reference handling) and
//!   case-local tolerances recorded in `validation-manifest.toml`.
//! - Passing comparison tests print one `QA_RESULT:` JSON line.
//!
//! One deliberate difference from `math-qa`: the required comparison gate
//! must never pass by executing zero comparisons. Comparison tests resolve
//! references through [`require_reference`], which fails loudly when
//! neither a live engine nor a checked-in golden is available. Unavailable
//! counts as incomplete, never as a pass.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde::Serialize;

/// Relative error `|a − b| / |b|` with deterministic zero handling: 0 when
/// both are zero (or both non-finite-equal), ∞ when only the reference is
/// zero, NaN when either side is NaN.
pub fn rel_error(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return 0.0;
    }
    if b == 0.0 {
        return f64::INFINITY;
    }
    ((a - b) / b).abs()
}

/// Relative error of two complex values: `|a − b| / |b|`.
///
/// This is the norm of the difference over the reference magnitude — not
/// `rel_error` of the two norms, which collapses to ~1 whenever the
/// values agree. Zero handling mirrors [`rel_error`]: 0 when `a == b`,
/// ∞ when only the reference is zero, NaN when either side is NaN.
pub fn complex_rel_error(a: num_complex::Complex64, b: num_complex::Complex64) -> f64 {
    let num = (a - b).norm();
    let denom = b.norm();
    if num.is_nan() || denom.is_nan() {
        return f64::NAN;
    }
    if num == 0.0 {
        return 0.0;
    }
    if denom == 0.0 {
        return f64::INFINITY;
    }
    num / denom
}

/// Assert `rel_error(actual, expected) <= tol` with a diagnostic message.
pub fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = rel_error(actual, expected);
    assert!(
        err <= tol,
        "{what}: actual={actual:.12e} expected={expected:.12e} rel_err={err:.3e} tol={tol:.1e}"
    );
}

/// Assert `|actual − expected| <= tol` (for near-zero quantities).
pub fn assert_close_abs(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{what}: actual={actual:.12e} expected={expected:.12e} abs_err={err:.3e} tol={tol:.1e}"
    );
}

/// Assert every value in `values` is finite.
pub fn assert_all_finite(values: &[f64], what: &str) {
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{what}: non-finite output in {values:?}"
    );
}

/// Assert the reference carries the expected case identity and schema.
pub fn assert_case_id(ref_json: &serde_json::Value, expected_case: &str, what: &str) {
    assert_eq!(
        ref_json["case"], expected_case,
        "{what}: case identity mismatch"
    );
    assert!(
        ref_json["schema_version"].is_number(),
        "{what}: reference is missing schema_version"
    );
}

/// Directory holding the checked-in golden JSON files.
///
/// Overridable with `AUTOEQ_QA_GOLDEN_DIR` (used by regeneration checks);
/// defaults to `wolfram/goldens` next to this crate's manifest.
pub fn golden_dir() -> PathBuf {
    if let Some(dir) = env::var_os("AUTOEQ_QA_GOLDEN_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/goldens")
}

/// Absolute path of a `.wls` oracle script.
pub fn wolfram_script(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram")
        .join(name)
}

/// Reference payload for a case: live engine output when `WOLFRAMSCRIPT`
/// is set, otherwise the checked-in golden. Returns `None` (loudly) when
/// neither is available so optional probes skip instead of failing blind.
///
/// Required comparison tests must use [`require_reference`] instead: a
/// missing reference is an incomplete gate, never a pass.
pub fn reference(case: &str, script: &str) -> Option<serde_json::Value> {
    match env::var_os("WOLFRAMSCRIPT") {
        Some(executable) => run_wolfram_script_with(&executable, script),
        None => reference_with(case, &golden_dir()),
    }
}

/// Required-gate reference loader: like [`reference`], but fails the test
/// when neither a live engine nor a checked-in golden is available.
///
/// This is the catalogue's strict missing-reference rule: the gate must
/// fail or explicitly report unavailable — it must never pass by
/// executing zero comparisons.
pub fn require_reference(case: &str, script: &str) -> serde_json::Value {
    match reference(case, script) {
        Some(payload) => payload,
        None => panic!(
            "UNAVAILABLE {case}: no live engine (WOLFRAMSCRIPT unset) and no golden for \
             script {script} (run `just qa-wolfram-goldens`). \
             Unavailable counts as incomplete, never as a pass."
        ),
    }
}

/// [`reference`] against an explicit golden directory (no env lookup), so
/// the golden/skip branches are unit-testable without touching process env.
pub fn reference_with(case: &str, golden_dir: &std::path::Path) -> Option<serde_json::Value> {
    let path = golden_dir.join(format!("{case}.json"));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(
            serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("golden {} is not JSON: {error}", path.display())),
        ),
        Err(_) => {
            println!(
                "SKIPPED {case}: no golden at {} and WOLFRAMSCRIPT unset (run `just qa-wolfram-goldens`)",
                path.display()
            );
            None
        }
    }
}

/// Run a `.wls` script through the live engine; `None` when the engine
/// binary is missing. Follows the `math-qa`/`sonium-qa` invocation pattern,
/// including the macOS `WolframKernel` fallback. Takes the engine binary
/// explicitly (no env lookup) so the launch/parse branches are
/// unit-testable with a fixture script; [`reference`] wires in
/// `WOLFRAMSCRIPT`.
pub fn run_wolfram_script_with(
    executable: &std::ffi::OsStr,
    script: &str,
) -> Option<serde_json::Value> {
    let path = wolfram_script(script);
    let mut command = Command::new(executable);
    command.arg("-file").arg(&path);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("WOLFRAMSCRIPT binary not found; skipping live reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram script {script} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram script {script} output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

/// Structured pass record printed by every comparison test.
#[derive(Debug, Clone, Serialize)]
pub struct QaResult {
    /// Manifest case id (e.g. `autoeq-qa.peq-response.v1`).
    pub case: String,
    /// True when the comparison passed.
    pub pass: bool,
    /// Worst relative error observed (0.0 for absolute-only cases).
    pub max_rel_error: f64,
    /// Worst absolute error observed.
    pub max_abs_error: f64,
    /// Case tolerance from the manifest.
    pub tolerance: f64,
    /// `rel` or `abs`: which error the tolerance applies to.
    pub tolerance_kind: String,
    /// `live-engine` or `checked-in-golden`.
    pub provenance: String,
}

/// Emit one `QA_RESULT:` JSON line.
pub fn emit_result(result: &QaResult) {
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string())
    );
}

/// Whether the reference came from the live engine.
pub fn provenance() -> String {
    provenance_with(env::var_os("WOLFRAMSCRIPT").is_some())
}

/// [`provenance`] over an explicit live flag (no env lookup).
pub fn provenance_with(live: bool) -> String {
    if live {
        "live-engine".to_string()
    } else {
        "checked-in-golden".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_error_handles_zeros() {
        assert_eq!(rel_error(0.0, 0.0), 0.0);
        assert_eq!(rel_error(1.0, 1.0), 0.0);
        assert_eq!(rel_error(1.0, 0.0), f64::INFINITY);
        assert!(rel_error(f64::NAN, 1.0).is_nan());
        assert!((rel_error(1.1, 1.0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn complex_rel_error_is_difference_over_reference() {
        use num_complex::Complex64;
        assert_eq!(
            complex_rel_error(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            0.0
        );
        assert_eq!(
            complex_rel_error(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
            0.0
        );
        // |(1+i) − 1| / |1| = |i| = 1.
        assert!(
            (complex_rel_error(Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)) - 1.0).abs()
                < 1e-12
        );
        assert_eq!(
            complex_rel_error(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)),
            f64::INFINITY
        );
        assert!(
            complex_rel_error(Complex64::new(f64::NAN, 0.0), Complex64::new(1.0, 0.0)).is_nan()
        );
    }

    #[test]
    fn missing_golden_without_engine_is_none() {
        let dir = std::env::temp_dir().join("autoeq-qa-no-such-golden-dir");
        assert!(reference_with("no-such-case", &dir).is_none());
    }
}
