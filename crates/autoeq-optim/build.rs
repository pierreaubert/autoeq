use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let rustc_version = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_owned())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=AUTOEQ_RUSTC_VERSION={rustc_version}");
}
