//! Thin compatibility launcher for the crate-owned RoomEQ command.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&shutdown);
    let signal_listener = tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            eprintln!("Could not install Ctrl-C listener: {error}");
            return;
        }
        signal_flag.store(true, Ordering::Release);
    });

    let command_shutdown = Arc::clone(&shutdown);
    let command = tokio::task::spawn_blocking(move || {
        roomeq_cli::roomeq::run_command_with_shutdown(command_shutdown)
    })
    .await;

    signal_listener.abort();
    let _ = signal_listener.await;
    command??;
    Ok(())
}
