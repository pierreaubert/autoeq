//! Thin compatibility launcher for the crate-owned RoomEQ command.

use std::future::{Future, poll_fn};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use tokio::task::JoinHandle;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let signal_listener = start_ctrl_c_listener(Arc::clone(&shutdown)).await?;

    // A signal can arrive while the listener is registering. Do not start
    // synchronous preparation if that signal has already requested shutdown.
    if shutdown.load(Ordering::Acquire) {
        signal_listener.await??;
        anyhow::bail!("RoomEQ cancelled before command startup");
    }

    let command_shutdown = Arc::clone(&shutdown);
    let command = tokio::task::spawn_blocking(move || {
        roomeq_cli::roomeq::run_command_with_shutdown(command_shutdown)
    })
    .await;

    signal_listener.abort();
    match signal_listener.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(error.into()),
        Err(error) if error.is_cancelled() => {}
        Err(error) => return Err(error.into()),
    }
    command??;
    Ok(())
}

async fn start_ctrl_c_listener(
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<JoinHandle<io::Result<()>>> {
    start_signal_listener(tokio::signal::ctrl_c(), shutdown).await
}

async fn start_signal_listener<F>(
    signal_future: F,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<JoinHandle<io::Result<()>>>
where
    F: Future<Output = io::Result<()>> + Send + 'static,
{
    let (ready_sender, ready_receiver) = tokio::sync::oneshot::channel();
    let listener = tokio::spawn(async move {
        let mut signal_future = Box::pin(signal_future);
        let mut ready_sender = Some(ready_sender);
        poll_fn(move |context| {
            let result = signal_future.as_mut().poll(context);
            if matches!(&result, Poll::Ready(Ok(()))) {
                shutdown.store(true, Ordering::Release);
            }
            if let Some(sender) = ready_sender.take() {
                let readiness = match &result {
                    Poll::Ready(Err(error)) => Err(error.to_string()),
                    Poll::Pending | Poll::Ready(Ok(())) => Ok(()),
                };
                let _ = sender.send(readiness);
            }
            result
        })
        .await
    });

    match ready_receiver.await {
        Ok(Ok(())) => Ok(listener),
        Ok(Err(error)) => match listener.await {
            Ok(Err(_)) => Err(anyhow::anyhow!(
                "Could not install Ctrl-C listener: {error}"
            )),
            Ok(Ok(())) => Err(anyhow::anyhow!(
                "Ctrl-C listener exited before it became ready"
            )),
            Err(join_error) => Err(anyhow::anyhow!(
                "Ctrl-C listener task failed during registration: {join_error}"
            )),
        },
        Err(channel_error) => {
            let listener_result = listener.await;
            Err(anyhow::anyhow!(
                "Ctrl-C listener exited before registration readiness: {channel_error}; task result: {listener_result:?}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::start_signal_listener;
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn ready_signal_sets_shutdown_before_readiness_returns() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let listener = start_signal_listener(std::future::ready(Ok(())), Arc::clone(&shutdown))
            .await
            .expect("register immediately-ready signal");

        assert!(shutdown.load(Ordering::Acquire));
        assert!(listener.await.expect("listener task").is_ok());
    }

    #[tokio::test]
    async fn signal_registration_error_is_returned_before_readiness() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let result = start_signal_listener(
            std::future::ready(Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected registration failure",
            ))),
            Arc::clone(&shutdown),
        )
        .await;

        let error = result.expect_err("registration failure must not start the command");
        assert!(
            error
                .to_string()
                .contains("Could not install Ctrl-C listener")
        );
        assert!(!shutdown.load(Ordering::Acquire));
    }
}
