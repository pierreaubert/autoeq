use std::sync::{Condvar, Mutex};

/// Counting-semaphore permit manager — same pattern as `roomeq-qa-coverage`.
/// Used to bound the number of test cases running concurrently.
#[derive(Debug)]
pub(super) struct CountingSemaphore {
    state: Mutex<usize>,
    cvar: Condvar,
}

/// One QA job slot, returned on success, error, or panic unwinding.
#[derive(Debug)]
#[must_use = "retain the permit until the QA case finishes"]
pub(super) struct Permit<'a>(&'a CountingSemaphore);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut count = self.0.state.lock().unwrap();
        *count += 1;
        self.0.cvar.notify_one();
    }
}

impl CountingSemaphore {
    pub(super) fn new(permits: usize) -> Self {
        Self {
            state: Mutex::new(permits),
            cvar: Condvar::new(),
        }
    }

    pub(super) fn acquire(&self) -> Permit<'_> {
        let mut count = self.state.lock().unwrap();
        while *count == 0 {
            count = self.cvar.wait(count).unwrap();
        }
        *count -= 1;
        Permit(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qa_permit_returns_on_early_error_and_panic() {
        let semaphore = CountingSemaphore::new(1);
        let failure = (|| -> Result<(), &'static str> {
            let _permit = semaphore.acquire();
            Err("case failed")?;
            Ok(())
        })();
        assert_eq!(failure, Err("case failed"));
        assert_eq!(*semaphore.state.lock().unwrap(), 1);
        let panic = std::panic::catch_unwind(|| {
            let _permit = semaphore.acquire();
            panic!("case panicked");
        });
        assert!(panic.is_err());
        assert_eq!(*semaphore.state.lock().unwrap(), 1);
    }

    #[test]
    fn qa_permit_release_wakes_waiting_case() {
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let semaphore = Arc::new(CountingSemaphore::new(1));
        let permit = semaphore.acquire();
        let worker_semaphore = Arc::clone(&semaphore);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let _permit = worker_semaphore.acquire();
            entered_tx.send(()).unwrap();
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            entered_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        drop(permit);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert_eq!(*semaphore.state.lock().unwrap(), 1);
    }
}
