use rayon::{ThreadPool, ThreadPoolBuilder};
use roomeq_engine::error::{AutoeqError, Result};

/// Owns the single Rayon worker budget for a complete RoomEQ workflow.
///
/// Population-level optimizer tasks and outer channel tasks installed beneath
/// this pool share its workers, so nested parallelism cannot oversubscribe the
/// machine.
pub struct RoomEqExecutor {
    pool: ThreadPool,
}

impl RoomEqExecutor {
    pub fn new(worker_count: usize) -> Result<Self> {
        let worker_count = worker_count.max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("roomeq-worker-{index}"))
            .build()
            .map_err(|error| AutoeqError::InvalidConfiguration {
                message: format!("failed to build {worker_count}-worker RoomEQ pool: {error}"),
            })?;
        Ok(Self { pool })
    }

    pub fn worker_count(&self) -> usize {
        self.pool.current_num_threads()
    }

    pub fn install<Operation, Output>(&self, operation: Operation) -> Output
    where
        Operation: FnOnce() -> Output + Send,
        Output: Send,
    {
        self.pool.install(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn executor_uses_exact_owned_worker_budget() {
        let executor = RoomEqExecutor::new(3).expect("executor");
        // Work stealing need not visit every worker for a short parallel
        // iterator. Broadcast exercises each worker in the installed registry.
        let workers = executor.install(|| {
            rayon::broadcast(|context| {
                let index = rayon::current_thread_index().expect("inside pool");
                assert_eq!(index, context.index());
                assert_eq!(rayon::current_num_threads(), 3);
                assert_eq!(
                    std::thread::current().name(),
                    Some(format!("roomeq-worker-{index}").as_str())
                );
                index
            })
        });
        assert_eq!(executor.worker_count(), 3);
        assert_eq!(
            workers.into_iter().collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1, 2])
        );
    }
}
