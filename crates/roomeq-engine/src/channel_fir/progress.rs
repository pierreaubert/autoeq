//! Share the user's progress callback across IIR and spatial FIR stages.
use autoeq_core::{AutoeqError, Result};
use autoeq_optim::optim::{
    OptimProgressCallback,
    scalar::{
        ScalarOptimConfig, ScalarOptimResult, optimize_bounded_scalar,
        optimize_bounded_scalar_with_callback,
    },
};
use std::sync::{Arc, Mutex};

pub(super) struct FirProgress(Option<Arc<Mutex<OptimProgressCallback>>>);

impl FirProgress {
    pub fn new(callback: Option<OptimProgressCallback>) -> Self {
        Self(callback.map(|callback| Arc::new(Mutex::new(callback))))
    }

    pub fn callback(&self) -> Option<OptimProgressCallback> {
        self.0.as_ref().map(|callback| {
            let callback = callback.clone();
            Box::new(move |iteration, loss, preference| {
                callback.lock().unwrap()(iteration, loss, preference)
            }) as OptimProgressCallback
        })
    }

    pub fn check(&self, iteration: usize, loss: f64) -> Result<()> {
        if let Some(mut callback) = self.callback()
            && matches!(
                callback(iteration, loss, None),
                autoeq_optim::de::CallbackAction::Stop
            )
        {
            return Err(AutoeqError::OptimizationFailed {
                message: "Hybrid FIR stopped by progress callback".into(),
            });
        }
        Ok(())
    }

    pub fn optimize<F>(
        &self,
        bounds: &[(f64, f64)],
        initial: &[f64],
        config: &ScalarOptimConfig,
        objective: F,
    ) -> std::result::Result<ScalarOptimResult, String>
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        let native = autoeq_optim::optim::registry::resolve(&config.algorithm)
            .is_some_and(|backend| matches!(backend.name(), "autoeq:de" | "autoeq:cmaes"));
        let mut result = if native {
            optimize_bounded_scalar_with_callback(
                bounds,
                initial,
                config,
                objective,
                self.callback(),
            )?
        } else {
            if self.0.is_some() {
                log::warn!(
                    "Hybrid FIR {} has stage-boundary cancellation only; the pinned backend has no native scalar stop hook",
                    config.algorithm
                );
            }
            optimize_bounded_scalar(bounds, initial, config, objective)?
        };
        // This is a final checkpoint, not a claim of in-flight cancellation on
        // unsupported backends. Never deliver a candidate after observed Stop.
        // Zero is a stage-boundary checkpoint, not a fabricated iteration
        // count inferred from the configured budget.
        self.check(0, result.fun)
            .map_err(|error| error.to_string())?;
        if self.0.is_some() {
            result.message.push_str(if native {
                "; callback_cancellation=native_generation_boundaries"
            } else {
                "; callback_cancellation=stage_boundaries_only"
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_capability_is_retained_in_completed_search_evidence() {
        for (algorithm, capability) in [
            ("autoeq:de", "native_generation_boundaries"),
            ("autoeq:cmaes", "native_generation_boundaries"),
            ("autoeq:cobyla", "stage_boundaries_only"),
            ("autoeq:isres", "stage_boundaries_only"),
        ] {
            let progress = FirProgress::new(Some(Box::new(|_, _, _| {
                autoeq_optim::de::CallbackAction::Continue
            })));
            let result = progress
                .optimize(
                    &[(-1.0, 1.0)],
                    &[0.7],
                    &ScalarOptimConfig {
                        algorithm: algorithm.into(),
                        max_iter: 30,
                        population: 4,
                        seed: Some(7),
                        ..Default::default()
                    },
                    |x| x[0] * x[0],
                )
                .unwrap();
            assert!(
                result
                    .message
                    .contains(&format!("callback_cancellation={capability}"))
            );
        }
    }

    #[test]
    fn boundary_only_backends_discard_result_after_observed_stop() {
        for algorithm in ["autoeq:cobyla", "autoeq:isres"] {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observed = calls.clone();
            let progress = FirProgress::new(Some(Box::new(move |iteration, loss, _| {
                assert_eq!(iteration, 0);
                assert!(loss.is_finite());
                observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                autoeq_optim::de::CallbackAction::Stop
            })));
            let result = progress.optimize(
                &[(-1.0, 1.0)],
                &[0.7],
                &ScalarOptimConfig {
                    algorithm: algorithm.into(),
                    max_iter: 30,
                    population: 4,
                    seed: Some(7),
                    ..Default::default()
                },
                |x| x[0] * x[0],
            );
            assert!(result.unwrap_err().contains("stopped by progress callback"));
            assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        }
    }
}
