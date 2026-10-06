//! AutoEQ Benchmark CLI: runs optimization scenarios across cached speakers and writes CSV results
//!
//! Scenarios per speaker:
//! 1) --loss speaker-flat --measurement CEA2034 --curve-name "Listening Window"
//! 2) --loss speaker-flat --measurement "Estimated In-Room Response" --curve-name "Estimated In-Room Response"
//! 3) --loss speaker-score --measurement CEA2034 --algo mh:rga
//! 4) --loss speaker-score --measurement CEA2034 --algo mh:pso
//! 5) --loss speaker-score --measurement CEA2034 --algo autoeq:de
//! 6) --loss speaker-score --measurement CEA2034 --algo autoeq:cmaes
//!
//! Input data is expected under data_cached/speakers/org.spinorama/{speaker}/{measurement}.json (Plotly JSON),
//! optionally data_cached/speakers/org.spinorama/{speaker}/metadata.json for metadata preference score.

use clap::Parser;
use consts::PAIR_TIE_EPS;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::select;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;

#[path = "benchmark/bench_row.rs"]
mod bench_row;
#[path = "benchmark/consts.rs"]
mod consts;
#[path = "benchmark/misc.rs"]
mod misc;
#[path = "benchmark/print.rs"]
mod print;
#[cfg(test)]
#[path = "benchmark/tests.rs"]
mod tests;
#[path = "benchmark/types.rs"]
mod types;

pub use types::*;

use bench_row::BenchRow;
use consts::CSV_HEADER;
use consts::DATA_CACHED;
use consts::DATA_GENERATED;
use consts::SCORE_OPTIMIZER_LABELS;
use consts::tied_best_mask;
use misc::finite_diff;
use misc::fmt_opt_f64;
use misc::list_speakers;
use misc::percentage;
use misc::push_finite_diff;
use misc::read_metadata_pref_score;
use misc::run_one;
use misc::{record_scenario, require_complete_benchmark};
use print::print_distribution_stats;
use print::print_pairwise_stats;
use std::sync::Mutex;

#[tokio::main]
pub async fn run_command() -> Result<(), Box<dyn Error>> {
    let args = BenchArgs::parse();

    // Check if user wants to see algorithm list
    if args.base.algo_list {
        autoeq::cli::display_algorithm_list();
    }

    // Validate before starting the signal listener, whose lifetime is then
    // bounded by the result of `run_benchmark` below.
    autoeq::cli::validate_args_or_exit(&args.base);

    // Set up signal handling for graceful shutdown
    let shutdown = ShutdownSignal::new();
    let shutdown_clone = shutdown.clone();

    // Spawn a dedicated signal handler task
    let signal_task = SignalTaskGuard::new(tokio::spawn(async move {
        loop {
            if let Err(e) = tokio::signal::ctrl_c().await {
                eprintln!("⚠️ Error setting up signal handler: {}", e);
                break;
            }

            eprintln!("\n🛑 Received interrupt signal. Stopping benchmark work cooperatively...");
            eprintln!("📝 Press Ctrl+C again within 5 seconds to force immediate termination.");
            shutdown_clone.request();

            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(5)) => {}
                result = tokio::signal::ctrl_c() => {
                    if let Err(error) = result {
                        eprintln!("⚠️ Error waiting for second interrupt signal: {error}");
                    } else {
                        eprintln!("\n‼️ Received second interrupt signal. Forcing immediate termination!");
                        std::process::exit(130);
                    }
                }
            }
        }
    }));

    let result = run_benchmark(args, shutdown).await;
    signal_task.abort_and_join().await;
    result
}

struct SignalTaskGuard(Option<tokio::task::JoinHandle<()>>);

impl SignalTaskGuard {
    fn new(task: tokio::task::JoinHandle<()>) -> Self {
        Self(Some(task))
    }

    async fn abort_and_join(mut self) {
        if let Some(mut task) = self.0.take() {
            task.abort();
            let _ = (&mut task).await;
        }
    }
}

impl Drop for SignalTaskGuard {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            task.abort();
        }
    }
}

async fn run_benchmark(args: BenchArgs, shutdown: ShutdownSignal) -> Result<(), Box<dyn Error>> {
    // Enumerate speakers as subdirectories of ./data_cached/speakers/org.spinorama/
    let speakers_dir = PathBuf::from(DATA_CACHED)
        .join("speakers")
        .join("org.spinorama");
    let speakers = list_speakers(speakers_dir)?;
    let speakers: Vec<String> = if args.smoke_test {
        speakers.into_iter().take(5).collect()
    } else {
        speakers
    };
    if speakers.is_empty() {
        eprintln!("No speakers found under ./data. Exiting.");
        return if args.base.qa.is_some() {
            Err("Benchmark QA requires a nonempty speaker corpus".into())
        } else {
            Ok(())
        };
    }

    // Determine parallelism
    let jobs = if args.jobs > 0 {
        args.jobs
    } else {
        num_cpus::get()
    };
    eprintln!("Running benchmark with {} parallel jobs", jobs);
    eprintln!("Press Ctrl+C to gracefully stop the benchmark and save partial results...");

    // Prepare the output before any workers start. If the destination cannot
    // be opened or the header cannot be written, there is no blocking work to
    // cancel or drain.
    let mut wtr =
        csv::Writer::from_path(std::path::Path::new(DATA_GENERATED).join("benchmark.csv"))?;
    wtr.write_record(CSV_HEADER)?;
    wtr.flush()?;

    // Channel for rows; writer runs on main task
    let (tx, mut rx) = mpsc::channel::<BenchRow>(jobs * 2);
    let sem = std::sync::Arc::new(Semaphore::new(jobs));
    let mut set = JoinSet::new();

    let scenario_failures = Arc::new(Mutex::new(Vec::new()));
    for speaker in speakers.clone() {
        let tx = tx.clone();
        let scenario_failures = scenario_failures.clone();
        let sem = sem.clone();
        let base_args = args.base.clone();
        let shutdown_clone = shutdown.clone();
        set.spawn(async move {
            let Some(_permit) = acquire_slot_or_shutdown(sem, shutdown_clone.clone()).await else {
                return;
            };

            // Check for shutdown signal before starting work
            if shutdown_clone.is_requested() {
                return;
            }

            // For local cache usage, version value is irrelevant provided cache exists.
            let version = "latest".to_string();

            // Scenario 1
            let mut a1 = base_args.clone();
            a1.speaker = Some(speaker.clone());
            a1.version = Some(version.clone());
            a1.measurement = Some("CEA2034".to_string());
            a1.curve_name = "Listening Window".to_string();
            a1.loss = autoeq::LossType::SpeakerFlat;
            let s1 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "flat_cea2034_lw",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a1, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "flat_cea2034_lw",
                    &scenario_failures,
                )
            };

            // Scenario 2
            let mut a2 = base_args.clone();
            a2.speaker = Some(speaker.clone());
            a2.version = Some(version.clone());
            a2.measurement = Some("Estimated In-Room Response".to_string());
            a2.curve_name = "Estimated In-Room Response".to_string();
            a2.loss = autoeq::LossType::SpeakerFlat;
            let s2 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "flat_eir",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a2, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "flat_eir",
                    &scenario_failures,
                )
            };

            // Scenario 3: Score loss with mh:rga
            let mut a3 = base_args.clone();
            a3.speaker = Some(speaker.clone());
            a3.version = Some(version.clone());
            a3.measurement = Some("CEA2034".to_string());
            a3.loss = autoeq::LossType::SpeakerScore;
            a3.algo = "mh:rga".to_string();
            let s3 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "score_mh_rga",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a3, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "score_mh_rga",
                    &scenario_failures,
                )
            };

            // Scenario 4: Score loss with mh:pso
            let mut a4 = base_args.clone();
            a4.speaker = Some(speaker.clone());
            a4.version = Some(version.clone());
            a4.measurement = Some("CEA2034".to_string());
            a4.loss = autoeq::LossType::SpeakerScore;
            a4.algo = "mh:pso".to_string();
            let s4 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "score_mh_pso",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a4, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "score_mh_pso",
                    &scenario_failures,
                )
            };

            // Scenario 5: Score loss with autoeq:de
            let mut a5 = base_args.clone();
            a5.speaker = Some(speaker.clone());
            a5.version = Some(version.clone());
            a5.measurement = Some("CEA2034".to_string());
            a5.loss = autoeq::LossType::SpeakerScore;
            a5.algo = "autoeq:de".to_string();
            let s5 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "score_autoeq_de",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a5, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "score_autoeq_de",
                    &scenario_failures,
                )
            };

            // Scenario 6: Score loss with autoeq:cmaes
            let mut a6 = base_args.clone();
            a6.speaker = Some(speaker.clone());
            a6.version = Some(version.clone());
            a6.measurement = Some("CEA2034".to_string());
            a6.loss = autoeq::LossType::SpeakerScore;
            a6.algo = "autoeq:cmaes".to_string();
            let s6 = if shutdown_clone.is_requested() {
                record_scenario(
                    Err("Scenario cancelled before evaluation".to_string()),
                    &speaker,
                    "score_autoeq_cmaes",
                    &scenario_failures,
                )
            } else {
                record_scenario(
                    run_one(&a6, shutdown_clone.clone())
                        .await
                        .map(|(metrics, qa_failure)| (metrics.pref_score, qa_failure)),
                    &speaker,
                    "score_autoeq_cmaes",
                    &scenario_failures,
                )
            };

            // Metadata preference
            let meta_pref = read_metadata_pref_score(&speaker).ok().flatten();

            let _ = tx
                .send(BenchRow {
                    speaker,
                    flat_cea2034_lw: s1,
                    flat_eir: s2,
                    score_cea2034_mh_rga: s3,
                    score_cea2034_mh_pso: s4,
                    score_cea2034_autoeq_de: s5,
                    score_cea2034_autoeq_cmaes: s6,
                    metadata_pref: meta_pref,
                })
                .await;
        });
    }
    drop(tx); // close sender when tasks finish

    // Collect deltas (scenario - metadata) for end-of-run statistics
    let mut deltas_s1: Vec<f64> = Vec::new();
    let mut deltas_s2: Vec<f64> = Vec::new();
    let mut deltas_s3: Vec<f64> = Vec::new();
    let mut deltas_s4: Vec<f64> = Vec::new();
    let mut deltas_s5: Vec<f64> = Vec::new();
    let mut deltas_s6: Vec<f64> = Vec::new();
    let mut deltas_pso_vs_rga: Vec<f64> = Vec::new();
    let mut deltas_de_vs_rga: Vec<f64> = Vec::new();
    let mut deltas_cmaes_vs_rga: Vec<f64> = Vec::new();
    let mut deltas_de_vs_pso: Vec<f64> = Vec::new();
    let mut deltas_cmaes_vs_pso: Vec<f64> = Vec::new();
    let mut deltas_cmaes_vs_de: Vec<f64> = Vec::new();
    let mut complete_score_rows = 0usize;
    let mut tied_best_counts = [0usize; SCORE_OPTIMIZER_LABELS.len()];

    let mut completed_speakers = 0;
    let total_speakers = speakers.len();
    let mut shutdown_seen = false;
    let mut receiver_closed = false;
    let mut worker_failures = Vec::new();
    let mut first_record_error = None;

    // Drain rows and join wrappers concurrently. A panicking worker requests
    // cancellation immediately; queued jobs then leave without starting.
    while !receiver_closed || !set.is_empty() {
        select! {
            result = rx.recv(), if !receiver_closed => {
                match result {
                    Some(row) => {
                        completed_speakers += 1;
                        eprintln!(
                            "Completed {}/{} speakers: {}",
                            completed_speakers, total_speakers, row.speaker
                        );

                        write_bench_row(
                            &mut wtr,
                            &row,
                            &shutdown,
                            &mut first_record_error,
                        );

                        // Accumulate deltas vs metadata when both values are present and finite
                        if let (Some(v), Some(m)) = (row.flat_cea2034_lw, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s1.push(v - m);
                            }
                        if let (Some(v), Some(m)) = (row.flat_eir, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s2.push(v - m);
                            }
                        if let (Some(v), Some(m)) = (row.score_cea2034_mh_rga, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s3.push(v - m);
                            }
                        if let (Some(v), Some(m)) = (row.score_cea2034_mh_pso, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s4.push(v - m);
                            }
                        if let (Some(v), Some(m)) = (row.score_cea2034_autoeq_de, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s5.push(v - m);
                            }
                        if let (Some(v), Some(m)) = (row.score_cea2034_autoeq_cmaes, row.metadata_pref)
                            && v.is_finite() && m.is_finite() {
                                deltas_s6.push(v - m);
                            }
                        push_finite_diff(
                            &mut deltas_pso_vs_rga,
                            row.score_cea2034_mh_pso,
                            row.score_cea2034_mh_rga,
                        );
                        push_finite_diff(
                            &mut deltas_de_vs_rga,
                            row.score_cea2034_autoeq_de,
                            row.score_cea2034_mh_rga,
                        );
                        push_finite_diff(
                            &mut deltas_cmaes_vs_rga,
                            row.score_cea2034_autoeq_cmaes,
                            row.score_cea2034_mh_rga,
                        );
                        push_finite_diff(
                            &mut deltas_de_vs_pso,
                            row.score_cea2034_autoeq_de,
                            row.score_cea2034_mh_pso,
                        );
                        push_finite_diff(
                            &mut deltas_cmaes_vs_pso,
                            row.score_cea2034_autoeq_cmaes,
                            row.score_cea2034_mh_pso,
                        );
                        if let Some(delta) = finite_diff(
                            row.score_cea2034_autoeq_cmaes,
                            row.score_cea2034_autoeq_de,
                        ) {
                            deltas_cmaes_vs_de.push(delta);
                        }
                        let score_values = [
                            row.score_cea2034_mh_rga,
                            row.score_cea2034_mh_pso,
                            row.score_cea2034_autoeq_de,
                            row.score_cea2034_autoeq_cmaes,
                        ];
                        if let Some(best_mask) = tied_best_mask(score_values) {
                            complete_score_rows += 1;
                            for (idx, is_best) in best_mask.iter().copied().enumerate() {
                                if is_best {
                                    tied_best_counts[idx] += 1;
                                }
                            }
                        }
                    }
                    None => {
                        receiver_closed = true;
                    }
                }
            }
            result = set.join_next(), if !set.is_empty() => {
                if let Some(Err(error)) = result {
                    remember_worker_failure(error, &mut worker_failures, &shutdown);
                }
            }
            _ = shutdown.cancelled(), if !shutdown_seen => {
                shutdown_seen = true;
                eprintln!("\n🛑 Shutdown signal detected. Waiting for active optimizers to stop and flushing completed rows...");
            }
        }
    }

    // The channel closes only after every worker has sent its row and dropped
    // its sender. Join wrappers before flushing so started blocking optimizers
    // cannot outlive the partial CSV publication.
    let drain_error = join_workers_then_flush(&mut set, worker_failures, || wtr.flush()).await;
    if first_record_error.is_some() || drain_error.is_some() {
        return Err(Box::new(BenchmarkRunError {
            record_error: first_record_error,
            drain_error,
        }));
    }

    if completed_speakers < total_speakers {
        eprintln!(
            "⚠️  Benchmark incomplete: {}/{} speakers processed due to early termination.",
            completed_speakers, total_speakers
        );
    } else {
        eprintln!(
            "✅ Benchmark completed successfully: {}/{} speakers processed.",
            completed_speakers, total_speakers
        );
    }

    // Print end-of-run statistics comparing scenarios to metadata.
    eprintln!("\n=== Benchmark statistics (scenario - metadata) ===");
    eprintln!("Closer to 0 means closer to metadata preference score.");
    print_distribution_stats("flat_cea2034_lw", &deltas_s1);
    print_distribution_stats("flat_eir", &deltas_s2);
    print_distribution_stats("score_mh_rga", &deltas_s3);
    print_distribution_stats("score_mh_pso", &deltas_s4);
    print_distribution_stats("score_de", &deltas_s5);
    print_distribution_stats("score_cmaes", &deltas_s6);

    eprintln!("\n=== Paired optimizer deltas (left - right preference score) ===");
    eprintln!(
        "Positive mean/median means the left optimizer scored higher. Ties use ±{PAIR_TIE_EPS:.0e}."
    );
    print_pairwise_stats("mh:pso - mh:rga", &deltas_pso_vs_rga);
    print_pairwise_stats("autoeq:de - mh:rga", &deltas_de_vs_rga);
    print_pairwise_stats("autoeq:cmaes - mh:rga", &deltas_cmaes_vs_rga);
    print_pairwise_stats("autoeq:de - mh:pso", &deltas_de_vs_pso);
    print_pairwise_stats("autoeq:cmaes - mh:pso", &deltas_cmaes_vs_pso);
    print_pairwise_stats("autoeq:cmaes - autoeq:de", &deltas_cmaes_vs_de);

    eprintln!("\n=== Tied-best counts among complete score-optimizer rows ===");
    eprintln!("Rows with all score optimizers present: {complete_score_rows}");
    for (label, count) in SCORE_OPTIMIZER_LABELS.iter().zip(tied_best_counts) {
        let pct = percentage(count, complete_score_rows);
        eprintln!("{label:>20}: tied-best={count:>4} ({pct:>5.1}%)");
    }

    let failures = scenario_failures
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for failure in failures.iter() {
        eprintln!("Scenario failure: {failure}");
    }
    if args.base.qa.is_some() {
        require_complete_benchmark(total_speakers, completed_speakers, &failures)?;
    }
    Ok(())
}

async fn acquire_slot_or_shutdown(
    semaphore: Arc<Semaphore>,
    shutdown: ShutdownSignal,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => None,
        permit = semaphore.acquire_owned() => permit.ok(),
    }
}

async fn join_workers_then_flush<F>(
    set: &mut JoinSet<()>,
    mut worker_failures: Vec<String>,
    flush: F,
) -> Option<BenchmarkDrainError>
where
    F: FnOnce() -> std::io::Result<()>,
{
    while let Some(result) = set.join_next().await {
        if let Err(error) = result {
            worker_failures.push(error.to_string());
        }
    }
    let flush_error = flush().err();
    if worker_failures.is_empty() && flush_error.is_none() {
        None
    } else {
        Some(BenchmarkDrainError {
            worker_failures,
            flush_error,
        })
    }
}

fn remember_worker_failure(
    error: tokio::task::JoinError,
    worker_failures: &mut Vec<String>,
    shutdown: &ShutdownSignal,
) {
    worker_failures.push(error.to_string());
    shutdown.request();
}

fn write_bench_row<W: io::Write>(
    writer: &mut csv::Writer<W>,
    row: &BenchRow,
    shutdown: &ShutdownSignal,
    first_error: &mut Option<csv::Error>,
) {
    if first_error.is_some() {
        return;
    }
    let record_result = writer.write_record([
        row.speaker.as_str(),
        fmt_opt_f64(row.flat_cea2034_lw).as_str(),
        fmt_opt_f64(row.flat_eir).as_str(),
        fmt_opt_f64(row.score_cea2034_mh_rga).as_str(),
        fmt_opt_f64(row.score_cea2034_mh_pso).as_str(),
        fmt_opt_f64(row.score_cea2034_autoeq_de).as_str(),
        fmt_opt_f64(row.score_cea2034_autoeq_cmaes).as_str(),
        fmt_opt_f64(finite_diff(
            row.score_cea2034_mh_rga,
            row.score_cea2034_autoeq_de,
        ))
        .as_str(),
        fmt_opt_f64(finite_diff(
            row.score_cea2034_mh_pso,
            row.score_cea2034_autoeq_de,
        ))
        .as_str(),
        fmt_opt_f64(finite_diff(
            row.score_cea2034_autoeq_cmaes,
            row.score_cea2034_autoeq_de,
        ))
        .as_str(),
        fmt_opt_f64(row.metadata_pref).as_str(),
    ]);
    if let Err(error) = record_result {
        *first_error = Some(error);
        shutdown.request();
        return;
    }
    if let Err(error) = writer.flush() {
        *first_error = Some(error.into());
        shutdown.request();
    }
}

#[derive(Debug)]
struct BenchmarkDrainError {
    worker_failures: Vec<String>,
    flush_error: Option<io::Error>,
}

impl fmt::Display for BenchmarkDrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.worker_failures.is_empty() {
            write!(
                f,
                "benchmark worker failure(s): {}",
                self.worker_failures.join("; ")
            )?;
        }
        if let Some(error) = &self.flush_error {
            if !self.worker_failures.is_empty() {
                f.write_str("; ")?;
            }
            write!(f, "failed to flush benchmark CSV: {error}")?;
        }
        Ok(())
    }
}

impl Error for BenchmarkDrainError {}

#[derive(Debug)]
struct BenchmarkRunError {
    record_error: Option<csv::Error>,
    drain_error: Option<BenchmarkDrainError>,
}

impl fmt::Display for BenchmarkRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = &self.record_error {
            write!(f, "failed to write benchmark CSV row: {error}")?;
        }
        if let Some(error) = &self.drain_error {
            if self.record_error.is_some() {
                f.write_str("; after draining workers: ")?;
            }
            write!(f, "{error}")?;
        }
        Ok(())
    }
}

impl Error for BenchmarkRunError {}

#[cfg(test)]
mod worker_drain_tests {
    use std::io;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::misc::{ActiveRunControl, BlockingOutcome, await_blocking_worker};
    use super::{
        BenchRow, ShutdownSignal, SignalTaskGuard, join_workers_then_flush,
        remember_worker_failure, write_bench_row,
    };

    struct FailingWriter;

    impl io::Write for FailingWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "injected CSV failure",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "injected flush failure",
            ))
        }
    }

    fn row() -> BenchRow {
        BenchRow {
            speaker: "test-speaker".into(),
            flat_cea2034_lw: Some(1.0),
            flat_eir: None,
            score_cea2034_mh_rga: None,
            score_cea2034_mh_pso: None,
            score_cea2034_autoeq_de: None,
            score_cea2034_autoeq_cmaes: None,
            metadata_pref: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn csv_record_failure_cancels_and_joins_active_optimizer() {
        let shutdown = ShutdownSignal::new();
        let active = ActiveRunControl::default();
        let worker_active = active.clone();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_worker = Arc::clone(&completed);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let worker = tokio::task::spawn_blocking(move || {
            let control = autoeq::optim::run_control::OptimizerRunControl::new(
                NonZeroUsize::new(8).expect("nonzero test budget"),
            );
            worker_active.register(control.clone());
            let _ = started_tx.send(());
            while !control.stop_requested() {
                std::thread::yield_now();
            }
            completed_worker.store(true, Ordering::Release);
        });
        started_rx.await.expect("optimizer worker started");

        let mut writer = csv::WriterBuilder::new()
            .buffer_capacity(1)
            .from_writer(FailingWriter);
        let mut first_error = None;

        write_bench_row(&mut writer, &row(), &shutdown, &mut first_error);
        let first_message = first_error
            .as_ref()
            .expect("injected writer fails while recording")
            .to_string();
        assert!(shutdown.is_requested());

        write_bench_row(&mut writer, &row(), &shutdown, &mut first_error);
        assert_eq!(
            first_error.expect("first error retained").to_string(),
            first_message
        );

        let outcome = await_blocking_worker(worker, shutdown, active)
            .await
            .expect("optimizer worker join succeeds");
        assert!(matches!(outcome, BlockingOutcome::Cancelled(())));
        assert!(completed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn worker_panic_is_reported_after_other_workers_and_flush_complete() {
        let shutdown = ShutdownSignal::new();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_worker = Arc::clone(&completed);
        let mut workers = tokio::task::JoinSet::new();
        workers.spawn(async { panic!("injected worker panic") });
        workers.spawn(async move {
            completed_worker.store(true, Ordering::Release);
        });

        let result = workers.join_next().await.expect("panic result");
        let mut failures = Vec::new();
        if let Err(error) = result {
            remember_worker_failure(error, &mut failures, &shutdown);
        }
        assert!(shutdown.is_requested());

        let flushed = Arc::new(AtomicBool::new(false));
        let flushed_after_drain = Arc::clone(&flushed);
        let error = join_workers_then_flush(&mut workers, failures, || {
            assert!(completed.load(Ordering::Acquire));
            flushed_after_drain.store(true, Ordering::Release);
            Ok(())
        })
        .await
        .expect("a worker panic must be returned after drain");

        assert!(error.worker_failures[0].contains("injected worker panic"));
        assert!(error.flush_error.is_none());
        assert!(completed.load(Ordering::Acquire));
        assert!(flushed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn signal_task_guard_aborts_and_joins_listener() {
        struct MarkDropped(Arc<AtomicBool>);

        impl Drop for MarkDropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_in_task = Arc::clone(&dropped);
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let listener = tokio::spawn(async move {
            let _mark_dropped = MarkDropped(dropped_in_task);
            let _ = started_tx.send(());
            std::future::pending::<()>().await;
        });
        let guard = SignalTaskGuard::new(listener);
        started_rx.await.expect("listener started");

        guard.abort_and_join().await;

        assert!(dropped.load(Ordering::Acquire));
    }
}
