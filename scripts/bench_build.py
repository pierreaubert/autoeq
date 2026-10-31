#!/usr/bin/env python3
"""Measure real edit feedback in independent source and target snapshots.

--crate selects focused library scenarios. --full additionally measures all
representative owners, affected consumers, complete native execution/build,
and shipping CLI/native-report/stable-WASM/nightly-GPUI paths. At least three
repetitions are required for acceptance; a one-repetition run is a smoke test.

The source checkout and any existing Cargo target remain untouched. --target-dir
is a legacy spelling for the scratch *parent*: a fresh owned child is created.
Normal timings exclude optional linker/LLVM diagnostic samples. Raw logs, Cargo
artifacts/timings, exact source identities, and per-repetition observations are
retained under the output directory. No speedup is inferred from target counts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import gzip
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.build_bench.execution import collect_timings, compiler_artifacts, dir_size_bytes, fingerprints, timed
from scripts.build_bench.plans import LIBRARY_SCENARIOS, WORKSPACE_SCENARIOS, affected_packages, cargo_cmd, cases, native_listing
from scripts.build_bench.probes import PUBLIC_TYPES, applied, select_edit
from scripts.build_bench.scopes import execute_matching_tests
from scripts.build_bench.snapshots import Storage, Toolchain, capture, dependency_roots, digest, metadata, snapshot

DEFAULT_OUTPUT = REPO / "docs/baselines/refactor-20261025"


def build_environment_identity(environment: dict[str, str]) -> dict:
    """Record build overrides that can change profile or native tool behavior."""
    names = {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER",
             "RUSTC_WORKSPACE_WRAPPER", "CARGO_INCREMENTAL", "CARGO_BUILD_TARGET",
             "RUSTUP_TOOLCHAIN", "MACOSX_DEPLOYMENT_TARGET", "SDKROOT", "CARGO_HOME",
             "CC", "CXX", "AR", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"}
    names.update(name for name in environment
                 if name.startswith(("CARGO_PROFILE_", "CARGO_TARGET_", "CARGO_BUILD_",
                                     "CC_", "CXX_", "AR_", "CFLAGS_", "CXXFLAGS_"))
                 and name != "CARGO_TARGET_DIR")
    return {name: environment.get(name) for name in sorted(names)}


def host_hardware() -> dict:
    if sys.platform == 'darwin':
        return {'cpu_model': capture(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
                'memory_bytes': int(capture(['/usr/sbin/sysctl', '-n', 'hw.memsize']))}
    return {'cpu_model': platform.processor(),
            'memory_bytes': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')}


def native_linker_identity(kind: str = 'system') -> dict:
    if kind not in {'system', 'lld-macho'}:
        raise ValueError(f'unsupported native linker: {kind}')
    if kind == 'lld-macho' and sys.platform != 'darwin':
        raise ValueError('Mach-O LLD experiments require a macOS host')
    driver = shutil.which('cc')
    linker = (shutil.which('ld64.lld') if kind == 'lld-macho' else
              capture(['xcrun', '--find', 'ld']) if sys.platform == 'darwin' else shutil.which('ld'))
    if not driver or not linker:
        raise RuntimeError('selected native compiler driver and linker must be available')
    version_flag = '-version_details' if sys.platform == 'darwin' and kind == 'system' else '--version'
    version = subprocess.run([linker, version_flag], check=True, capture_output=True, text=True)
    return {'kind': kind, 'driver_path': str(Path(driver).resolve()),
            'driver_sha256': digest(Path(driver)), 'linker_path': str(Path(linker).resolve()),
            'linker_invocation_path': str(Path(linker).absolute()),
            'version': (version.stdout + version.stderr).strip(), 'sha256': digest(Path(linker))}


def summarize(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in results:
        if not row.get("diagnostic"):
            groups[(row["case"], row["scenario"], row["profile"])].append(row)
    output = []
    for (case, scenario, profile_name), rows in sorted(groups.items()):
        walls = [row["wall_seconds"] for row in rows]
        output.append({"case": case, "scenario": scenario, "profile": profile_name,
                       "samples": len(rows), "median_wall_seconds": statistics.median(walls),
                       "min_wall_seconds": min(walls), "max_wall_seconds": max(walls),
                       "spread_seconds": max(walls) - min(walls)})
    return output


def cargo_shim(directory: Path, toolchain: Toolchain) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cargo"
    rustup = shutil.which("rustup")
    if rustup is None:
        raise RuntimeError("rustup is required to pin toolchains")
    path.write_text("#!/bin/sh\ncase \"$1\" in\n  +*) selected=\"${1#+}\"; shift; exec " + shlex.quote(rustup) + " run \"$selected\" cargo \"$@\" ;;\n  *) exec " + shlex.quote(str(toolchain.cargo)) + " \"$@\" ;;\nesac\n")
    path.chmod(0o755)
    return directory


def checked(row: dict) -> None:
    if row["exit_code"]:
        tail = "\n".join(Path(row["stderr"]).read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"benchmark command failed ({row['exit_code']}): {row['command']}\n{tail}")


def run_library(root: Path, package: dict, all_packages: dict, case_name: str, scenario: str,
                profile_name: str, target: Path, env: dict, logs: Path, toolchain: Toolchain,
                probe_file: str | None, repetition: int, storage: Storage, retain_target: bool = False,
                test_probe_file: str | None = None, public_type: str | None = None,
                public_function_file: str | None = None, test_scope: dict | None = None) -> dict:
    reused = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    env = dict(env, CARGO_TARGET_DIR=str(target))
    selected = affected_packages(package["name"], all_packages) if scenario in ("affected-edit", "public-edit") else [package["name"]]
    command = cargo_cmd(scenario, selected, profile_name, target, str(toolchain.cargo))
    edit = None
    prime = None
    if scenario in ("warm-noop", "private-edit", "affected-edit", "public-edit", "test-edit"):
        prime = timed(command, root, env, logs / "prime")
        checked(prime)
    before_fingerprints = fingerprints(target) if scenario == "warm-noop" else None
    if scenario in ("private-edit", "affected-edit", "public-edit", "test-edit"):
        kind = "private-edit" if scenario == "affected-edit" else scenario
        selected_probe = probe_file if kind == "private-edit" else test_probe_file if kind == "test-edit" else public_function_file
        edit = select_edit(root, package, kind, selected_probe, public_type=public_type,
                           public_function=bool(public_function_file) if kind == "public-edit" else False)
        with applied(edit, root):
            row = timed(command, root, env, logs / "measured")
    else:
        row = timed(command, root, env, logs / "measured")
    checked(row)
    artifacts = compiler_artifacts(Path(row["stdout"]))
    owning = [item for item in artifacts if item["package_id"] == package["id"]]
    if not owning:
        raise RuntimeError(f"no compiler artifact proves the selected package was built: {package['name']}")
    if scenario == "warm-noop":
        if any(not item["fresh"] for item in artifacts) or fingerprints(target) != before_fingerprints:
            raise RuntimeError("warm no-op unexpectedly changed a compilation unit or fingerprint")
    if edit is not None and not any(not item["fresh"] for item in owning):
        raise RuntimeError(f"edit probe did not rebuild its owning Cargo unit: {edit.path}")
    if scenario == "test-edit":
        before_execution = fingerprints(target)
        execution = []
        for index, item in enumerate(owning):
            if item["executable"] and item["profile"]["test"]:
                if test_scope is not None:
                    result = execute_matching_tests(item["executable"], test_scope,
                                                    Path(package["manifest_path"]).parent,
                                                    env, logs / f"execution-{index}", compiler_profile=item["profile"])
                else:
                    result = timed([item["executable"], "--test-threads=1"], Path(package["manifest_path"]).parent,
                                   env, logs / f"execution-{index}")
                    checked(result)
                execution.append(result)
        if not execution:
            raise RuntimeError("test-only edit produced no runnable owning test binary")
        if fingerprints(target) != before_execution:
            raise RuntimeError("owning test execution unexpectedly changed compiler fingerprints")
        row["owning_test_execution"] = execution
    row.update(case=case_name, scenario=scenario, profile=profile_name, repetition=repetition,
               selected_packages=selected, compiler_artifacts=artifacts,
               preparation=prime, primed_target_reused=reused,
               rebuilt_unit_count=sum(not item["fresh"] for item in artifacts),
               target_dir_bytes=dir_size_bytes(target),
               cargo_timings=collect_timings(target, logs / "cargo-timings"),
               probe=edit.report(root) if edit else None)
    toolchain.verify()
    if not retain_target:
        storage.remove(target)
    return row


def run_execution(root: Path, target: Path, profile_name: str, env: dict, logs: Path,
                  toolchain: Toolchain, repetition: int) -> dict:
    # Python report adapters need a normal producer executable, not its test harness.
    # Prepare it explicitly; execution must never fall back to a nested Cargo run.
    producer = timed([str(toolchain.cargo), "build", "--offline",
                      "-p", "autoeq-report-wasm", "--bin", "report-producer",
                      "--profile", profile_name, "--target-dir", str(target),
                      "--message-format=json-render-diagnostics"], root, env, logs / "prepare-report-producer")
    checked(producer)
    producers = [item["executable"] for item in compiler_artifacts(Path(producer["stdout"]))
                 if item["executable"] and item["target"]["name"] == "report-producer"]
    if len(producers) != 1:
        raise RuntimeError("report preparation did not identify its executable")
    env = dict(env, AUTOEQ_REPORT_PRODUCER=producers[0], AUTOEQ_WORKSPACE_ROOT=str(root))
    # Reuse metadata is generated by nextest itself; execution never invokes a build.
    build = timed(native_listing(str(toolchain.cargo), target, profile_name), root, env, logs / "prepare")
    checked(build)
    binary_metadata = Path(build["stdout"])
    binaries = json.loads(binary_metadata.read_text())
    ordinary = [entry for entries in binaries["rust-build-meta"]["non-test-binaries"].values() for entry in entries]
    helpers = {entry["name"]: str(target / entry["path"]) for entry in ordinary if entry["kind"] == "bin-exe"}
    for required in ("autoeq", "roomeq"):
        if required not in helpers or not Path(helpers[required]).is_file():
            raise RuntimeError(f"test compilation did not prepare required CLI artifact: {required}")
    metadata_env = dict(env, CARGO_TARGET_DIR=str(target))
    cargo_metadata = subprocess.run([str(toolchain.cargo), "metadata", "--offline", "--all-features", "--format-version", "1"], cwd=root, env=metadata_env, check=True, capture_output=True, text=True).stdout
    cargo_metadata_path = logs / "cargo-metadata.json"
    cargo_metadata_path.write_text(cargo_metadata)
    reuse = ["--binaries-metadata", str(binary_metadata), "--cargo-metadata", str(cargo_metadata_path)]
    listing = subprocess.run([str(toolchain.cargo), "nextest", "list", *reuse, "--message-format", "json"], cwd=root, env=env, check=True, capture_output=True, text=True).stdout
    (logs / "test-identities.json").write_text(listing)
    identities = json.loads(listing)
    configuration = root / ".config/nextest.toml"
    profiles = tomllib.loads(configuration.read_text()).get("profile", {}) if configuration.is_file() else {}
    execution_profile = "full" if "full" in profiles else "default"
    before = fingerprints(target)
    row = timed([str(toolchain.cargo), "nextest", "run", *reuse, "--profile", execution_profile, "--no-tests", "fail"], root, env, logs / "measured")
    checked(row)
    if fingerprints(target) != before:
        raise RuntimeError("execution-only scenario changed compiler fingerprints")
    row.update(case="workspace", scenario="execution", profile=profile_name, repetition=repetition,
               preparation=build, report_producer_preparation=producer,
               report_producer=producers[0], cargo_generated_cli_helpers=helpers,
               test_count=identities["test-count"],
               test_executable_count=len(identities["rust-suites"]),
               test_identity_file=str(logs / "test-identities.json"),
               execution_profile=execution_profile, rebuild_verified_absent=True,
               target_dir_bytes=dir_size_bytes(target), cargo_timings=collect_timings(target, logs / "cargo-timings"))
    return row


def product_command(scenario: str, packages: dict, cargo: str, *, root: Path | None = None,
                    qa_profile: str | None = None, producer_record: Path | None = None) -> list[str]:
    if scenario == "full-native":
        if root is None or qa_profile is None or producer_record is None:
            raise ValueError("full-native requires source, native profile, and producer record")
        return [sys.executable, str(REPO / "scripts/build_bench/native_lane.py"),
                "--root", str(root), "--profile", qa_profile, "--cargo", cargo,
                "--producer-record", str(producer_record), "--", "just", "ntest"]
    if scenario == "product-cli":
        selected = ["autoeq-cli", "roomeq-cli", "autoeq-measurements"] if "autoeq-cli" in packages and "autoeq" not in packages else ["autoeq"]
        command = [cargo, "build", "--offline", "--release", "--bins", "--timings", "--message-format=json-render-diagnostics"]
        for name in selected:
            command += ["-p", name]
        if selected == ["autoeq"]:
            command += ["--features", "cli"]
        return command
    if scenario == "report-native":
        return [cargo, "build", "--offline", "--release", "-p", "autoeq-report-wasm", "--bin", "report-producer", "--timings", "--message-format=json-render-diagnostics"]
    return ["just", "report-dist-gpui" if scenario == "report-gpui" else "report-dist"]


def diagnostic_base_flags(root: Path, env: dict, host: str) -> list[str]:
    """Resolve existing Cargo flags before adding instrumentation overrides."""
    if 'CARGO_ENCODED_RUSTFLAGS' in env:
        return env['CARGO_ENCODED_RUSTFLAGS'].split('\x1f') if env['CARGO_ENCODED_RUSTFLAGS'] else []
    if 'RUSTFLAGS' in env:
        return shlex.split(env['RUSTFLAGS'])
    # This read-only Cargo query uses nightly because config-get is unstable;
    # it does not change the compiler selected for the measured build.
    query = Toolchain.resolve('nightly')
    def config(key):
        result = subprocess.run([str(query.cargo), '-Z', 'unstable-options', 'config', 'get', key, '--format=json'],
                                cwd=root, env=query.environment(), capture_output=True, text=True)
        if result.returncode:
            if 'not set' in result.stderr:
                return {}
            raise RuntimeError('cannot resolve Cargo flags: ' + result.stderr)
        return json.loads(result.stdout)
    targets = config('target').get('target', {})
    if any(key.startswith('cfg(') and value.get('rustflags') for key, value in targets.items()):
        raise RuntimeError('conditional target Rust flags require an explicit diagnostic flag resolution')
    flags = targets.get(host, {}).get('rustflags')
    if flags is None:
        flags = config('build.rustflags').get('build', {}).get('rustflags', [])
    return shlex.split(flags) if isinstance(flags, str) else list(flags)



def native_linker_environment(root: Path, env: dict, host: str, identity: dict) -> dict:
    """Select the Mach-O experiment without dropping deployment or framework flags."""
    selected = dict(env)
    if identity['kind'] == 'system':
        return selected
    if not host.endswith('-apple-darwin') or env.get('CARGO_BUILD_TARGET', host) != host:
        raise ValueError('Mach-O LLD selection is limited to native host builds')
    flags = diagnostic_base_flags(root, env, host)
    if any('fuse-ld' in flag or flag.startswith(('linker=', '-Clinker=', '-C=linker=', '--codegen=linker=')) for flag in flags):
        raise ValueError('explicit Rust linker selection conflicts with the Mach-O experiment')
    flags += ['-C', 'linker=' + identity['driver_path'],
              '-C', 'link-arg=-fuse-ld=' + identity.get('linker_invocation_path', identity['linker_path'])]
    selected.pop('RUSTFLAGS', None)
    selected['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
    selected['AUTOEQ_BENCH_NATIVE_LINKER_KIND'] = 'lld-macho'
    selected.pop('AUTOEQ_BENCH_APPLE_LINK_STATS', None)
    return selected


def run_diagnostics(root: Path, packages: dict, target: Path, env: dict,
                    logs: Path, toolchain: Toolchain, llvm: bool, scope: str,
                    selected: list[str], profile_name: str, detailed_traces: bool = False) -> dict:
    """Separate instrumented build; never included in timing summaries."""
    target.mkdir(parents=True)
    logs.mkdir(parents=True)
    driver = shutil.which("cc")
    if driver is None:
        raise RuntimeError("native diagnostic requires a C linker driver")
    shim = target.parent / "linker-shim"
    linker_source = shim.parent / "linker_driver.py"
    shutil.copy2(Path(__file__).parent / "build_bench/linker.py", linker_source)
    shim.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(linker_source)) + ' "$@"\n')
    shim.chmod(0o755)
    instrumented = dict(env, AUTOEQ_BENCH_REAL_LINKER=driver,
                        AUTOEQ_BENCH_LINK_LOG=str(logs / "link-events"))
    apple_statistics = sys.platform == "darwin" and env.get("AUTOEQ_BENCH_NATIVE_LINKER_KIND", "system") == "system"
    instrumented.pop("AUTOEQ_BENCH_APPLE_LINK_STATS", None)
    if apple_statistics:
        instrumented["AUTOEQ_BENCH_APPLE_LINK_STATS"] = "1"
    flags = diagnostic_base_flags(root, instrumented, toolchain.host)
    original_flags = list(flags)
    instrumented.pop("CARGO_ENCODED_RUSTFLAGS", None)
    instrumented.pop("RUSTFLAGS", None)
    flags += ["-C", "linker=" + str(shim)]
    if llvm:
        flags += ["-Ztime-passes", "-Ztime-passes-format=json"]
        if detailed_traces:
            flags += ["-Zllvm-time-trace"]
        compiler_source = shim.parent / "compiler_driver.py"
        shutil.copy2(Path(__file__).parent / "build_bench/compiler.py", compiler_source)
        compiler_shim = shim.parent / "compiler-shim"
        compiler_shim.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(compiler_source)) + ' "$@"\n')
        compiler_shim.chmod(0o755)
        instrumented.pop("RUSTC_WORKSPACE_WRAPPER", None)
        instrumented.update(RUSTC_WRAPPER=str(compiler_shim),
                            AUTOEQ_BENCH_COMPILE_LOG=str(logs / "compiler-events"),
                            CARGO_BUILD_JOBS="1")
    instrumented["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    instrumented["CARGO_TARGET_DIR"] = str(target)
    if scope == "product-cli":
        command = product_command("product-cli", packages, str(toolchain.cargo))
        profile_name = "release"
    else:
        command = cargo_cmd("affected-edit", selected, profile_name, target, str(toolchain.cargo))
        if set(selected) == set(packages):
            command.insert(command.index("--tests") + 1, "--benches")
    row = timed(command, root, instrumented, logs / "measured")
    checked(row)
    events = sorted((logs / "link-events").glob("*.json"))
    events = [path for path in events if not path.name.endswith(".argv.json")]
    if not events:
        raise RuntimeError("linker diagnostic produced no instrumented link events")
    compiler_events = sorted((logs / "compiler-events").glob("*.json")) if llvm else []
    phase_count = sum(len(json.loads(path.read_text())["phases"]) for path in compiler_events)
    if llvm and not phase_count:
        raise RuntimeError("compiler instrumentation emitted no phase timings; diagnostics are incomplete")
    traces = []
    if detailed_traces:
        trace_directory = logs / "llvm-traces"
        trace_directory.mkdir()
        for index, trace in enumerate(sorted(target.rglob("*.llvm_timings.json"))):
            compressed = trace_directory / f"{index}-{trace.name}.gz"
            with trace.open("rb") as source, gzip.open(compressed, "wb") as destination:
                shutil.copyfileobj(source, destination)
            traces.append(str(compressed))
        if not traces:
            raise RuntimeError("LLVM instrumentation emitted no traces; diagnostics are incomplete")
    row.update(case="workspace", scenario="linker-llvm-diagnostic" if llvm else "linker-diagnostic",
               profile=profile_name, diagnostic=True, scope=scope, includes_workspace_benches=(scope == "test-build" and set(selected) == set(packages)), diagnostic_jobs=1 if llvm else env.get("CARGO_BUILD_JOBS"), preserved_rustflags=original_flags, compiler_events=[str(path) for path in compiler_events], compiler_phase_count=phase_count, tools=toolchain.report(), llvm_traces=traces, link_events=[str(path) for path in events],
               llvm_pass_timing_stderr=row["stderr"] if llvm else None,
               native_linker_kind=env.get("AUTOEQ_BENCH_NATIVE_LINKER_KIND", "system"),
               measurement_scope=("C driver wall time and Apple native linker timer table" if apple_statistics else "C driver wall time without Apple-only native timers") + "; LLVM pass diagnostics are separate from normal samples",
               target_dir_bytes=dir_size_bytes(target), cargo_timings=collect_timings(target, logs / "cargo-timings"))
    toolchain.verify()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crate")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--case", action="append", help="restrict logical library cases for diagnostic preflights; requires --full")
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--source-ref", "--baseline-ref", dest="source_ref")
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--scenario", action="append", choices=[*LIBRARY_SCENARIOS, *WORKSPACE_SCENARIOS, "edit"])
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--test-profile", default="test")
    parser.add_argument("--qa-profile")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--toolchain")
    parser.add_argument("--probe-file")
    parser.add_argument("--wasm-bindgen", type=Path)
    parser.add_argument("--native-linker", choices=("system", "lld-macho"), default="system", help="explicit native Mach-O linker experiment; excludes WASM scenarios")
    parser.add_argument("--instrument-linker", action="store_true")
    parser.add_argument("--llvm-diagnostics", action="store_true")
    parser.add_argument("--llvm-traces", action="store_true", help="also retain large per-function LLVM traces")
    parser.add_argument("--diagnostic-scope", choices=("test-build", "product-cli"), default="test-build")
    parser.add_argument("--scratch-dir", "--target-dir", dest="scratch", type=Path, default=Path("/Volumes/home_tmp/tmp") if Path("/Volumes/home_tmp/tmp").is_dir() else Path(tempfile.gettempdir()))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    if not args.full and not args.crate:
        parser.error("--crate or --full is required")
    if args.case and not args.full:
        parser.error("--case requires --full")
    if args.repetitions < 1 or args.jobs < 1:
        parser.error("repetitions and jobs must be positive")
    if not hasattr(os, "wait4"):
        parser.error("this memory/timing runner requires a POSIX host")
    if args.native_linker == "lld-macho":
        if sys.platform != "darwin":
            parser.error("Mach-O LLD experiments require a macOS host")
        if (args.full and not args.scenario) or set(args.scenario or ()) & {"report-wasm", "report-gpui"}:
            parser.error("Mach-O LLD requires explicit native-only scenarios")
    native_linker = native_linker_identity(args.native_linker)
    source = args.source_root.resolve()
    toolchain = Toolchain.resolve(args.toolchain)
    source_roots = [source, *dependency_roots(source, toolchain)]
    if any(args.scratch.resolve().is_relative_to(root.resolve()) for root in source_roots):
        parser.error("benchmark scratch must be outside every source/dependency repository")
    args.scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.scratch).free < 20 * 1024 ** 3:
        parser.error("isolated build measurements require at least 20 GiB of free scratch space")
    storage = Storage(args.scratch, source_roots)
    stamp = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = args.output_dir.resolve() / f"bench-{args.label}-{stamp}"
    report = {"schema": "refactor-20261025/bench/v2", "label": args.label,
              "status": "preparing", "harness_sha256": {str(path.relative_to(REPO)): digest(path) for path in [Path(__file__), *sorted((Path(__file__).parent / "build_bench").glob("*.py")), Path(__file__).parent / "build_bench/test_scopes.json"]}, "tools": {**toolchain.report(),
              "cargo_version": capture([str(toolchain.cargo), "--version"]),
              "nextest_version": capture([str(toolchain.cargo), "nextest", "--version"]),
              "compiler_driver_version": capture(["cc", "--version"]),
              "native_linker": native_linker},
              "environment": build_environment_identity(dict(os.environ, CARGO_BUILD_JOBS=str(args.jobs))),
              "host": {**host_hardware(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "jobs": args.jobs},
              "results": [], "normal_timing_excludes_diagnostics": True,
              "acceptance_repetitions_satisfied": False}
    try:
        root, sources = snapshot(source, storage.root / "source", toolchain, args.source_ref)
        output.mkdir(parents=True)
        report["sources"] = sources
        source_evidence = output / "sources"
        source_evidence.mkdir()
        for name, record in sources.items():
            copied = Path(record["snapshot"])
            patch = source_evidence / f"{name}.patch"
            patch.write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=copied))
            archive = source_evidence / f"{name}-untracked.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for relative in record["snapshot_identity"]["untracked_source_sha256"]:
                    bundle.add(copied / relative, arcname=relative, recursive=False)
            record["tracked_patch_file"] = str(patch)
            record["untracked_source_archive"] = str(archive)
        data = metadata(root, toolchain)
        member_ids = set(data["workspace_members"])
        packages = {item["name"]: item for item in data["packages"] if item["id"] in member_ids}
        selected_cases = cases(packages, args.crate, args.full)
        if args.case:
            requested = set(args.case)
            unknown = requested - {case.name for case in selected_cases}
            if unknown or len(requested) != len(args.case):
                raise ValueError(f"unknown or duplicate logical cases: {args.case}")
            selected_cases = [case for case in selected_cases if case.name in requested]
        report["selected_library_cases"] = [case.name for case in selected_cases]
        logical_test_scopes = json.loads((Path(__file__).parent / "build_bench/test_scopes.json").read_text())["cases"] if args.full else {}
        manifest = tomllib.loads((root / "Cargo.toml").read_text())
        qa_profile = args.qa_profile or ("qa" if "qa" in manifest.get("profile", {}) else "release")
        scenarios = ["private-edit" if name == "edit" else name for name in (args.scenario or (list(LIBRARY_SCENARIOS) + list(WORKSPACE_SCENARIOS) if args.full else ["clean-check", "warm-noop", "private-edit", "test-build"]))]
        env = toolchain.environment()
        env["CARGO_BUILD_JOBS"] = str(args.jobs)
        env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "scripts"), env.get("PYTHONPATH", "")])
        env["AUTOEQ_WORKSPACE_ROOT"] = str(root)
        env.pop("AUTOEQ_REPORT_PRODUCER", None)
        env = native_linker_environment(root, env, toolchain.host, native_linker)
        env["PATH"] = str(cargo_shim(storage.root / "tools/stable", toolchain)) + os.pathsep + env.get("PATH", "")
        python = source / "venv/bin/python"
        if python.is_file():
            env.setdefault("ROOMEQ_REPORT_PYTHON", str(python))
            env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
        bindgen = args.wasm_bindgen or (Path(env["WANDBIN"]) if env.get("WANDBIN") else None)
        if any(name in scenarios for name in ("report-wasm", "report-gpui")):
            bindgen = bindgen or (Path(shutil.which("wasm-bindgen")) if shutil.which("wasm-bindgen") else None)
            if bindgen is None:
                raise RuntimeError("report build scenarios require --wasm-bindgen matching Cargo.lock")
            locked = tomllib.loads((root / "Cargo.lock").read_text())
            expected = next(item["version"] for item in locked["package"] if item["name"] == "wasm-bindgen")
            if expected not in capture([str(bindgen), "--version"]):
                raise RuntimeError(f"wasm-bindgen CLI must match locked version {expected}")
            env["WANDBIN"] = str(bindgen.resolve())
        report["effective_build_environment"] = build_environment_identity(env)
        for repetition in range(args.repetitions):
            for case in selected_cases:
                package = packages[case.package]
                for scenario in scenarios:
                    if scenario not in LIBRARY_SCENARIOS:
                        continue
                    if scenario == "public-edit" and not (case.public_type or case.public_function_file or package["name"] in PUBLIC_TYPES):
                        continue
                    profile_name = args.profile if scenario in ("clean-check", "clean-build", "warm-noop", "private-edit") else args.test_profile
                    retain_target = scenario in ("warm-noop", "private-edit", "affected-edit", "public-edit", "test-edit")
                    if retain_target:
                        scope = "affected" if scenario in ("affected-edit", "public-edit") else case.name + ("/test" if scenario == "test-edit" else "/check")
                        target = storage.root / "targets/primed" / str(repetition) / profile_name / scope
                    else:
                        target = storage.root / "targets" / case.name / scenario / str(repetition)
                    logs = output / case.name / scenario / str(repetition)
                    print(f"{args.label}: {case.name}/{scenario}, repetition {repetition + 1}/{args.repetitions}", flush=True)
                    row = run_library(root, package, packages, case.name, scenario, profile_name, target, dict(env), logs, toolchain, args.probe_file or case.probe_file, repetition, storage, retain_target,
                                      test_probe_file=args.probe_file or case.test_probe_file,
                                      public_type=case.public_type, public_function_file=case.public_function_file,
                                      test_scope=logical_test_scopes.get(case.name))
                    report["results"].append(row)
                    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            for scenario in scenarios:
                if scenario not in WORKSPACE_SCENARIOS:
                    continue
                target = storage.root / "targets/workspace" / scenario / str(repetition)
                target.mkdir(parents=True)
                logs = output / "workspace" / scenario / str(repetition)
                scenario_env = dict(env, CARGO_TARGET_DIR=str(target))
                selected_toolchain = toolchain
                if scenario == "report-gpui":
                    selected_toolchain = Toolchain.resolve("nightly")
                    report["nightly_toolchain"] = selected_toolchain.report()
                    scenario_env.update(selected_toolchain.environment())
                    scenario_env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS=str(args.jobs), WANDBIN=str(bindgen.resolve()))
                    scenario_env["PATH"] = str(cargo_shim(storage.root / "tools/nightly", selected_toolchain)) + os.pathsep + env["PATH"]
                print(f"{args.label}: workspace/{scenario}, repetition {repetition + 1}/{args.repetitions}", flush=True)
                if scenario == "execution":
                    row = run_execution(root, target, qa_profile, scenario_env, logs, toolchain, repetition)
                else:
                    producer_record = logs / "report-producer.json"
                    command = product_command(scenario, packages, str(selected_toolchain.cargo),
                                              root=root, qa_profile=qa_profile,
                                              producer_record=producer_record)
                    row = timed(command, root, scenario_env, logs / "measured")
                    checked(row)
                    row.update(case="workspace", scenario=scenario, profile=qa_profile if scenario == "full-native" else "release", repetition=repetition, target_dir_bytes=dir_size_bytes(target), cargo_timings=collect_timings(target, logs / "cargo-timings"))
                    if scenario == "full-native":
                        row.update(report_producer_preparation=json.loads(producer_record.read_text()),
                                   producer_build_included_in_measurement=True)
                report["results"].append(row)
                selected_toolchain.verify()
                storage.remove(target)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            primed = storage.root / "targets/primed" / str(repetition)
            if primed.exists():
                storage.remove(primed)
        # Instrumentation is an additional diagnostic, never a normal timing sample.
        if args.instrument_linker or args.llvm_diagnostics or args.llvm_traces or (args.full and not args.scenario):
            target = storage.root / "targets/diagnostics/release"
            use_llvm = args.llvm_diagnostics or args.llvm_traces or args.full
            diagnostic_toolchain = Toolchain.resolve("nightly") if use_llvm else toolchain
            diagnostic_env = dict(env)
            diagnostic_env.update(diagnostic_toolchain.environment())
            diagnostic_env["CARGO_BUILD_JOBS"] = str(args.jobs)
            row = run_diagnostics(root, packages, target, diagnostic_env, output / "diagnostics", diagnostic_toolchain, use_llvm,
                                  args.diagnostic_scope, sorted(packages) if args.full else [args.crate], args.test_profile, args.llvm_traces)
            report["results"].append(row)
            storage.remove(target)
        if native_linker_identity(args.native_linker) != native_linker:
            raise RuntimeError("native compiler driver or linker changed during measurement")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        report["summary"] = summarize(report["results"])
        report["acceptance_repetitions_satisfied"] = (
            report["status"] == "passed" and bool(report["summary"])
            and all(group["samples"] >= 3 for group in report["summary"]))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not args.keep_artifacts:
            storage.close()
        print(f"wrote {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
