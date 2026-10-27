"""Safety and measurement-contract tests for the isolated build harness."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_build import (build_environment_identity, cargo_cmd, diagnostic_base_flags,
                         native_linker_environment, native_linker_identity, run_library, summarize)
from build_bench.probes import applied, literal_edit, select_edit
from build_bench.snapshots import Storage, copy_repository
from build_bench.execution import timed
from build_bench.plans import native_listing


class IsolationTests(unittest.TestCase):
    def test_dirty_snapshot_is_independent_and_preserves_deletions(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "code.rs").write_text("fn value() -> usize { 2usize.max(1) }\n")
            (source / "deleted.rs").write_text("old")
            subprocess.run(["git", "add", "."], cwd=source, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "fixture"], cwd=source, check=True)
            (source / "deleted.rs").unlink()
            (source / "input.json").write_text('{"new": true}')
            original = (source / "code.rs").read_bytes()
            copy = base / "copy"
            copy_repository(source, copy)
            self.assertFalse((copy / "deleted.rs").exists())
            self.assertEqual((copy / "input.json").read_bytes(), (source / "input.json").read_bytes())
            edit = literal_edit(copy / "code.rs", "private-edit")
            self.assertIsNotNone(edit)
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                with applied(edit, copy):
                    self.assertNotEqual((copy / "code.rs").read_bytes(), original)
                    self.assertEqual((source / "code.rs").read_bytes(), original)
                    raise RuntimeError("build failed")
            self.assertEqual((copy / "code.rs").read_bytes(), original)

    def test_snapshot_preserves_prefixed_runtime_and_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source = base / 'source'; source.mkdir()
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            files = {'data_tests/room/target-low.csv': b'frequency,spl\n20,1\n',
                     'data_tests/target-bank/curve.csv': b'target bank',
                     'data_tests/bench-bank/impulse.csv': b'benchmark fixture',
                     'src/target-state.rs': b'pub const VALUE: usize = 1;\n',
                     'benches/bench-response.rs': b'fn main() {}\n',
                     'src/target-basis/mod.rs': b'pub const GAIN: f64 = 0.0;\n',
                     'benches/bench-resources/curve.csv': b'benchmark source resource',
                     'docs/baselines/refactor-fixture/bench-notes.json': b'{"source": true}'}
            for name, data in files.items():
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            subprocess.run(['git', 'add', '.'], cwd=source, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-qm', 'fixture'], cwd=source, check=True)
            artifact = source / 'target-local'; artifact.mkdir(); (artifact / '.rustc_info.json').write_text('{}'); (artifact / 'noise.o').write_bytes(b'build artifact')
            output = source / 'docs/baselines/refactor-fixture/bench-smoke'; output.mkdir(); (output / 'report.json').write_text('{}')
            for index, reference in enumerate((None, 'HEAD')):
                destination = base / f'copy-{index}'
                record = copy_repository(source, destination, reference)
                for name, data in files.items():
                    self.assertEqual((destination / name).read_bytes(), data)
                self.assertEqual(len(record['snapshot_identity']['runtime_inputs_sha256']), 3)
                self.assertEqual(record['origin_identity']['runtime_inputs_sha256'], record['snapshot_identity']['runtime_inputs_sha256'])
                self.assertFalse((destination / 'target-local').exists())
                self.assertFalse((destination / 'docs/baselines/refactor-fixture/bench-smoke').exists())

    def test_reference_snapshot_preserves_ignored_assets_without_private_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source = base / 'source'; source.mkdir()
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            (source / '.gitignore').write_text('.env\nignored.csv\n')
            (source / 'data_tests').mkdir(); (source / 'data_tests/curve.csv').write_text('fixture')
            subprocess.run(['git', 'add', '.'], cwd=source, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-qm', 'fixture'], cwd=source, check=True)
            (source / 'data_tests/.env').write_text('private fixture sentinel')
            (source / 'data_tests/ignored.csv').write_text('ignored runtime input')
            destination = base / 'copy'; copy_repository(source, destination, 'HEAD')
            self.assertEqual((destination / 'data_tests/ignored.csv').read_text(), 'ignored runtime input')
            self.assertFalse((destination / 'data_tests/.env').exists())

    def test_cleanup_refuses_unowned_paths_and_source_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = parent / "source"
            source.mkdir()
            sentinel = source / "keep"
            sentinel.write_text("unchanged")
            with self.assertRaises(ValueError):
                Storage(source / "scratch", [source])
            storage = Storage(parent, [source])
            try:
                for path in (source, storage.root):
                    with self.assertRaises(ValueError):
                        storage.remove(path)
                link = storage.root / "escape"
                link.symlink_to(source, target_is_directory=True)
                with self.assertRaises(ValueError):
                    storage.remove(link)
            finally:
                storage.close()
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_probe_rejects_comment_and_outside_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "code.rs"
            path.write_text("// x.max(1)\nfn f() { let _ = 2usize.max(1); }\n")
            edit = literal_edit(path, "private-edit")
            self.assertEqual(edit.line, 2)
            self.assertIn(b"let autoeq_build_bench_value", edit.changed)
            with self.assertRaises(ValueError):
                with applied(edit, root / "elsewhere"):
                    self.fail("outside edit was allowed")

    def test_test_import_does_not_hide_production_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lib.rs"
            path.write_text("#[cfg(test)] use example::Fixture;\nfn f() { let _ = 2usize.max(1); }\n#[cfg(test)] mod tests { fn f() { let _ = 2usize.max(2); } }\n")
            edit = literal_edit(path, "private-edit")
            self.assertEqual(edit.line, 2)
            self.assertIsNone(literal_edit(path, "test-edit"))

    def test_test_probe_requires_test_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lib.rs"
            path.write_text("fn f() { assert_eq!(2, 2); }\n")
            self.assertIsNone(literal_edit(path, "test-edit"))
            path.write_text("#[cfg(test)] mod tests { #[test] fn f() { assert_eq!(2, 2); } }\n")
            self.assertIsNotNone(literal_edit(path, "test-edit"))


class MeasurementTests(unittest.TestCase):
    def test_failed_requested_three_repetitions_cannot_claim_acceptance(self):
        import bench_build
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'source'
            source.mkdir()
            output = base / 'reports'
            toolchain = Mock(cargo=Path('/fixture-cargo'))
            toolchain.report.return_value = {}
            arguments = ['bench_build.py', '--crate', 'autoeq-core',
                         '--source-root', str(source), '--scratch-dir', str(base / 'scratch'),
                         '--output-dir', str(output), '--repetitions', '3']
            with patch.object(sys, 'argv', arguments), \
                 patch('bench_build.Toolchain.resolve', return_value=toolchain), \
                 patch('bench_build.native_linker_identity', return_value={'kind': 'system'}), \
                 patch('bench_build.dependency_roots', return_value=[]), \
                 patch('bench_build.capture', return_value='fixture tool version'), \
                 patch('bench_build.host_hardware', return_value={}), \
                 patch('bench_build.shutil.disk_usage', return_value=Mock(free=100 * 1024 ** 3)), \
                 patch('bench_build.snapshot', side_effect=RuntimeError('fixture snapshot failed')):
                with self.assertRaisesRegex(RuntimeError, 'fixture snapshot failed'):
                    bench_build.main()
            reports = list(output.glob('bench-*/report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['results'], [])
            self.assertFalse(report['acceptance_repetitions_satisfied'])

    def test_profile_and_target_overrides_change_comparison_identity(self):
        baseline = {"CARGO_TARGET_DIR": "/baseline/target", "CARGO_PROFILE_TEST_OPT_LEVEL": "1",
                    "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER": "/usr/bin/cc"}
        candidate = dict(baseline, CARGO_TARGET_DIR="/candidate/target")
        self.assertEqual(build_environment_identity(baseline), build_environment_identity(candidate))
        candidate["CARGO_PROFILE_TEST_OPT_LEVEL"] = "2"
        self.assertNotEqual(build_environment_identity(baseline), build_environment_identity(candidate))
        candidate = dict(baseline, CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="/custom/cc")
        self.assertNotEqual(build_environment_identity(baseline), build_environment_identity(candidate))

    def test_scope_and_profile_are_explicit(self):
        command = cargo_cmd("affected-edit", ["autoeq-core", "autoeq-cli"], "qa", Path("/tmp/target"))
        self.assertIn("--all-features", command)
        self.assertIn("--locked", command)
        self.assertIn("--offline", command)
        self.assertEqual(command[command.index("--profile") + 1], "qa")
        self.assertEqual(command.count("-p"), 2)
        self.assertIn("--all-features", cargo_cmd("test-edit", "autoeq-optim", "test", Path("/tmp/target")))
        with self.assertRaises(ValueError):
            cargo_cmd("unknown", "core", "dev", Path("/tmp/target"))

    def test_execution_scope_matches_the_library_and_integration_inventory(self):
        command = native_listing("cargo", Path("/tmp/owned-target"), "qa")
        self.assertIn("--lib", command)
        self.assertIn("--tests", command)
        self.assertNotIn("--all-targets", command)
        self.assertIn("--benches", command)
        self.assertEqual(command[command.index("--cargo-profile") + 1], "qa")

    def test_explicit_rustflags_precedence_is_preserved(self):
        flags = diagnostic_base_flags(Path('/source'), {'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-cpu=native',
                                                       'RUSTFLAGS': '-C opt-level=0'}, 'host')
        self.assertEqual(flags, ['-C', 'target-cpu=native'])
        self.assertEqual(diagnostic_base_flags(Path('/source'), {'RUSTFLAGS': '-C "link-arg=-L space path"'}, 'host'),
                         ['-C', 'link-arg=-L space path'])

    def test_cargo_target_flags_survive_instrumentation(self):
        flags = ['-C', 'link-arg=-mmacosx-version-min=15.0', '-C', 'link-arg=-framework', '-C', 'link-arg=Accelerate']
        query = Mock(cargo=Path('/nightly/cargo'))
        query.environment.return_value = {}
        response = Mock(returncode=0, stdout=json.dumps({'target': {'host': {'rustflags': flags}}}))
        with patch('bench_build.Toolchain.resolve', return_value=query), patch('bench_build.subprocess.run', return_value=response):
            self.assertEqual(diagnostic_base_flags(Path('/source'), {}, 'host'), flags)

    def test_unresolved_conditional_flags_fail_closed(self):
        query = Mock(cargo=Path('/nightly/cargo'))
        query.environment.return_value = {}
        response = Mock(returncode=0, stdout=json.dumps({'target': {'cfg(unix)': {'rustflags': ['-C', 'debuginfo=1']}}}))
        with patch('bench_build.Toolchain.resolve', return_value=query), patch('bench_build.subprocess.run', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'conditional target'):
                diagnostic_base_flags(Path('/source'), {}, 'host')

    def test_diagnostics_are_excluded_from_summary(self):
        normal = dict(case="core", scenario="clean-build", profile="dev", wall_seconds=1)
        diagnostic = dict(normal, wall_seconds=100, diagnostic=True)
        self.assertEqual(summarize([normal, diagnostic])[0]["median_wall_seconds"], 1)

    def test_focused_test_edit_executes_owning_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / 'owning-tests'
            executable.write_text('#!' + sys.executable + '\nimport sys\nassert "--test-threads=1" in sys.argv\nprint("owning test executed")\n')
            executable.chmod(0o755)
            package = {'name': 'fixture', 'id': 'fixture-id', 'manifest_path': str(root / 'Cargo.toml')}
            artifact = {'package_id': package['id'], 'executable': str(executable),
                        'profile': {'test': True, 'debug_assertions': True}, 'fresh': False}
            toolchain = Mock(cargo=Path('/fixture-cargo'))
            edit = Mock(); edit.report.return_value = {'kind': 'test-edit'}
            def measure(command, cwd, env, prefix):
                if command[0] == '/fixture-cargo':
                    return {'exit_code': 0, 'stdout': str(root / 'fake-build-output')}
                return timed(command, cwd, env, prefix)
            with patch('bench_build.timed', side_effect=measure), \
                 patch('bench_build.compiler_artifacts', return_value=[artifact]), \
                 patch('bench_build.select_edit', return_value=edit), \
                 patch('bench_build.applied', return_value=contextlib.nullcontext()):
                row = run_library(root, package, {'fixture': package}, 'fixture', 'test-edit',
                                  'test', root / 'target', {'PATH': os.defpath}, root / 'logs',
                                  toolchain, None, 0, Mock(), retain_target=True)
            executions = row['owning_test_execution']
            self.assertEqual(len(executions), 1)
            self.assertEqual(executions[0]['exit_code'], 0)
            self.assertEqual(Path(executions[0]['stdout']).read_text().strip(), 'owning test executed')
            toolchain.verify.assert_called_once()

    def test_failed_child_preserves_output_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = timed([sys.executable, "-c", "import sys; print('evidence'); sys.exit(7)"], root, dict(os.environ), root / "run")
            self.assertEqual(row["exit_code"], 7)
            self.assertEqual(Path(row["stdout"]).read_text().strip(), "evidence")
            self.assertGreater(row["wall_seconds"], 0)
            self.assertGreater(row["max_process_rss_bytes"], 0)



class NativeLinkerTests(unittest.TestCase):
    def test_system_selection_preserves_environment(self):
        original = {'RUSTFLAGS': '-C opt-level=1'}
        selected = native_linker_environment(Path('/source'), original, 'host', {'kind': 'system'})
        self.assertEqual(selected, original)
        self.assertIsNot(selected, original)

    def test_lld_preserves_flags_and_pins_both_tools(self):
        flags = ['-C', 'link-arg=-mmacosx-version-min=15.0', '-C', 'link-arg=-framework', '-C', 'link-arg=Accelerate']
        original = {'CARGO_ENCODED_RUSTFLAGS': '\x1f'.join(flags), 'RUSTFLAGS': '-C opt-level=0',
                    'AUTOEQ_BENCH_APPLE_LINK_STATS': '1'}
        identity = {'kind': 'lld-macho', 'driver_path': '/usr/bin/clang', 'linker_path': '/tools with spaces/ld64.lld'}
        selected = native_linker_environment(Path('/source'), original, 'aarch64-apple-darwin', identity)
        self.assertEqual(selected['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'), flags +
                         ['-C', 'linker=/usr/bin/clang', '-C', 'link-arg=-fuse-ld=/tools with spaces/ld64.lld'])
        self.assertNotIn('RUSTFLAGS', selected)
        self.assertNotIn('AUTOEQ_BENCH_APPLE_LINK_STATS', selected)
        self.assertIn('RUSTFLAGS', original)

    def test_lld_rejects_cross_targets_and_existing_linker_flags(self):
        identity = {'kind': 'lld-macho', 'driver_path': '/cc', 'linker_path': '/ld64.lld'}
        for host, environment in [('x86_64-unknown-linux-gnu', {}),
                                  ('aarch64-apple-darwin', {'CARGO_BUILD_TARGET': 'wasm32-unknown-unknown'})]:
            with self.assertRaisesRegex(ValueError, 'native host'):
                native_linker_environment(Path('/source'), environment, host, identity)
        for flags in ('-C linker=/custom', '-Clinker=/custom', '--codegen=linker=/custom', '-C link-arg=-fuse-ld=other'):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                native_linker_environment(Path('/source'), {'RUSTFLAGS': flags}, 'aarch64-apple-darwin', identity)

    def test_lld_cli_rejects_wasm_before_creating_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / 'scratch'
            output = Path(temporary) / 'results'
            for selection in ([], ['--scenario', 'report-wasm'], ['--scenario', 'report-gpui']):
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('bench_build.py')),
                                         '--full', '--native-linker', 'lld-macho', '--scratch-dir', str(scratch),
                                         '--output-dir', str(output), *selection], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('Mach-O LLD', result.stderr)
                self.assertFalse(scratch.exists())
                self.assertFalse(output.exists())

    def test_lld_identity_rejects_other_platforms(self):
        with patch('bench_build.sys.platform', 'linux'):
            with self.assertRaisesRegex(ValueError, 'macOS host'):
                native_linker_identity('lld-macho')

    def test_lld_identity_records_selected_tool_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver = Path(temporary) / 'clang'; driver.write_bytes(b'compiler')
            binary = Path(temporary) / 'lld'; binary.write_bytes(b'linker')
            linker = Path(temporary) / 'ld64.lld'; linker.symlink_to(binary)
            response = Mock(stdout='LLD fixture version', stderr='')
            with patch('bench_build.sys.platform', 'darwin'), \
                 patch('bench_build.shutil.which', side_effect=lambda name: str(driver if name == 'cc' else linker)), \
                 patch('bench_build.subprocess.run', return_value=response) as run:
                identity = native_linker_identity('lld-macho')
            self.assertEqual(identity['linker_path'], str(linker.resolve()))
            self.assertEqual(identity['driver_path'], str(driver.resolve()))
            self.assertEqual(identity['linker_invocation_path'], str(linker.absolute()))
            selected = native_linker_environment(Path('/source'), {'CARGO_ENCODED_RUSTFLAGS': ''}, 'aarch64-apple-darwin', identity)
            self.assertIn('link-arg=-fuse-ld=' + str(linker.absolute()), selected['CARGO_ENCODED_RUSTFLAGS'].split('\x1f'))
            self.assertEqual(len(identity['sha256']), 64)
            self.assertNotEqual(identity['driver_sha256'], identity['sha256'])
            self.assertEqual(run.call_args.args[0], [str(linker), '--version'])


if __name__ == "__main__":
    unittest.main()
