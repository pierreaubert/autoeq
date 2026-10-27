"""The full-native measurement prepares its canonical helper inside the timed lane."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_build import product_command


class NativeLanePrerequisiteTests(unittest.TestCase):
    def fixture(self, parent):
        source = parent / 'original source'
        source.mkdir()
        (source / 'Cargo.toml').write_text('[workspace]\n')
        tools = parent / 'fake tools'
        tools.mkdir()
        cargo = tools / 'cargo'
        cargo.write_text('#!' + sys.executable + '\n' + r'''import json, os, pathlib, sys
args = sys.argv[1:]
assert pathlib.Path.cwd() == pathlib.Path(os.environ['EXPECTED_SOURCE'])
assert args[:1] == ['build']
assert '--offline' in args and '--locked' in args
profile = args[args.index('--profile') + 1]
target = pathlib.Path(args[args.index('--target-dir') + 1])
if os.environ.get('FAIL_BUILD'):
    sys.exit(19)
producer = target / profile / 'report-producer'
producer.parent.mkdir(parents=True, exist_ok=True)
producer.write_text('#!' + sys.executable + '\nprint("prepared")\n')
producer.chmod(0o755)
event = {'reason': 'compiler-artifact', 'package_id': 'autoeq-report-wasm 0.1.0',
         'target': {'name': 'report-producer', 'kind': ['bin']},
         'profile': {'test': bool(os.environ.get('BAD_TEST_ARTIFACT')), 'opt_level': '2', 'debug_assertions': False},
         'features': [], 'filenames': [str(producer)], 'executable': str(producer), 'fresh': False}
print(json.dumps(event))
''')
        cargo.chmod(0o755)
        recipe = tools / 'just'
        recipe.write_text('#!' + sys.executable + '\n' + r'''import json, os, pathlib, subprocess, sys
if not os.environ.get('AUTOEQ_REPORT_PRODUCER'):
    sys.stderr.write('missing prepared report producer\n')
    sys.exit(77)
assert pathlib.Path.cwd() == pathlib.Path(os.environ['EXPECTED_SOURCE'])
assert os.environ['AUTOEQ_WORKSPACE_ROOT'] == os.environ['EXPECTED_SOURCE']
record = json.loads(pathlib.Path(os.environ['EXPECTED_RECORD']).read_text())
assert record['requested_profile'] == os.environ['EXPECTED_PROFILE']
producer = pathlib.Path(os.environ['AUTOEQ_REPORT_PRODUCER'])
assert producer == pathlib.Path(record['executable']) and producer.is_file()
if os.environ.get('REBUILD_CARGO_PRODUCER'):
    cargo_producer = pathlib.Path(record['artifact']['executable'])
    cargo_producer.write_text('#!' + sys.executable + '\nprint("rebuilt by native recipe")\n')
    assert producer != cargo_producer
assert subprocess.check_output([str(producer)], text=True).strip() == 'prepared'
if os.environ.get('MUTATE_BOUND_PRODUCER'):
    producer.write_text('changed pinned producer after native execution\n')
pathlib.Path(os.environ['RECIPE_RAN']).write_text('yes')
''')
        recipe.chmod(0o755)
        record = parent / 'logs' / 'report-producer.json'
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                   EXPECTED_SOURCE=str(source), EXPECTED_RECORD=str(record),
                   RECIPE_RAN=str(parent / 'recipe-ran'))
        env.pop('AUTOEQ_REPORT_PRODUCER', None)
        return source, cargo, record, env

    def run_lane(self, parent, profile, **extra):
        parent = parent.resolve()
        source, cargo, record, env = self.fixture(parent)
        env.update(EXPECTED_PROFILE=profile, **extra)
        target = parent / 'owned target'
        env['CARGO_TARGET_DIR'] = str(target)
        command = product_command('full-native', {'autoeq': {}}, str(cargo),
                                  root=source, qa_profile=profile, producer_record=record)
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True)
        return result, record, parent / 'recipe-ran', target

    def test_full_native_prepares_and_binds_helper_for_each_native_profile(self):
        for profile in ('release', 'qa'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                result, record, ran, target = self.run_lane(Path(temporary), profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(ran.exists())
                saved = json.loads(record.read_text())
                self.assertEqual(saved['requested_profile'], profile)
                self.assertTrue(Path(saved['executable']).is_relative_to(target))
                self.assertFalse(saved['artifact']['profile']['test'])
                self.assertEqual(saved['exit_code'], 0)

    def test_native_cargo_rebuild_cannot_replace_bound_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, ran, target = self.run_lane(
                Path(temporary), 'release', REBUILD_CARGO_PRODUCER='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(ran.exists())
            saved = json.loads(record.read_text())
            helper = Path(saved['executable'])
            cargo_artifact = Path(saved['artifact']['executable'])
            self.assertNotEqual(helper, cargo_artifact)
            self.assertTrue(helper.is_relative_to(target))
            self.assertNotEqual(helper.stat().st_ino, cargo_artifact.stat().st_ino)
            self.assertEqual(saved['after_lane_sha256'], saved['executable_sha256'])
            self.assertNotEqual(saved['cargo_executable_after_lane_sha256'],
                                saved['cargo_executable_sha256_at_prepare'])
            self.assertTrue(saved['independent_copy_verified'])

    def test_mutating_bound_helper_still_rejects_native_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, ran, _ = self.run_lane(
                Path(temporary), 'qa', MUTATE_BOUND_PRODUCER='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(ran.exists())
            saved = json.loads(record.read_text())
            self.assertNotEqual(saved['after_lane_sha256'], saved['executable_sha256'])
            self.assertIn('changed its prepared report-producer artifact', result.stderr)

    def test_failed_helper_build_never_runs_native_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, ran, _ = self.run_lane(Path(temporary), 'release', FAIL_BUILD='1')
            self.assertEqual(result.returncode, 19, result.stderr)
            self.assertFalse(ran.exists())

    def test_test_harness_artifact_cannot_stand_in_for_report_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, record, ran, _ = self.run_lane(Path(temporary), 'qa', BAD_TEST_ARTIFACT='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(ran.exists())
            self.assertIn('ordinary report-producer executable', result.stderr)


if __name__ == '__main__':
    unittest.main()
