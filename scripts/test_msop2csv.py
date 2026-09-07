#!/usr/bin/env python3

import gzip
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import msop2csv


def complex_from_db_phase(spl_db, phase_deg):
    magnitude = 10.0 ** (spl_db / 20.0)
    radians = math.radians(phase_deg)
    return magnitude * math.cos(radians), magnitude * math.sin(radians)


def build_block(name, freqs, db_phase_pairs, marked=True):
    """Build one raw (decompressed) measurement block."""
    payload = msop2csv.BLOCK_MARKER + struct.pack('<i', len(freqs))
    payload += struct.pack(f'<{len(freqs)}d', *freqs)
    flat = []
    for spl_db, phase_deg in db_phase_pairs:
        flat.extend(complex_from_db_phase(spl_db, phase_deg))
    payload += struct.pack(f'<{len(flat)}d', *flat)
    encoded = name.encode('utf-8')
    prefix = (b'\x01' if marked else b'\x00') + struct.pack('<i', len(encoded)) + encoded
    return prefix + payload


def write_msop(path, blocks, raw=False, prefix=b'', suffix=b''):
    payload = prefix + b''.join(blocks) + suffix
    path.write_bytes(payload if raw else gzip.compress(payload))


def wrap_phase(phase_deg):
    return (phase_deg + 180.0) % 360.0 - 180.0


class Msop2CsvTests(unittest.TestCase):
    def test_round_trip_recovers_freq_spl_and_phase(self):
        freqs = [10.0 + 0.5 * i for i in range(32)]
        curves = [(80.0 - 0.1 * i, wrap_phase(-20.0 - 2.0 * i)) for i in range(32)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('LLP - Sub1.txt', freqs, curves)])

            measurements = msop2csv.parse_msop(path)

        self.assertEqual(len(measurements), 1)
        measurement = measurements[0]
        self.assertEqual(measurement['name'], 'LLP - Sub1.txt')
        self.assertEqual(measurement['freq'], freqs)
        for observed, (expected_db, _) in zip(measurement['spl'], curves):
            self.assertAlmostEqual(observed, expected_db, places=9)
        for observed, (_, expected_phase) in zip(measurement['phase'], curves):
            self.assertAlmostEqual(observed, expected_phase, places=9)

    def test_raw_payload_without_gzip_is_accepted(self):
        freqs = [20.0 + i for i in range(16)]
        curves = [(75.0, 10.0)] * 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('Sub A', freqs, curves)], raw=True)

            measurements = msop2csv.parse_msop(path)

        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]['name'], 'Sub A')

    def test_log_spaced_axis_is_accepted(self):
        freqs = [10.0 * (2.0 ** (i / 10.0)) for i in range(24)]
        curves = [(70.0, -45.0)] * 24
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('Log Sweep', freqs, curves)])

            measurements = msop2csv.parse_msop(path)

        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]['freq'], freqs)

    def test_multiple_blocks_keep_file_order(self):
        freqs = [10.0 + i for i in range(16)]
        blocks = [
            build_block(f'Meas {name}', freqs, [(70.0 + j, 0.0)] * 16)
            for j, name in enumerate('ABC')
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, blocks)

            measurements = msop2csv.parse_msop(path)

        self.assertEqual([m['name'] for m in measurements],
                         ['Meas A', 'Meas B', 'Meas C'])

    def test_duplicate_names_are_deduplicated(self):
        freqs = [10.0 + i for i in range(16)]
        blocks = [build_block('Same', freqs, [(70.0, 0.0)] * 16) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, blocks)

            measurements = msop2csv.parse_msop(path)

        self.assertEqual([m['name'] for m in measurements], ['Same', 'Same_2'])
        with tempfile.TemporaryDirectory() as directory:
            paths = [msop2csv.export_csv(m, directory) for m in measurements]
            self.assertEqual(len(set(paths)), 2)

    def test_missing_name_falls_back_to_measurement_index(self):
        freqs = [10.0 + i for i in range(16)]
        flat = []
        for _ in range(16):
            flat.extend(complex_from_db_phase(70.0, 0.0))
        payload = (msop2csv.BLOCK_MARKER + struct.pack('<i', 16)
                   + struct.pack('<16d', *freqs)
                   + struct.pack(f'<{len(flat)}d', *flat))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            path.write_bytes(gzip.compress(payload))

            measurements = msop2csv.parse_msop(path)

        self.assertEqual([m['name'] for m in measurements], ['measurement_1'])

    def test_unmarked_name_still_recovers_via_fallback(self):
        freqs = [10.0 + i for i in range(16)]
        curves = [(70.0, 0.0)] * 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('First Block', freqs, curves, marked=False)])

            measurements = msop2csv.parse_msop(path)

        self.assertEqual(measurements[0]['name'], 'First Block')

    def test_stray_markers_without_valid_blocks_are_ignored(self):
        freqs = [10.0 + i for i in range(16)]
        curves = [(70.0, 0.0)] * 16
        decoy = (b'\x01\x00' + struct.pack('<i', 4) + bytes(4 * 8)
                 + b'\x01\x00' + struct.pack('<i', 2000000))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('Real', freqs, curves)],
                       prefix=decoy, suffix=decoy)

            measurements = msop2csv.parse_msop(path)

        self.assertEqual([m['name'] for m in measurements], ['Real'])

    def test_truncated_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.msop'
            path.write_bytes(b'\x1f\x8b too short')

            with self.assertRaises(ValueError):
                msop2csv.parse_msop(path)

    def test_payload_without_measurements_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.msop'
            path.write_bytes(gzip.compress(b'MSO project with no measurements' * 10))

            with self.assertRaises(ValueError):
                msop2csv.parse_msop(path)

    def test_export_csv_and_recordings_json(self):
        freqs = [10.0 + i for i in range(16)]
        curves = [(70.0, -30.0)] * 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.msop'
            write_msop(path, [build_block('Front Left', freqs, curves)])
            measurements = msop2csv.parse_msop(path)

            csv_path = msop2csv.export_csv(measurements[0], directory)
            lines = Path(csv_path).read_text(encoding='utf-8').splitlines()

            self.assertEqual(lines[0], 'freq_hz,spl_db,phase_deg')
            self.assertEqual(len(lines), 17)
            freq, spl, phase = lines[1].split(',')
            self.assertAlmostEqual(float(freq), 10.0)
            self.assertAlmostEqual(float(spl), 70.0, places=6)
            self.assertAlmostEqual(float(phase), -30.0, places=6)

            output = msop2csv.export_recordings_json(measurements, [csv_path], directory)
            payload = json.loads(Path(output).read_text(encoding='utf-8'))

        self.assertEqual(payload['version'], '2.1.0')
        self.assertEqual(payload['speakers']['Front_Left']['name'], 'Front Left')

    def test_identifier_collapses_punctuation_and_whitespace(self):
        self.assertEqual(
            msop2csv.sanitize_identifier('Left &  Right together'),
            'Left_Right_together',
        )


if __name__ == '__main__':
    unittest.main()
