#!/usr/bin/env python3

import json
import struct
import tempfile
import unittest
from pathlib import Path

import mdat2csv


class Mdat2CsvTests(unittest.TestCase):
    def test_measurements_can_have_different_point_counts(self):
        fields = [('I', 'dataLength'), ('I', 'sampleRate')]
        first = struct.pack('>ii', 1068, 48000)
        reference = b'\x73\x71\x00\x7e\x00\x13'
        data = first
        expected = [0]
        for length in (1068, 448, 448):
            data += reference
            expected.append(len(data))
            data += struct.pack('>ii', length, 48000)
        data += bytes(1068 * 4)
        self.assertEqual(
            mdat2csv.find_measurement_primitives(data, 0, fields), expected,
        )

    def test_real_mixed_length_mdat_exports_all_four_curves(self):
        path = (Path(__file__).resolve().parents[1]
                / 'data_tests/roomeq/measured/test/2.2.mdat')
        if not path.exists():
            self.skipTest('Local mixed-length REW fixture is unavailable')
        measurements = mdat2csv.parse_mdat(path)
        self.assertEqual([len(m['freq']) for m in measurements], [1068, 1068, 448, 448])
        self.assertEqual([m['name'] for m in measurements], [
            'L No EQ Sep 1', 'R No EQ Sep 1',
            'L Sub No EQ Sep 1', 'R Sub No EQ Sep 1',
        ])
        for measurement in measurements:
            self.assertEqual(len(measurement['spl']), len(measurement['freq']))
            self.assertEqual(len(measurement['phase']), len(measurement['freq']))
        with tempfile.TemporaryDirectory() as directory:
            paths = [mdat2csv.export_csv(m, directory) for m in measurements]
            self.assertEqual(len(set(paths)), 4)
            output = mdat2csv.export_recordings_json(measurements, paths, directory)
            self.assertEqual(len(json.loads(Path(output).read_text())['speakers']), 4)

    def test_sub_zero_spl_tails_are_not_rejected(self):
        raw = (-24.0, 42.0, 68.0, 86.0)
        calibrated = (-23.0, 43.0, 69.0, 87.0)
        data = struct.pack('>4f4f', *(raw + calibrated))
        arrays = [
            (0, 4, 0, '[F'),
            (16, 4, 16, '[F'),
        ]

        observed_raw, observed_calibrated = mdat2csv.identify_spl_array(arrays, data, 4)

        self.assertEqual(observed_raw, raw)
        self.assertEqual(observed_calibrated, calibrated)

    def test_short_description_preserves_spaces_and_date(self):
        def string(value):
            encoded = value.encode('utf-8')
            return b'\x74' + struct.pack('>H', len(encoded)) + encoded

        data = (bytes(8) + string('SLOPE_24DB')
                + string('Notes\nPrivate room details')
                + string('L No EQ Sep 1') + string('-20 dBFS'))
        names = mdat2csv.find_measurement_names(data, [(0, 1, 0, '[F')], [len(data)])
        self.assertEqual(names, ['L No EQ Sep 1'])

    def test_embedded_notes_are_reduced_to_the_channel_label(self):
        private_note = (
            'Front Left\n'
            'Room dimensions and private equipment notes\n'
            'Delay and clock details'
        ).encode('utf-8')
        prefix = bytes(8)
        encoded_note = b'\x74' + struct.pack('>H', len(private_note)) + private_note
        data = prefix + encoded_note
        ir_arrays = [(0, 1, 0, '[F')]

        names = mdat2csv.find_measurement_names(data, ir_arrays, [len(data)])

        self.assertEqual(names, ['Front Left'])
        self.assertNotIn('private', names[0])

    def test_measurement_title_is_preferred_over_embedded_notes(self):
        notes = (
            'Front Left\n'
            'CDSP Straight Through - REFERENCE\n'
            'Kef R3M ported (part of 5.1 config)\n'
            'Room dimensions and timing details'
        ).encode('utf-8')
        title = b'L r3m_ported_48k_FL_260706a_direct Jul 6 -20 dBFS'
        data = (
            bytes(8)
            + b'\x74'
            + struct.pack('>H', len(notes))
            + notes
            + b'\x74'
            + struct.pack('>H', len(title))
            + title
        )
        ir_arrays = [(0, 1, 0, '[F')]

        names = mdat2csv.find_measurement_names(data, ir_arrays, [len(data)])

        self.assertEqual(names, ['L r3m_ported_48k_FL_260706a_direct Jul 6 -20 dBFS'])

    def test_real_mdat_uses_measurement_title(self):
        path = (
            Path(__file__).resolve().parents[1]
            / 'data_tests/roomeq/measured/5.1_kef'
            / '260706a_1800_cdsp_straightthru_baseline.mdat'
        )

        measurements = mdat2csv.parse_mdat(path)

        self.assertEqual(
            measurements[0]['name'],
            'L r3m_ported_48k_FL_260706a_direct Jul 6 -20 dBFS',
        )

    def test_html_descriptions_do_not_retain_raw_metadata(self):
        html = b'<BODY>Jul 6<BR>18:00<BR>20 to 20000 Hz<BR>20 to 90 dB SPL</HTML>'

        descriptions = mdat2csv.find_html_descriptions(html)

        self.assertEqual(descriptions[0]['spl_range'], '20 to 90 dB SPL')
        self.assertNotIn('raw', descriptions[0])

    def test_recordings_export_contains_only_sanitized_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / 'Front_Left.csv'
            csv_path.write_text('freq_hz,spl_db,phase_deg\n', encoding='utf-8')
            output = mdat2csv.export_recordings_json(
                [{'name': 'Front Left'}],
                [str(csv_path)],
                directory,
            )
            payload = json.loads(Path(output).read_text(encoding='utf-8'))

        self.assertEqual(payload['version'], '2.1.0')
        self.assertEqual(payload['speakers']['Front_Left']['name'], 'Front Left')

    def test_identifier_collapses_punctuation_and_whitespace(self):
        self.assertEqual(
            mdat2csv.sanitize_identifier('Left &  Right together'),
            'Left_Right_together',
        )


if __name__ == '__main__':
    unittest.main()
