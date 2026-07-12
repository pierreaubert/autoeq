#!/usr/bin/env python3

import json
import struct
import tempfile
import unittest
from pathlib import Path

import mdat2csv


class Mdat2CsvTests(unittest.TestCase):
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

        self.assertEqual(payload['speakers']['Front_Left']['name'], 'Front Left')

    def test_identifier_collapses_punctuation_and_whitespace(self):
        self.assertEqual(
            mdat2csv.sanitize_identifier('Left &  Right together'),
            'Left_Right_together',
        )


if __name__ == '__main__':
    unittest.main()
