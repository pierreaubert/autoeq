"""Negative contracts for the checked-in pairwise manifest validator."""

import unittest

from scripts.check_parameter_matrix import validate_manifest


class ParameterMatrixManifestTests(unittest.TestCase):
    def test_accepts_variable_cardinality_domains(self):
        manifest = {
            "max_rows": 4,
            "dimensions": {"phase": ["absent", "coherent"], "crossover": ["fixed", "auto"]},
            "rows": [[0, 0], [0, 1], [1, 0], [1, 1]],
        }
        self.assertEqual(validate_manifest(manifest), [])

    def test_rejects_fabricated_row_budget(self):
        manifest = {
            "max_rows": 25,
            "dimensions": {"phase": ["absent", "coherent"]},
            "rows": [[0]],
        }
        self.assertTrue(any("max_rows" in error for error in validate_manifest(manifest)))

    def test_rejects_boolean_row_budget_and_boolean_indices(self):
        manifest = {
            "max_rows": True,
            "dimensions": {"phase": ["absent", "coherent"]},
            "rows": [[True]],
        }
        self.assertTrue(any("max_rows" in error for error in validate_manifest(manifest)))
        manifest["max_rows"] = 2
        self.assertTrue(any("not an integer" in error for error in validate_manifest(manifest)))

    def test_rejects_duplicate_semantic_axis_values_and_rows(self):
        manifest = {
            "dimensions": {"phase": ["absent", "absent"]},
            "rows": [[0], [0]],
        }
        errors = validate_manifest(manifest)
        self.assertTrue(any("repeats a semantic value" in error for error in errors))
        self.assertTrue(any("duplicate rows" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
