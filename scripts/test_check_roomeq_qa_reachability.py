import unittest

from scripts.check_roomeq_qa_reachability import (
    missing_suite_runners,
    parse_just_recipes,
)


class RoomEqQaReachabilityTest(unittest.TestCase):
    def test_recipe_parser_keeps_commands_and_dependencies(self) -> None:
        recipes = parse_just_recipes(
            "root: child\n"
            "    cargo run --bin parent\n"
            "child:\n"
            "    cargo run --bin child\n"
        )
        self.assertIn("child", recipes["root"])
        self.assertIn("parent", recipes["root"])
        self.assertIn("child", recipes["child"])

    def test_all_declared_suite_runners_are_reachable(self) -> None:
        self.assertEqual(missing_suite_runners(), {})


if __name__ == "__main__":
    unittest.main()
