"""Executable test/recipe resolution. Mutation execution remains a separate gate."""
import json
import re
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"id", "stage", "invariant", "regression_test", "pr_recipe", "mutant_fixture"}


def command_json(command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"discovery failed: {command!r}\n{result.stderr}")
    return json.loads(result.stdout)


def discover_tests(packages):
    command = ["cargo", "nextest", "list", "--offline", "--lib", "--message-format", "json"]
    for package in sorted(packages):
        command.extend(["-p", package])
    tests = set()
    for suite in command_json(command).get("rust-suites", {}).values():
        if suite.get("kind") != "lib":
            continue
        for name, case in suite.get("testcases", {}).items():
            if not case.get("ignored") and case.get("filter-match", {}).get("status") == "matches":
                tests.add((suite["package-name"], suite["binary-name"], name))
    if not tests:
        raise ValueError("executable discovery selected zero non-ignored tests")
    return tests


def execute_regression(identity):
    """Run one exact, discovered library test and reject Cargo's zero-test success."""
    key = tuple(identity[k] for k in ("package", "target", "name"))
    if key not in discover_tests({identity["package"]}):
        raise ValueError(f"unresolved executable regression: {identity}")
    command = ["cargo", "test", "--offline", "-p", identity["package"], "--lib",
               identity["name"], "--", "--exact", "--format", "terse"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    summary = re.search(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", result.stdout)
    if result.returncode or not summary or summary.groups() != ("1", "0", "0"):
        raise ValueError(f"regression did not execute exactly one passing test: {command!r}\n"
                         f"{result.stdout}\n{result.stderr}")
    return {"test": identity, "command": command, "executed": 1, "passed": 1}


def just_calls(command):
    """Resolve literal invocations only, never guess dynamic shell recipe names."""
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        return set()
    if len(words) >= 2 and words[0] == "just" and not words[1].startswith("-"):
        return {words[1]}
    return set()


def workflow_roots(path):
    # Only run-step commands count, not comments, display names or arbitrary text.
    # Literal inline and block forms cover this repository's workflow commands.
    roots, block_indent = set(), None
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if block_indent is not None:
            if not stripped or indent > block_indent:
                roots.update(just_calls(stripped))
                continue
            block_indent = None
        if stripped.startswith("run:"):
            command = stripped[4:].strip()
            if command in {"|", ">", "|-", ">-"}:
                block_indent = indent
            else:
                roots.update(just_calls(command))
    return roots


def reachable_recipes(recipes, roots):
    reachable, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in reachable or name not in recipes:
            continue
        reachable.add(name)
        recipe = recipes[name]
        pending.extend(dep["recipe"] for dep in recipe.get("dependencies", []))
        for line in recipe.get("body", []):
            if all(isinstance(fragment, str) for fragment in line):
                pending.extend(just_calls("".join(line)))
    return reachable


def validate(defects, tests, recipes, reachable, root=ROOT):
    errors, seen = [], set()
    if not defects:
        return ["escaped-defects registry is empty"]
    for item in defects:
        identity = item.get("id", "<unknown>")
        if identity in seen:
            errors.append(f"duplicate defect ID: {identity}")
        seen.add(identity)
        if missing := REQUIRED - item.keys():
            errors.append(f"{identity}: missing {sorted(missing)}")
        test = item.get("regression_test")
        if not isinstance(test, dict) or set(test) != {"package", "target", "name"}:
            errors.append(f"{identity}: regression_test must identify package, target and fully qualified name")
        elif not all(isinstance(test[k], str) and test[k].strip() for k in test):
            errors.append(f"{identity}: empty or invalid test identity")
        elif "::" not in test["name"] or tuple(test[k] for k in ("package", "target", "name")) not in tests:
            errors.append(f"{identity}: unresolved or ignored executable test {test}")
        recipe = item.get("pr_recipe")
        if recipe not in recipes:
            errors.append(f"{identity}: missing recipe {recipe!r}")
        elif recipe not in reachable:
            errors.append(f"{identity}: recipe {recipe!r} is unreachable from .github/workflows/ci.yml")
        fixture = item.get("mutant_fixture")
        if not isinstance(fixture, str) or not fixture:
            errors.append(f"{identity}: missing mutant fixture path")
        else:
            location = (root / fixture).resolve()
            if not location.is_relative_to(root.resolve()):
                errors.append(f"{identity}: mutant fixture escapes repository")
            elif not (location / "manifest.json").is_file():
                errors.append(f"{identity}: missing executable mutant manifest (README is not evidence)")
            else:
                manifest = json.loads((location / "manifest.json").read_text())
                if manifest.get("id") != identity or manifest.get("regression_test") != test:
                    errors.append(f"{identity}: mutant manifest does not match registered defect/test identity")
                runner = manifest.get("runner")
                if not isinstance(runner, str) or not runner:
                    errors.append(f"{identity}: mutant manifest has no runner")
                elif not (location / runner).resolve().is_relative_to(location) or not (location / runner).is_file():
                    errors.append(f"{identity}: missing or escaping mutant runner")
    return errors


def check(path):
    defects = json.loads(path.read_text(encoding="utf-8")).get("defects", [])
    packages = {
        item["regression_test"]["package"] for item in defects
        if isinstance(item.get("regression_test"), dict)
        and isinstance(item["regression_test"].get("package"), str)
        and item["regression_test"]["package"]
    }
    tests = discover_tests(packages) if packages else set()
    recipes = command_json(["just", "--dump", "--dump-format", "json"])["recipes"]
    reachable = reachable_recipes(recipes, workflow_roots(ROOT / ".github/workflows/ci.yml"))
    if errors := validate(defects, tests, recipes, reachable):
        raise ValueError("\n".join(errors))
    return len(defects)
