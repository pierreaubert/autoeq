"""Explicit Cargo scopes and equivalent probe locations before and after extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LIBRARY_SCENARIOS = ("clean-check", "clean-build", "warm-noop", "private-edit", "affected-edit", "public-edit", "test-edit", "test-build")
WORKSPACE_SCENARIOS = ("execution", "full-native", "product-cli", "report-native", "report-wasm", "report-gpui")


@dataclass(frozen=True)
class Case:
    name: str
    package: str
    probe_file: str | None
    test_probe_file: str | None = None
    public_type: str | None = None
    public_function_file: str | None = None


def cases(packages: dict[str, dict], selected: str | None, full: bool) -> list[Case]:
    if not full:
        if selected not in packages:
            raise ValueError(f"unknown workspace package: {selected}")
        return [Case(str(selected), str(selected), None)]
    objective = "autoeq-objective" if "autoeq-objective" in packages else "autoeq-optim"
    dsp = "roomeq-dsp" if "roomeq-dsp" in packages else "roomeq-engine"
    return [
        Case("core", "autoeq-core", "crates/autoeq-core/src/param_utils.rs",
             "crates/autoeq-core/src/curve_interpolation_tests.rs", "Curve"),
        Case("objective", objective, f"crates/{objective}/src/optim/compute.rs",
             f"crates/{objective}/src/loss/epa/score/tests.rs", "ObjectiveData"),
        Case("optimizer", "autoeq-optim", "crates/autoeq-optim/src/optim/optimize.rs",
             "crates/autoeq-optim/src/optim/de/tests.rs", "OptimParams"),
        Case("config", "roomeq-model", None,
             "crates/roomeq-model/src/validation_rules/tests.rs", "RoomConfig"),
        Case("engine", "roomeq-engine", "crates/roomeq-engine/src/eq/optimize.rs",
             "crates/roomeq-engine/src/channel_fir/tests.rs", "room_result::RoomOptimizationResult"),
        Case("dsp", dsp, f"crates/{dsp}/src/physical_routing.rs",
             f"crates/{dsp}/src/physical_routing.rs", public_function_file=f"crates/{dsp}/src/physical_routing.rs"),
        Case("workflow", "roomeq-workflow", "crates/roomeq-workflow/src/room_optimization.rs",
             "crates/roomeq-workflow/src/ctc/tests.rs", "RoomPipelineRequest<'_>"),
        Case("renderer", "autoeq-report-wasm", "crates/autoeq-report-wasm/src/grid.rs",
             "crates/autoeq-report-wasm/src/acoustic_reflections.rs", "ReportPayload"),
    ]


def affected_packages(owner: str, packages: dict[str, dict]) -> list[str]:
    """Widen over normal/dev/build/optional declarations, never guess a subset."""
    reverse = {name: set() for name in packages}
    for name, package in packages.items():
        for dependency in package["dependencies"]:
            if dependency["name"] in reverse:
                reverse[dependency["name"]].add(name)
    seen = {owner}
    queue = [owner]
    while queue:
        for dependent in reverse[queue.pop()]:
            if dependent not in seen:
                seen.add(dependent)
                queue.append(dependent)
    return sorted(seen)


def cargo_cmd(scenario: str, package: str | list[str], profile: str, target: Path, cargo: str = "cargo") -> list[str]:
    package_names = [package] if isinstance(package, str) else package
    if scenario in ("clean-check", "warm-noop", "private-edit"):
        command = [cargo, "check", "--lib"]
    elif scenario == "clean-build":
        command = [cargo, "build", "--lib"]
    elif scenario in ("test-build", "test-edit"):
        command = [cargo, "test", "--lib", "--all-features", "--no-run"]
    elif scenario in ("affected-edit", "public-edit"):
        command = [cargo, "test", "--lib", "--tests", "--all-features", "--no-run"]
    else:
        raise ValueError(f"unknown library scenario: {scenario}")
    command += ["--offline", "--locked", "--profile", profile, "--target-dir", str(target), "--timings", "--message-format=json-render-diagnostics"]
    for name in package_names:
        command += ["-p", name]
    return command


def native_listing(cargo: str, target: Path, profile: str) -> list[str]:
    return [cargo, "nextest", "list", "--locked", "--offline", "--workspace", "--lib", "--tests", "--benches", "--all-features", "--cargo-profile", profile, "--target-dir", str(target), "--list-type", "binaries-only", "--message-format", "json"]
