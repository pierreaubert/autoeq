from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any

@dataclass(frozen=True)
class Series: id: str; x: list[float]; y: list[float]; label: str
def display_series(identifier: str, frequencies: list[Any], values: list[Any], label: str = "") -> Series:
    if len(frequencies) != len(values): raise ValueError("curve frequency/value lengths differ")
    pairs = [(float(x), float(y)) for x, y in zip(frequencies, values)]
    if any(not isfinite(x) or x <= 0 or not isfinite(y) for x, y in pairs): raise ValueError("curve values must be finite and frequencies positive")
    if len(pairs) > 2000:
        step = (len(pairs) - 1) / 1999; pairs = [pairs[round(index * step)] for index in range(2000)]
    return Series(identifier, [x for x, _ in pairs], [y for _, y in pairs], label)
class ResultReview:
    def __init__(self, result: dict[str, Any]): self.result = result
    def overview(self) -> dict[str, Any]:
        source = self.result
        keys = ("version", "algorithm", "loss", "iterations", "pre_score", "post_score", "timestamp")
        data = {key: source[key] for key in keys if key in source}
        if "pre_score" in data and "post_score" in data: data["improvement"] = data["post_score"] - data["pre_score"]
        for key in ("epa", "fir_masking", "bass_management", "advisories"): 
            if key in source: data[key] = source[key]
        return data
    def curves(self) -> list[Series]:
        out: list[Series] = []
        def visit(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                x = value.get("frequencies", value.get("frequency_hz")); y = value.get("values", value.get("spl_db", value.get("response_db")))
                if isinstance(x, list) and isinstance(y, list):
                    try: out.append(display_series(path or "curve", x, y, path.rsplit("/", 1)[-1]))
                    except ValueError: pass
                for key, child in value.items(): visit(child, f"{path}/{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value): visit(child, f"{path}/{index}")
        visit(self.result); return out
