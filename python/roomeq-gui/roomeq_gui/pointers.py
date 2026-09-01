"""Small RFC 6901 helpers used as the editor's stable field addresses."""
from __future__ import annotations
from copy import deepcopy
from typing import Any

def escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")

def join(pointer: str, token: str | int) -> str:
    return f"{pointer}/{escape(str(token))}" if pointer else f"/{escape(str(token))}"

def tokens(pointer: str) -> list[str]:
    if pointer == "": return []
    if not pointer.startswith("/"): raise ValueError(f"not a JSON Pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]

def get(value: Any, pointer: str, default: Any = None) -> Any:
    current = value
    try:
        for token in tokens(pointer): current = current[int(token)] if isinstance(current, list) else current[token]
        return current
    except (KeyError, IndexError, ValueError, TypeError): return default

def set(value: Any, pointer: str, replacement: Any) -> Any:
    result = deepcopy(value)
    parts = tokens(pointer)
    if not parts: return replacement
    current = result
    for token in parts[:-1]:
        if isinstance(current, list):
            index = int(token)
            while len(current) <= index: current.append({})
            current = current[index]
        else: current = current.setdefault(token, {})
    last = parts[-1]
    if isinstance(current, list):
        index = int(last)
        while len(current) <= index: current.append(None)
        current[index] = replacement
    else: current[last] = replacement
    return result
