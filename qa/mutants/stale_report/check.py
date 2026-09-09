#!/usr/bin/env python3
"""Omit committing the fresh deployed response after correction reversion."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from escaped_defect_mutation import run

if __name__ == "__main__":
    run(Path(__file__).with_name("manifest.json"))
