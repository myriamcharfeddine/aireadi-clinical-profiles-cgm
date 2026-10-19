#!/usr/bin/env python3
"""Pre-commit scan: report real AI-READI participant IDs appearing in identifying contexts in the
repository's files (tracked files when run inside git, otherwise every file outside restricted/).
IDs are read from the local parquet (AIREADI_DATA_DIR). Exit code 1 if any are found."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgmprofiles.io import PID, read_columns  # noqa: E402

PATTERNS = [re.compile(r"participant[_ ]?ids?\W{0,6}(\d{4})(?!\d)", re.I), re.compile(r"\bpids?\W{0,6}(\d{4})(?!\d)", re.I),
            re.compile(r"(?<![0-9A-Za-z.\-])['\"](\d{4})['\"](?![0-9A-Za-z])"), re.compile(r"participant\s+(\d{4})(?!\d)", re.I)]
BINARY = (".png", ".pdf")
SKIP_DIRS = {"restricted", ".git", "__pycache__", ".pytest_cache"}


def files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode == 0 and listed.stdout.strip():
        return [ROOT / f for f in listed.stdout.splitlines() if f]
    return [p for p in ROOT.rglob("*") if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)]


def scan(ids: set[str]) -> list[tuple[str, str, str]]:
    hits = []
    for path in files():
        name = str(path.relative_to(ROOT))
        texts = [name] + ([] if path.suffix in BINARY or not path.exists() else [path.read_text(errors="ignore")])
        for text in texts:
            for pattern in PATTERNS:
                for m in pattern.finditer(text):
                    if m.group(1) in ids:
                        hits.append((name, m.group(1), text[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")))
    return hits


def main() -> int:
    ids = set(read_columns([PID])[PID].unique())
    hits = scan(ids)
    for h in hits:
        print("HIT", h[0], h[1], "|", h[2])
    print(f"{len(hits)} participant-ID hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
