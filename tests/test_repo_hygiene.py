"""Repository rules: no torch, no absolute paths, no large files, no participant IDs, no small cells."""
import re
import subprocess
from pathlib import Path

import pandas as pd

from cgmprofiles.disclosure import is_small

from conftest import needs_data

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"restricted", ".git", "__pycache__", ".pytest_cache"}
MAX_BYTES = 5 * 1024 * 1024
COUNT_COLUMNS = re.compile(r"^(n|n_.*|C\d|G\d|.*_count|count|assigned_C\d|sizes?|out_of_range)$")
NOT_PARTICIPANT_COUNTS = {"n_clusters", "n_permutations", "n_features", "n_runs"}


def repo_files(suffixes: tuple[str, ...] | None = None) -> list[Path]:
    files = [p for p in ROOT.rglob("*") if p.is_file() and not SKIP.intersection(p.relative_to(ROOT).parts) and ".egg-info" not in str(p)]
    return [p for p in files if suffixes is None or p.suffix in suffixes]


def test_no_torch_import():
    pattern = re.compile(r"^\s*(import torch|from torch)", re.M)
    assert not [p for p in repo_files((".py",)) if pattern.search(p.read_text())]


def test_no_absolute_paths_in_code():
    pattern = re.compile("/" + "home/|/" + "Users/|[A-Za-z]:" + re.escape(chr(92)))
    offenders = [p.name for p in repo_files((".py", ".yaml", ".yml", ".sh")) if pattern.search(p.read_text())]
    assert not offenders, offenders


def test_no_file_over_5_mb():
    assert not [p.name for p in repo_files() if p.stat().st_size > MAX_BYTES]


def test_no_counts_of_1_to_4_in_tracked_tables():
    offenders = []
    for path in repo_files((".csv",)):
        if "verification" in path.parts:
            continue
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        for column in [c for c in table.columns if COUNT_COLUMNS.match(c) and c not in NOT_PARTICIPANT_COUNTS]:
            for value in table[column]:
                head = value.split(" ")[0]
                if head.isdigit() and is_small(int(head)):
                    offenders.append(f"{path.name}:{column}={value}")
    assert not offenders, offenders[:10]


@needs_data
def test_no_participant_ids():
    result = subprocess.run(["python", str(ROOT / "tools" / "scan_participant_ids.py")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
