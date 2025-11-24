#!/usr/bin/env python3
"""Utility helpers for reclaiming disk space safely.

The script deletes Python-generated cache folders/files (``__pycache__``,
``*.pyc`` and similar artifacts) while skipping directories that matter for the
project (Resources, datasets, virtual environments, checkpoints, etc.).  It
also prints a short summary of large training/data artifacts so it is obvious
where the remaining space is going.  Destructive removal of those big
directories is intentionally opt-in via ``--remove-artifacts``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, List, Sequence, Tuple

# Directories we never descend into while cleaning caches.
SKIP_TOP_LEVEL = {
    "venv_py311",  # local virtualenv, can be recreated but leave it unless asked
    ".git",
    "Resources",
}

# Directory names that only contain Python cache/metadata.
CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}

# File glob patterns that are safe to delete.
CACHE_FILE_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*~",
    "*.tmp",
    ".DS_Store",
)

# Large artifacts we only delete when explicitly requested.
ARTIFACTS = [
    ("dprk_bert_test_output", "TPU test training checkpoints + final model"),
    ("dprk_bert_enhanced_output", "Enhanced fine-tuning checkpoints"),
    ("dprk_bert_local_output", "Local checkpoint experiments"),
    ("experiment_outputs", "adhoc experiment exports"),
    ("experiment_outputs/local-cache-fast-pass", "cached run artifacts"),
    ("experiment_outputs/local-validation-run", "validation artifacts"),
    ("experiment_outputs/local-smoke-test", "smoke test outputs"),
    ("experiment_outputs/local-cache-fast-pass-sample", "sample cache outputs"),
    ("DPRK-BERT-Public", "upstream released checkpoints"),
    ("local_training_data", "locally materialized train/validation JSON"),
    ("enhanced_training_data", "enhanced training dataset JSON"),
    ("final_bert_training_dataset.json", "combined JSON dataset"),
]


def human_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if num < 1024 or unit == units[-1]:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}EB"


def du_human(path: Path) -> str:
    """Return a human readable size for *path* using ``du`` when available."""

    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.split("\t", 1)[0].strip()
        return token
    except (FileNotFoundError, subprocess.CalledProcessError):
        total = 0
        if path.is_dir():
            for root, _, files in os.walk(path):
                for name in files:
                    fp = Path(root) / name
                    try:
                        total += fp.stat().st_size
                    except FileNotFoundError:
                        continue
        elif path.exists():
            total = path.stat().st_size
        return human_bytes(float(total))


def should_skip(rel_parts: Sequence[str]) -> bool:
    return bool(rel_parts) and rel_parts[0] in SKIP_TOP_LEVEL


def collect_cache_targets(root: Path) -> Tuple[List[Path], List[Path]]:
    dir_targets: set[Path] = set()
    file_targets: set[Path] = set()

    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        if current_path == root:
            rel_parts: Tuple[str, ...] = ()
        else:
            rel_parts = current_path.relative_to(root).parts

        if should_skip(rel_parts):
            dirnames[:] = []
            continue

        for dirname in list(dirnames):
            if dirname in CACHE_DIR_NAMES:
                dir_targets.add(current_path / dirname)
                dirnames.remove(dirname)

        for filename in filenames:
            if any(fnmatch.fnmatch(filename, pattern) for pattern in CACHE_FILE_PATTERNS):
                file_targets.add(current_path / filename)

    sorted_dirs = sorted(dir_targets, key=lambda p: len(p.parts), reverse=True)
    resolved_dirs = [p.resolve() for p in sorted_dirs]

    filtered_files: List[Path] = []
    for file_path in file_targets:
        resolved = file_path.resolve()
        if any(resolved.is_relative_to(dir_path) for dir_path in resolved_dirs):
            continue
        filtered_files.append(file_path)

    filtered_files.sort()
    return sorted_dirs, filtered_files


def remove_path(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"DRY-RUN would remove {path}")
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            return


def summarize_artifacts(root: Path) -> None:
    print("\nLarge artifacts (kept unless --remove-artifacts is provided):")
    for rel_path, desc in ARTIFACTS:
        target = root / rel_path
        if not target.exists():
            continue
        size = du_human(target)
        print(f" - {rel_path:<35} {size:>8}  -> {desc}")


def remove_artifacts(root: Path, aggressive: bool, dry_run: bool) -> None:
    if not aggressive:
        summarize_artifacts(root)
        return

    print("Removing large artifacts (aggressive mode)...")
    for rel_path, desc in ARTIFACTS:
        target = root / rel_path
        if not target.exists():
            continue
        print(f" - deleting {rel_path} ({desc})")
        remove_path(target, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        help="Project root (defaults to repository root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--remove-artifacts",
        action="store_true",
        help="Also delete large generated artifacts listed in the script.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Cleaning caches under {root}...\n")

    dir_targets, file_targets = collect_cache_targets(root)

    if not dir_targets and not file_targets:
        print("No cache directories or files detected.")
    else:
        for path in dir_targets:
            print(f"Removing directory {path.relative_to(root)}")
            remove_path(path, dry_run=args.dry_run)
        for path in file_targets:
            print(f"Removing file      {path.relative_to(root)}")
            remove_path(path, dry_run=args.dry_run)

    remove_artifacts(root, aggressive=args.remove_artifacts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
