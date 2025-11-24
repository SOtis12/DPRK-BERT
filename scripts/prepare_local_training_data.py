#!/usr/bin/env python3
"""Merge data/local artifacts into DPRK-BERT training JSON files.

This script consumes the cleaned outputs produced by
`scripts/build_local_dataset.py` and writes `local_training_data/train.json`
and `local_training_data/validation.json` in the format expected by
`DPRK-BERT-master/mlm_trainer.py` and other legacy utilities.

It keeps the process reproducible (fixed seed) and enforces simple heuristics
to avoid leaking dictionary noise or low-Korean segments into the MLM corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


KOREAN_RE = re.compile(r"[\uac00-\ud7af]")


def has_korean(text: str) -> bool:
    return bool(KOREAN_RE.search(text or ""))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def korean_ratio(text: str) -> float:
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return len(KOREAN_RE.findall(stripped)) / len(stripped)


def deduplicate(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for item in items:
        text = item.get("text", "")
        if text in seen:
            continue
        seen.add(text)
        unique.append(item)
    return unique


def make_entry(text: str, source: str, **extra: str) -> Dict[str, str]:
    """Return a record compatible with legacy DPRK-BERT JSON format."""
    entry: Dict[str, str] = {"text": text, "data": text, "source": source}
    for key, value in extra.items():
        if value is not None:
            entry[key] = value
    return entry


@dataclass
class CorpusStats:
    counts: Dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + amount


class LocalTrainingBuilder:
    def __init__(
        self,
        local_data_dir: Path,
        output_dir: Path,
        seed: int = 17,
        dictionary_frac: float = 0.05,
        min_korean_ratio: float = 0.6,
    ) -> None:
        self.local_data_dir = local_data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.dictionary_frac = dictionary_frac
        self.min_korean_ratio = min_korean_ratio
        self.stats = CorpusStats()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_monolingual(self, split: str) -> List[str]:
        path = self.local_data_dir / "monolingual" / f"{split}_nk.txt"
        if not path.exists():
            return []
        lines = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            text = normalize(raw)
            if len(text) < 30:
                continue
            if korean_ratio(text) < self.min_korean_ratio:
                continue
            lines.append(text)
        return lines

    def load_parallel(self, split: str) -> List[Dict[str, str]]:
        path = self.local_data_dir / "parallel" / f"{split}.nk-sk.tsv"
        items: List[Dict[str, str]] = []
        if not path.exists():
            return items
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                north, south = raw.split("\t", maxsplit=1)
            except ValueError:
                continue
            north = normalize(north)
            south = normalize(south)
            if len(north) > 5 and korean_ratio(north) >= self.min_korean_ratio:
                items.append(make_entry(north, f"parallel_{split}_nk", variant="nk"))
            if len(south) > 5 and korean_ratio(south) >= 0.4:
                items.append(make_entry(south, f"parallel_{split}_sk", variant="sk"))
        return items

    def load_dictionary_entries(self) -> List[Dict[str, str]]:
        path = self.local_data_dir / "dictionaries" / "nk_lexicon.tsv"
        entries: List[Dict[str, str]] = []
        if not path.exists():
            return entries
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = raw.split("\t")
            if len(parts) < 3:
                continue
            word, category, desc = [normalize(p) for p in parts[:3]]
            if not has_korean(word) and not has_korean(desc):
                continue
            combined = f"{word} ({category}): {desc}".strip()
            if korean_ratio(combined) < 0.3:
                continue
            entries.append(make_entry(combined, "dictionary"))
        return entries

    # ------------------------------------------------------------------
    def build(self) -> None:
        monolingual_train = self.load_monolingual("train")
        monolingual_valid = self.load_monolingual("valid")
        parallel_train = self.load_parallel("train")
        parallel_valid = self.load_parallel("valid") + self.load_parallel("test")
        dictionary_entries = self.load_dictionary_entries()

        self.stats.bump("monolingual_train", len(monolingual_train))
        self.stats.bump("monolingual_valid", len(monolingual_valid))
        self.stats.bump("parallel_train_entries", len(parallel_train))
        self.stats.bump("parallel_valid_entries", len(parallel_valid))

        dictionary_limit = min(
            len(dictionary_entries),
            max(1, int(self.dictionary_frac * max(1, len(monolingual_train)))),
        )
        self.rng.shuffle(dictionary_entries)
        dictionary_subset = dictionary_entries[:dictionary_limit]
        self.stats.bump("dictionary_used", len(dictionary_subset))

        train_items: List[Dict[str, str]] = []
        for text in monolingual_train:
            train_items.append(make_entry(text, "monolingual_nk"))
        train_items.extend(parallel_train)
        train_items.extend(dictionary_subset)

        valid_items: List[Dict[str, str]] = []
        for text in monolingual_valid:
            valid_items.append(make_entry(text, "monolingual_nk"))
        valid_items.extend(parallel_valid)

        train_items = deduplicate(train_items)
        valid_items = deduplicate(valid_items)
        self.stats.bump("train_total", len(train_items))
        self.stats.bump("valid_total", len(valid_items))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / "train.json"
        valid_path = self.output_dir / "validation.json"
        with train_path.open("w", encoding="utf-8") as fh:
            json.dump({"data": train_items}, fh, ensure_ascii=False, indent=2)
        with valid_path.open("w", encoding="utf-8") as fh:
            json.dump({"data": valid_items}, fh, ensure_ascii=False, indent=2)

        stats_path = self.output_dir / "metadata.json"
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(self.stats.counts, fh, ensure_ascii=False, indent=2)

        print(f"✅ Wrote {len(train_items)} train items and {len(valid_items)} validation items")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DPRK-BERT training data from data/local")
    parser.add_argument("--local-data", default="data/local")
    parser.add_argument("--output", default="local_training_data")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dictionary-frac", type=float, default=0.05, help="Fraction of dictionary entries relative to monolingual count")
    parser.add_argument("--min-korean-ratio", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = LocalTrainingBuilder(
        local_data_dir=Path(args.local_data),
        output_dir=Path(args.output),
        seed=args.seed,
        dictionary_frac=args.dictionary_frac,
        min_korean_ratio=args.min_korean_ratio,
    )
    builder.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
