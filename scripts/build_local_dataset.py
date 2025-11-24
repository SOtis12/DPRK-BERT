#!/usr/bin/env python3
"""Build cleaned local training artifacts from offline DPRK resources.

Outputs (under data/local):
- parallel/train|valid|test.nk-sk.tsv   (tab separated NK \t SK)
- monolingual/train_nk.txt & valid_nk.txt (one sentence per line)
- dictionaries/nk_lexicon.tsv            (nk_form \t category \t metadata)
- stats.json + sample snippets in samples/

This script is intentionally local-only and performs conservative cleaning so
that we only promote high confidence text into training.  It avoids touching
the TPU and can be run repeatedly as new resources are parsed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional PDF extraction dependencies.  We fall back gracefully when missing.
try:  # pragma: no cover - import guard
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - best effort import
    pdfplumber = None

try:  # pragma: no cover - import guard
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - import guard
    PdfReader = None


KOREAN_RE = re.compile(r"[\uac00-\ud7af]")
PRIVATE_USE_RE = re.compile(r"[\uf000-\uf8ff]")
PRIVATE_USE_TRANSLATION = str.maketrans(
    {
        "\uf113": "김",  # 김일성 → split into 김/일/성 for standard Hangul
        "\uf114": "일",
        "\uf115": "성",
        "\uf116": "김",  # 김정일 glyphs observed in OCR output
        "\uf117": "정",
        "\uf118": "일",
    }
)


def has_korean(text: str) -> bool:
    return bool(KOREAN_RE.search(text or ""))


def normalize_text(text: str) -> str:
    text = (text or "").translate(PRIVATE_USE_TRANSLATION)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_unknown_private_glyphs(text: str) -> bool:
    return bool(PRIVATE_USE_RE.search(text or ""))


def korean_ratio(text: str) -> float:
    stripped = re.sub(r"\s", "", text or "")
    if not stripped:
        return 0.0
    kor = len(KOREAN_RE.findall(stripped))
    return kor / len(stripped)


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        first_char = fh.read(1)
        if first_char != "\ufeff":
            fh.seek(0)
        reader = csv.DictReader(fh)
        for row in reader:
            yield {k.strip(): (v or "").strip() for k, v in row.items()}


@dataclass
class ParallelPair:
    north: str
    south: str
    source: str
    meta: Dict[str, str]


class LocalDatasetBuilder:
    def __init__(
        self,
        resources_dir: Path,
        output_dir: Path,
        seed: int = 13,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_pdf_pages: int = 40,
        max_dictionary_entries: int = 20000,
        dictionary_fraction: float = 0.08,
        process_pdfs: bool = True,
        max_pdfs_per_folder: Optional[int] = None,
        max_total_pdfs: Optional[int] = None,
        pdf_cache_file: Optional[Path] = None,
    ) -> None:
        self.resources_dir = resources_dir
        self.output_dir = output_dir
        self.parallel_dir = output_dir / "parallel"
        self.monolingual_dir = output_dir / "monolingual"
        self.dictionary_dir = output_dir / "dictionaries"
        self.samples_dir = output_dir / "samples"
        self.seed = seed
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.max_pdf_pages = max_pdf_pages
        self.max_dictionary_entries = max_dictionary_entries
        self.dictionary_fraction = dictionary_fraction
        self.rng = random.Random(seed)
        self.process_pdfs = process_pdfs
        self.max_pdfs_per_folder = max_pdfs_per_folder
        self.max_total_pdfs = max_total_pdfs
        self._pdf_counter = 0
        self.pdf_cache_file = pdf_cache_file
        self._pdf_cache: Dict[str, List[str]] = {}
        if self.pdf_cache_file:
            try:
                if self.pdf_cache_file.exists():
                    self._pdf_cache = json.loads(
                        self.pdf_cache_file.read_text(encoding="utf-8")
                    )
            except Exception as exc:
                print(f"[build_local_dataset] Warning: failed to load PDF cache: {exc}")
                self._pdf_cache = {}
        self.stats: Dict[str, Dict[str, int]] = {
            "parallel_sources": {},
            "monolingual_sources": {},
            "dictionary_sources": {},
        }

    # ------------------------------------------------------------------
    # Parallel data
    # ------------------------------------------------------------------
    def load_parallel_pairs(self) -> List[ParallelPair]:
        pairs: List[ParallelPair] = []
        pairs.extend(self._from_parallel_boost())
        pairs.extend(self._from_gyeoremal())

        unique = {}
        for pair in pairs:
            key = (pair.north, pair.south)
            if key not in unique:
                unique[key] = pair
        deduped = list(unique.values())
        self.rng.shuffle(deduped)
        return deduped

    def _from_parallel_boost(self) -> List[ParallelPair]:
        boost_dir = self.resources_dir / "Parallel Boost"
        collected: List[ParallelPair] = []
        if not boost_dir.exists():
            return collected
        for csv_path in sorted(boost_dir.glob("*.csv")):
            for row in iter_csv_rows(csv_path):
                north = normalize_text(
                    row.get("north_korean")
                    or row.get("north")
                    or row.get("nk_sentence")
                    or row.get("north_text")
                    or ""
                )
                south = normalize_text(
                    row.get("south_korean")
                    or row.get("south")
                    or row.get("sk_sentence")
                    or row.get("south_text")
                    or ""
                )
                if not north or not south:
                    continue
                if has_unknown_private_glyphs(north) or has_unknown_private_glyphs(south):
                    continue
                if len(north) < 6 or len(south) < 6:
                    continue
                if korean_ratio(north) < 0.7 or korean_ratio(south) < 0.7:
                    continue
                meta = {"source": csv_path.name, "domain": row.get("domain", ""), "tags": row.get("tags", "")}
                collected.append(ParallelPair(north=north, south=south, source=csv_path.name, meta=meta))
                self.stats["parallel_sources"].setdefault(csv_path.stem, 0)
                self.stats["parallel_sources"][csv_path.stem] += 1
        return collected

    def _from_gyeoremal(self) -> List[ParallelPair]:
        g_dir = self.resources_dir / "gyeoremal"
        collected: List[ParallelPair] = []
        if not g_dir.exists():
            return collected
        for csv_path in g_dir.glob("*.csv"):
            if not csv_path.name.endswith(".csv"):
                continue
            lower = csv_path.name.lower()
            if "notation" not in lower and "meaning" not in lower:
                continue
            for row in iter_csv_rows(csv_path):
                north = normalize_text(row.get("north_meaning") or row.get("north"))
                south = normalize_text(row.get("south_meaning") or row.get("south"))
                if not north or not south:
                    continue
                if has_unknown_private_glyphs(north) or has_unknown_private_glyphs(south):
                    continue
                if "표준국어대사전" in north or "조선말대사전" in north:
                    continue
                if "표준국어대사전" in south or "조선말대사전" in south:
                    continue
                if len(north) < 4 or len(south) < 4:
                    continue
                if korean_ratio(north) < 0.6 or korean_ratio(south) < 0.6:
                    continue
                meta = {"source": csv_path.name, "word": row.get("word", "")}
                collected.append(ParallelPair(north=north, south=south, source=csv_path.name, meta=meta))
                self.stats["parallel_sources"].setdefault(csv_path.stem, 0)
                self.stats["parallel_sources"][csv_path.stem] += 1
        return collected

    def save_parallel_splits(self, pairs: Sequence[ParallelPair]) -> None:
        total = len(pairs)
        test_count = max(1, int(total * self.test_ratio))
        valid_count = max(1, int(total * self.valid_ratio))
        train = pairs[: total - test_count - valid_count]
        valid = pairs[total - test_count - valid_count : total - test_count]
        test = pairs[total - test_count :]

        self.parallel_dir.mkdir(parents=True, exist_ok=True)
        for name, subset in ("train", train), ("valid", valid), ("test", test):
            path = self.parallel_dir / f"{name}.nk-sk.tsv"
            with path.open("w", encoding="utf-8") as fh:
                for pair in subset:
                    fh.write(f"{pair.north}\t{pair.south}\n")

        self._write_sample(self.samples_dir / "parallel_train_head.tsv", [f"{p.north}\t{p.south}" for p in train[:25]])
        self.stats.setdefault("counts", {})["parallel_total"] = total
        self.stats["counts"]["parallel_train"] = len(train)
        self.stats["counts"]["parallel_valid"] = len(valid)
        self.stats["counts"]["parallel_test"] = len(test)

    # ------------------------------------------------------------------
    # Monolingual data
    # ------------------------------------------------------------------
    def load_monolingual_sentences(self) -> List[Tuple[str, str]]:
        sentences: List[Tuple[str, str]] = []
        sentences.extend(self._from_speeches())
        if self.process_pdfs:
            sentences.extend(self._from_pdfs("PDFs"))
            sentences.extend(self._from_pdfs("With The Century"))
        unique: Dict[str, Tuple[str, str]] = {}
        for text, source in sentences:
            if text not in unique:
                unique[text] = (text, source)
        deduped = list(unique.values())
        self.rng.shuffle(deduped)
        return deduped

    def _from_speeches(self) -> List[Tuple[str, str]]:
        speech_dir = self.resources_dir / "Kim's New Years Speeches"
        collected: List[Tuple[str, str]] = []
        if not speech_dir.exists():
            return collected
        for txt in speech_dir.glob("*.txt"):
            text = txt.read_text(encoding="utf-8", errors="ignore")
            for sentence in re.split(r"[.!?\n]+", text):
                cleaned = normalize_text(sentence)
                if len(cleaned) < 30:
                    continue
                if has_unknown_private_glyphs(cleaned):
                    continue
                if korean_ratio(cleaned) < 0.7:
                    continue
                collected.append((cleaned, txt.name))
                self.stats["monolingual_sources"].setdefault(txt.stem, 0)
                self.stats["monolingual_sources"][txt.stem] += 1
        return collected

    def _from_pdfs(self, folder_name: str) -> List[Tuple[str, str]]:
        target_dir = self.resources_dir / folder_name
        collected: List[Tuple[str, str]] = []
        if not target_dir.exists():
            return collected
        processed_in_folder = 0
        new_pdfs = 0
        for pdf_path in sorted(target_dir.glob("*.pdf")):
            pdf_id = f"{folder_name}/{pdf_path.name}"
            cached = self._pdf_cache.get(pdf_id)
            # Treat an empty list as cached so we do not keep reprocessing
            # previously failed extractions on every run.
            if cached is not None:
                for sentence in cached:
                    normalized = normalize_text(sentence)
                    if not normalized or has_unknown_private_glyphs(normalized):
                        continue
                    collected.append((normalized, pdf_path.name))
                    self.stats["monolingual_sources"].setdefault(pdf_path.stem, 0)
                    self.stats["monolingual_sources"][pdf_path.stem] += 1
                continue
            if (
                self.max_total_pdfs is not None
                and self._pdf_counter >= self.max_total_pdfs
            ):
                continue
            if (
                self.max_pdfs_per_folder is not None
                and processed_in_folder >= self.max_pdfs_per_folder
            ):
                continue
            raw_text = self._extract_pdf_text(pdf_path)
            if not raw_text:
                self._pdf_cache[pdf_id] = []
                processed_in_folder += 1
                self._pdf_counter += 1
                continue
            cleaned_lines = self._clean_pdf_text(raw_text)
            if not cleaned_lines:
                self._pdf_cache[pdf_id] = []
                processed_in_folder += 1
                self._pdf_counter += 1
                continue
            self._pdf_cache[pdf_id] = cleaned_lines
            for sentence in cleaned_lines:
                collected.append((sentence, pdf_path.name))
                self.stats["monolingual_sources"].setdefault(pdf_path.stem, 0)
                self.stats["monolingual_sources"][pdf_path.stem] += 1
            processed_in_folder += 1
            self._pdf_counter += 1
            new_pdfs += 1
        if (
            processed_in_folder
            and self.max_pdfs_per_folder is not None
            and processed_in_folder >= self.max_pdfs_per_folder
        ):
            print(
                f"[build_local_dataset] Reached per-folder PDF limit ({self.max_pdfs_per_folder}) for {folder_name}"
            )
        if new_pdfs:
            print(
                f"[build_local_dataset] Added {new_pdfs} new PDFs from {folder_name}; total cached PDFs: {len(self._pdf_cache)}"
            )
        return collected

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        if pdfplumber is None and PdfReader is None:
            return ""
        text_parts: List[str] = []
        if pdfplumber is not None:
            logging.getLogger("pdfminer").setLevel(logging.ERROR)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages[: self.max_pdf_pages]:
                        try:
                            page_text = page.extract_text() or ""
                        except Exception:
                            page_text = ""
                        if page_text.strip():
                            text_parts.append(page_text)
            except Exception as exc:
                print(
                    f"[build_local_dataset] pdfplumber failed for {pdf_path.name}: {exc}; falling back to PyPDF2"
                )
                text_parts = []
        if not text_parts and PdfReader is not None:
            try:
                reader = PdfReader(str(pdf_path))
                for page in reader.pages[: self.max_pdf_pages]:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
            except Exception as exc:
                print(
                    f"[build_local_dataset] PyPDF2 failed for {pdf_path.name}: {exc}"
                )
                return ""
        return "\n".join(text_parts)

    def _clean_pdf_text(self, text: str) -> List[str]:
        results: List[str] = []
        if not text:
            return results
        lines = [normalize_text(line) for line in text.split("\n")]
        for line in lines:
            if len(line) < 25:
                continue
            if has_unknown_private_glyphs(line):
                continue
            if korean_ratio(line) < 0.6:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            # split again into sentences because PDF lines are long.
            for sent in re.split(r"[.!?\u3002]+", line):
                cleaned = normalize_text(sent)
                if len(cleaned) < 40:
                    continue
                if has_unknown_private_glyphs(cleaned):
                    continue
                if korean_ratio(cleaned) < 0.6:
                    continue
                results.append(cleaned)
        return results[: 2000]

    def save_monolingual(self, sentences: Sequence[Tuple[str, str]]) -> None:
        total = len(sentences)
        valid_count = max(1, int(total * self.valid_ratio))
        train = sentences[:-valid_count]
        valid = sentences[-valid_count:]
        self.monolingual_dir.mkdir(parents=True, exist_ok=True)

        def dump(path: Path, subset: Sequence[Tuple[str, str]]) -> None:
            with path.open("w", encoding="utf-8") as fh:
                for text, _ in subset:
                    fh.write(text + "\n")

        dump(self.monolingual_dir / "train_nk.txt", train)
        dump(self.monolingual_dir / "valid_nk.txt", valid)
        self._write_sample(self.samples_dir / "monolingual_train_head.txt", [t for t, _ in train[:25]])
        self.stats.setdefault("counts", {})["monolingual_total"] = total
        self.stats["counts"]["monolingual_train"] = len(train)
        self.stats["counts"]["monolingual_valid"] = len(valid)

    # ------------------------------------------------------------------
    # Dictionaries
    # ------------------------------------------------------------------
    def load_dictionary_entries(self) -> List[Tuple[str, str, str]]:
        dict_dir = self.resources_dir / "Dictionaries"
        entries: List[Tuple[str, str, str]] = []
        if not dict_dir.exists():
            return entries
        for csv_path in sorted(dict_dir.glob("*.csv")):
            for row in iter_csv_rows(csv_path):
                fallback = next(iter(row.values())) if row else ""
                word = normalize_text(row.get("word") or row.get("entry") or fallback)
                definition = normalize_text(row.get("definition") or row.get("meaning") or "")
                if not word or not definition:
                    continue
                if has_unknown_private_glyphs(word) or has_unknown_private_glyphs(definition):
                    continue
                if not has_korean(word) and not has_korean(definition):
                    continue
                entries.append((word, "definition", definition[:400]))
                self.stats["dictionary_sources"].setdefault(csv_path.stem, 0)
                self.stats["dictionary_sources"][csv_path.stem] += 1
                if len(entries) >= self.max_dictionary_entries:
                    return entries
        return entries

    def save_dictionary(self, entries: Sequence[Tuple[str, str, str]]) -> None:
        limit = int(self.dictionary_fraction * max(1, self.stats.get("counts", {}).get("monolingual_total", len(entries))))
        limit = min(limit or len(entries), len(entries))
        trimmed = entries[:limit]
        self.dictionary_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.dictionary_dir / "nk_lexicon.tsv"
        with out_path.open("w", encoding="utf-8") as fh:
            for word, category, definition in trimmed:
                fh.write(f"{word}\t{category}\t{definition}\n")
        self._write_sample(self.samples_dir / "dictionary_head.tsv", ["\t".join(entry) for entry in trimmed[:25]])
        self.stats.setdefault("counts", {})["dictionary_total"] = len(trimmed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _write_sample(self, path: Path, lines: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    def write_stats(self) -> None:
        stats_path = self.output_dir / "stats.json"
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(self.stats, fh, ensure_ascii=False, indent=2)
        if self.pdf_cache_file:
            try:
                self.pdf_cache_file.parent.mkdir(parents=True, exist_ok=True)
                self.pdf_cache_file.write_text(
                    json.dumps(self._pdf_cache, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"[build_local_dataset] Warning: failed to write PDF cache: {exc}")

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        parallel_pairs = self.load_parallel_pairs()
        if parallel_pairs:
            self.save_parallel_splits(parallel_pairs)
        monolingual = self.load_monolingual_sentences()
        if monolingual:
            self.save_monolingual(monolingual)
        dictionary_entries = self.load_dictionary_entries()
        if dictionary_entries:
            self.save_dictionary(dictionary_entries)
        self.write_stats()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned local datasets")
    parser.add_argument("--resources", default="Resources", help="Path to Resources directory")
    parser.add_argument("--output", default="data/local", help="Output directory")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_pdf_pages", type=int, default=40)
    parser.add_argument("--max_dictionary_entries", type=int, default=20000)
    parser.add_argument("--dictionary_fraction", type=float, default=0.08)
    parser.add_argument("--max_pdfs_per_folder", type=int, default=None, help="Limit PDFs per folder to avoid long runs")
    parser.add_argument("--max_total_pdfs", type=int, default=None, help="Global PDF limit across folders")
    parser.add_argument("--pdf_cache_file", type=Path, default=None, help="Path to cache JSON storing processed PDF sentences")
    parser.add_argument("--skip_pdfs", action="store_true", help="Skip PDF extraction (much faster)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    builder = LocalDatasetBuilder(
        resources_dir=Path(args.resources),
        output_dir=Path(args.output),
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        max_pdf_pages=args.max_pdf_pages,
        max_dictionary_entries=args.max_dictionary_entries,
        dictionary_fraction=args.dictionary_fraction,
        process_pdfs=not args.skip_pdfs,
        max_pdfs_per_folder=args.max_pdfs_per_folder,
        max_total_pdfs=args.max_total_pdfs,
        pdf_cache_file=args.pdf_cache_file,
    )
    builder.run()
    print(f"✅ Local dataset built under {builder.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
