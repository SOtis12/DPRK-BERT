#!/usr/bin/env python3
"""
Convert final_bert_training_dataset.json into a cleaned line-per-text file suitable
for masked language model pretraining. Adds options to deduplicate, filter, and
down-sample dominant sources like dictionary entries.

Usage:
python scripts/convert_final_dataset_to_mlm.py \
    --input ../final_bert_training_dataset.json \
    --output ../enhanced_training/final_bert_training_dataset_lines.txt \
    --min-length 10 --max-length 512 --max-dictionary 500000

"""
import argparse
import json
import random
from pathlib import Path


def stream_json_array(file_path):
    """Read a top-level JSON array and yield items one by one.
    Uses a simple incremental parser that avoids loading the entire array.
    """
    # The file is a list e.g. [ {..}, {..}, ... ] on disk. We'll parse roughly.
    with open(file_path, 'r', encoding='utf-8') as f:
        buf = ''
        # Skip initial whitespace and leading '['
        while True:
            c = f.read(1)
            if not c:
                break
            if c == '[':
                break
        depth = 0
        current = ''
        in_string = False
        escape = False
        while True:
            c = f.read(1)
            if not c:
                break
            if c == '"' and not escape:
                in_string = not in_string
            if c == '{' and not in_string:
                depth += 1
            if depth > 0:
                current += c
            if c == '}' and not in_string:
                depth -= 1
                if depth == 0:
                    # we have a complete object
                    try:
                        obj = json.loads(current)
                        yield obj
                    except Exception as e:
                        # If parse fails, try to skip
                        # This is a best-effort streaming parser
                        pass
                    current = ''
            if c == '\n':
                escape = False
            elif c == '\\':
                escape = not escape
            else:
                escape = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to final_bert_training_dataset.json')
    parser.add_argument('--output', required=True, help='Path for the output line-per-text file')
    parser.add_argument('--min-length', type=int, default=6, help='Minimum length of text to keep')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length of text to keep')
    parser.add_argument('--dedupe', action='store_true', help='Deduplicate texts (keep first seen)')
    parser.add_argument('--max-dictionary', type=int, default=500000, help='Maximum dictionary entries to keep (down-sample)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle outputs before writing (requires extra memory)')
    parser.add_argument('--seed', type=int, default=42, help='Seed used for sampling/shuffle')
    parser.add_argument('--max-processed', type=int, default=0, help='Maximum number of items to process (0 = all)')

    args = parser.parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_texts = set() if args.dedupe else None
    dictionary_kept = 0

    # best-effort streaming: iterate items from JSON
    items_processed = 0
    items_written = 0
    buffer = []

    for obj in stream_json_array(str(input_path)):
        items_processed += 1
        text = obj.get('text', '')
        if not isinstance(text, str):
            continue
        text = text.strip()
        L = len(text)
        if L < args.min_length or L > args.max_length:
            continue

        data_type = obj.get('data_type', 'unknown')

        if data_type == 'dictionary_entry':
            if dictionary_kept >= args.max_dictionary:
                # skip further dictionary entries with probability 0.99
                if random.random() > float(args.max_dictionary) / max(1, 3_500_000):
                    continue
            else:
                dictionary_kept += 1

        if seen_texts is not None:
            if text in seen_texts:
                continue
            seen_texts.add(text)

        if args.shuffle:
            buffer.append(text)
        else:
            output_path.write_text((text + "\n"), encoding='utf-8', append=False) if False else None
            # We'll use write in append mode for streaming
            with open(output_path, 'a', encoding='utf-8') as fw:
                fw.write(text + "\n")
            items_written += 1

        if items_processed % 100000 == 0:
            print(f"Processed: {items_processed:,}; Written: {items_written:,}; Dictionary kept: {dictionary_kept:,}")
        if args.max_processed and items_processed >= args.max_processed:
            break

    # If shuffle and buffer is used, write out
    if args.shuffle:
        random.shuffle(buffer)
        with open(output_path, 'w', encoding='utf-8') as fw:
            for t in buffer:
                fw.write(t + "\n")
        items_written = len(buffer)

    print(f"Done. Processed {items_processed:,} items; Wrote {items_written:,} lines; Dictionary kept: {dictionary_kept:,}")


if __name__ == '__main__':
    main()
