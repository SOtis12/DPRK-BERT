# DPRK-BERT Enhancement – LOCAL TRAINING FIRST

Repository: fork of `DPRK-BERT` (original code in `DPRK-BERT-master/`).

Goal: **Improve DPRK-BERT’s NK↔SK translation quality** by fine‑tuning on additional North Korean dialect resources, **without discarding or overwriting the existing Rodong‑trained weights**.

The automation is driven by `Codex Long Runner / Agent Loop`. It must work in **small, testable increments**, always logging to `runs/SUMMARY.md` and **never using the TPU until local validation passes.**

---

## 0. Hard Guardrails (Non‑Negotiable)

1. **TPU Usage — Controlled**
   - Default assumption: **TPU is OFF and MUST NOT be used**.
   - TPU usage is permitted only after strict local validation has completed and been documented.
   - Do **not** start or use any TPU for training **until ALL of the following are true**:
     1. Local data preparation is complete and validated (see Phase 1).
     2. Local fine‑tuning on Mac with `DPRK-BERT-master/mlm_trainer.py` has completed successfully on a **small subset** of the new data.
     3. Evaluation shows **no degradation** on:
        - Original Rodong test set.
        - A held‑out NK↔SK parallel test set.
     4. `runs/SUMMARY.md` clearly states that **local validation passed**.
   - When TPU usage begins:
     - Use at most **one** well‑specified training run.
     - After training, explicitly note in `runs/SUMMARY.md` that the TPU must be **stopped**, and record the command:
       - `gcloud alpha compute tpus tpu-vm stop dprk-bert-v5p --zone=us-central1-a`
   - **Never** launch additional or experimental TPU jobs from this automation.

2. **Local‑First Principle**
   - All new scripts, data pipelines, and training configs must:
     - Run on local machine (Mac) first.
     - Be tested on **small slices** of data before scaling up.
   - If something cannot run locally, it must **not** be run on TPU.

3. **Data Quality & Safety**
   - Do **not** add any data to training that:
     - Is mis‑parsed (e.g., PDF garbage, repeated headers/footers, mixed line fragments).
     - Is mostly non‑Korean, or dominated by page numbers, figure labels, etc.
   - Prefer **less but high‑quality** data over noisy bulk data.
   - Every phase that touches data must include **sanity checks and small sample inspection**.

4. **Progress Logging**
   - After each concrete change:
     - Append a **short entry** to `runs/SUMMARY.md`:
       - What changed (files/scripts).
       - How it was tested.
       - Next 2–3 planned steps.
     - Never delete unrelated files.

---

## 1. Data Sources & Priority

We only use **local resources**—no web scraping in this phase.

**Priority 1 – Parallel / Contrastive**
1. `Resources/Parallel Boost/`
   - NK/SK sentence pairs in CSV.
   - These are **most valuable** for translation: use as **bidirectional** examples (NK→SK, SK→NK).
2. `Resources/gyeoremal/`
   - CSVs with NK/SK dialect comparisons and example sentences.
   - Extract:
     - Parallel pairs (NK vs SK).
     - Notation/orthography differences for contrastive examples.

**Priority 2 – Monolingual NK**
3. `Resources/Kim's New Years Speeches/` (2013–2019 `.txt`)
   - Clean, high‑quality DPRK political text.
4. `Resources/With The Century/`
   - PDFs in NK dialect.
   - Need thorough PDF clean‑up.
5. `Resources/Dictionaries/`
   - NK phone dictionary data.
   - Use **sparingly**; don’t let dictionary entries dominate.
6. `Resources/PDFs/`
   - NK regulations and documents.
   - Some PDFs have English; only use Korean segments.

**Low Priority (Skip for Now)**
7. `Custom Code/` web scraping
   - **Do not use** scraping for this phase.
   - Ignore fetch/scrape scripts unless explicitly re‑enabled later.

---

## 2. Target Artifacts

The automation should produce, in a new `data/local/` tree (or similar):

1. **Parallel corpora (most important)**
   - `data/local/parallel/train.nk-sk.tsv`
   - `data/local/parallel/valid.nk-sk.tsv`
   - `data/local/parallel/test.nk-sk.tsv`
   - Tab‑separated: `nk_sentence<TAB>sk_sentence`
   - Only high‑confidence pairs from:
     - `Parallel Boost`
     - `gyeoremal`

2. **Monolingual NK training text**
   - `data/local/monolingual/train_nk.txt`
   - `data/local/monolingual/valid_nk.txt`
   - One cleaned sentence per line (UTF‑8).

3. **Optionally: dictionary‑style entries**
   - `data/local/dictionaries/nk_lexicon.tsv`
   - Format: `nk_form<TAB>category/pos<TAB>optional_metadata`
   - Will be used in **small proportion** to avoid over‑fitting to dictionary style.

4. **Documented stats**
   - `data/local/stats.json` (or similar) containing:
     - Counts of lines/sentences per source.
     - Percentage of lines filtered out.
     - Character distribution (ratio of Hangul vs Latin).

5. **Validation samples**
   - Small human‑inspectable samples:
     - `data/local/samples/parallel_head.tsv`
     - `data/local/samples/monolingual_head.txt`

---

## 3. Phase 1 – Local Data Preparation (Current Focus)

### 3.1 General Cleaning Rules

For **every** source:

- Encode as **UTF‑8**.
- Only keep lines / segments that:
  - Contain Korean Hangul characters (`\uAC00-\uD7AF`).
  - Aren’t dominated by ASCII/Latin (e.g., drop if >50–70% non‑Hangul).
- Remove:
  - Page numbers, headers/footers (common patterns: numbers alone, “Page X”, repeated boilerplate).
  - Line artifacts like hyphenated word breaks from PDF extraction.
  - Extra whitespace, tabs, and repeated punctuation.
- Deduplicate:
  - Within each file.
  - Optionally across the global corpus (e.g., by hashing normalized lines).

Record all filters applied and counts per step.

---

### 3.2 Parallel Boost (`Resources/Parallel Boost/`)

**Goal:** Extract clean NK/SK sentence pairs.

Tasks:
1. Write or refine a loader that:
   - Reads all CSVs in `Resources/Parallel Boost/`.
   - Identifies columns for NK and SK sentences (document this mapping in code comments).
   - Normalizes whitespace and punctuation.
2. Filter:
   - Drop rows where either side is empty, extremely short (e.g. < 3 Hangul characters), or mostly non‑Korean.
   - Remove duplicate pairs.
3. Save:
   - Clean TSV (`nk<TAB>sk`) into `data/local/parallel/*.tsv`.
   - Split into train/valid/test using a fixed random seed.
4. Sample check:
   - Manually inspect first ~50 lines of each split and record impressions in `runs/SUMMARY.md`.

---

### 3.3 Gyeoremal (`Resources/gyeoremal/`)

**Goal:** Use as both parallel and contrastive data.

Tasks:
1. Parse CSVs to find:
   - NK and SK sentence columns.
   - Any explicit notation/orthography differences.
2. Extract:
   - High‑confidence NK/SK sentence pairs and append to the **parallel corpus**.
   - Contrastive examples:
     - `(nk_sentence, sk_sentence, note_about_difference)`
     - Optionally save to `data/local/contrastive/gyeoremal.tsv` for future use.
3. Apply **same cleaning rules** as Parallel Boost.
4. Ensure that `gyeoremal` pairs don’t overwhelm `Parallel Boost`:
   - Record source proportions in `stats.json`.

---

### 3.4 New Year Speeches (`Resources/Kim's New Years Speeches/`)

**Goal:** High‑quality monolingual NK text for MLM.

Tasks:
1. For each `.txt` file:
   - Normalize whitespace.
   - Split into sentences (simple heuristic is fine; avoid over‑segmenting).
2. Filter:
   - Remove lines that are clearly headings, dates, or annotation artifacts.
3. Append to `data/local/monolingual/train_nk.txt`, with a small portion reserved for validation.

---

### 3.5 With The Century PDFs (`Resources/With The Century/`)

**Goal:** Add substantial DPRK narrative text, carefully cleaned.

Tasks:
1. Use a PDF text extractor (e.g. `pdftotext`) to get raw text per PDF.
2. Cleaning:
   - Remove page numbers and running headers:
     - Typical patterns: lines containing only digits, or repeated phrases at top/bottom of pages.
   - Merge broken lines where PDF extraction splits sentences.
3. Sentence segmentation and filtering:
   - Extract sentences that contain enough Hangul and aren’t mostly numeric or Latin.
4. Append to monolingual NK corpus.

---

### 3.6 Other PDFs (`Resources/PDFs/`)

**Goal:** Add additional DPRK text without English contamination.

Tasks:
1. Extract text as above.
2. For each segment:
   - If a block is mostly English (Latin alphabet), drop it.
   - If bilingual (NK + English), **keep only the Korean part**.
3. Clean and append to monolingual corpus following the same rules.

---

### 3.7 Dictionaries (`Resources/Dictionaries/`)

**Goal:** Lightly augment with lexical coverage.

Tasks:
1. Parse raw dictionary data to produce structured entries:
   - `nk_form`, optional `sk_equivalent`, `pos`, etc.
2. Clean:
   - Remove non‑Korean or clearly broken entries.
3. Sampling:
   - Randomly sample a **small subset** to avoid dictionary over‑representation (e.g., at most 5–10% of total MLM training lines).
4. Save to `data/local/dictionaries/nk_lexicon.tsv`.

---

### 3.8 Data Balance & Final Checks

Tasks:
1. Compute corpus stats:
   - Number of lines from each source.
   - Overall size of monolingual vs parallel data.
2. Apply balancing:
   - Ensure no single monolingual source (e.g., a huge PDF) dwarfs the others.
   - Ensure dictionary entries remain a small fraction.
3. Produce:
   - `data/local/stats.json`
   - Sample files under `data/local/samples/`.
4. Log a summary in `runs/SUMMARY.md`:
   - Data sizes.
   - Key cleaning decisions.
   - Any suspicious patterns that were dropped.

---

## 4. Phase 2 – Local Training & Validation (Mac Only)

Script to use: `DPRK-BERT-master/mlm_trainer.py` plus your `train_with_local_data.py`.

### 4.1 Integration with DPRK-BERT Format

Tasks:
1. Confirm how `mlm_trainer.py` expects data (one sentence per line, masked LM format, etc.).
2. Adapt `train_with_local_data.py` so that it:
   - Reads from `data/local/…` artifacts.
   - Produces training/validation text files in the exact format `mlm_trainer.py` needs.
3. Run a **tiny smoke test** locally:
   - Subsample: e.g., 1k–5k lines total.
   - Run `mlm_trainer.py` for 1–2 epochs:
     - Verify it finishes.
     - Check that loss decreases.

Record results and loss trend in `runs/SUMMARY.md`.

---

### 4.2 Proper Local Fine‑Tuning

Tasks:
1. Use original DPRK-BERT Rodong‑trained checkpoint as the starting point.
2. Train locally (still on Mac) with:
   - Full monolingual NK corpus.
   - Optional up‑weighting of parallel data in the MLM objective (if supported).
3. Monitoring:
   - Track training and validation loss curves.
   - Save intermediate checkpoints safely.

---

### 4.3 Local Evaluation

Use:
- NK↔SK parallel test split from `Parallel Boost` + `gyeoremal`.
- Original Rodong test set from the original project (unchanged).

Tasks:
1. Run evaluation script(s) to compute:
   - MLM perplexity or equivalent metrics.
   - Simple translation quality checks using the parallel test set.
2. Ensure:
   - No degradation on Rodong benchmark vs original DPRK-BERT.
   - Measurable improvement on the NK↔SK parallel test set (even simple accuracy proxies).
3. Only if results are satisfactory:
   - Add a clear entry to `runs/SUMMARY.md`:
     - “Local validation PASSED” with metrics.
     - Document intention to run a single TPU fine‑tune run (no human authorization step required here; the run may proceed once the documented local validation criteria are met).

If validation fails, iterate locally (adjust data cleaning, weighting, or hyperparameters) **without touching TPU**.

---

## 5. Phase 3 – TPU Deployment (ONLY After Local Success)

Once `runs/SUMMARY.md` documents that local validation passed and the preconditions in Phase 2 are met:

Tasks:
1. Prepare:
   - Pack the validated dataset and config files.
   - Estimate the number of TPU training steps required.
2. Before starting TPU:
   - Confirm TPUs status:
     - It is acceptable to run:
       - `gcloud alpha compute tpus tpu-vm list --zone=us-central1-a`
       - **Read‑only** check; do not start/stop instances here without following the controlled deployment procedure.
3. Run a single fine‑tuning job on TPU:
   - Same code (`mlm_trainer.py`), same dataset, only larger batch/longer training.
   - Log configuration and commands used.
4. After training:
   - Download the trained model to local machine.
   - Validate again on:
     - Rodong test set.
     - NK↔SK parallel test set.
   - Add a detailed summary to `runs/SUMMARY.md`.
   - Ensure TPU is stopped with:
     - `gcloud alpha compute tpus tpu-vm stop dprk-bert-v5p --zone=us-central1-a`

---

## 6. Per‑Iteration Behavior for the Agent Loop

At each iteration (as driven by `LongRunner`):

1. Make a **small, verifiable change**:
   - One script improvement, one cleaning function, one evaluation script, or one experiment config at a time.
2. Run relevant checks:
   - For data steps: run small sampling/validation scripts.
   - For training steps: run tiny local smoke tests first.
3. Output:
   - Print filenames and diffs for any edits.
   - Do **not** invoke TPU training commands unless Phase 3 conditions have been met and documented in `runs/SUMMARY.md`.
4. Update `runs/SUMMARY.md`:
   - What was done this iteration.
   - Results/metrics.
   - Next 2–3 concrete actions.

Keep everything incremental, reproducible, and **local‑first** until local validation clearly passes.

