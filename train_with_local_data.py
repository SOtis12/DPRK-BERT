#!/usr/bin/env python3
"""
Train DPRK-BERT with local data sources only
No web scraping - uses only the resources you've already gathered
Includes checkpointing to resume from interruptions
"""

import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import re
import hashlib

# PDF processing libraries
try:
    import fitz  # PyMuPDF - better for Korean text extraction
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF libraries not available. Install with: pip install PyPDF2 PyMuPDF")

# Paths
PROJECT_ROOT = Path(__file__).parent
RESOURCES = PROJECT_ROOT / "Resources"
DPRK_BERT_DIR = PROJECT_ROOT / "DPRK-BERT-master"
DPRK_BERT_PUBLIC = PROJECT_ROOT / "DPRK-BERT-Public"
CHECKPOINT_DIR = PROJECT_ROOT / "data_processing_checkpoints"
PDF_CACHE_PATH = PROJECT_ROOT / "data/local/pdf_cache.json"

PARALLEL_NK_KEYS = ("north_korean", "north", "nk", "nk_sentence", "north_text")
PARALLEL_SK_KEYS = ("south_korean", "south", "sk", "sk_sentence", "south_text")


def _extract_field(row: Dict[str, str], candidate_keys: Iterable[str]) -> Optional[str]:
    """Return the first non-empty, stripped value for any of the candidate keys."""
    for key in candidate_keys:
        if key in row:
            raw_value = row[key]
            if raw_value is None:
                continue
            value = raw_value.strip()
            if value:
                return value
    return None

def log(msg):
    print(f"[INFO] {msg}")

def get_checkpoint_path(source_name: str) -> Path:
    """Get checkpoint file path for a data source"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{source_name}.json"

def save_checkpoint(source_name: str, data: List[Dict[str, Any]]):
    """Save processed data to checkpoint"""
    checkpoint_path = get_checkpoint_path(source_name)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"data": data, "count": len(data)}, f, ensure_ascii=False, indent=2)
    log(f"✓ Checkpoint saved: {source_name} ({len(data)} items)")

def load_checkpoint(source_name: str) -> List[Dict[str, Any]]:
    """Load data from checkpoint if it exists"""
    checkpoint_path = get_checkpoint_path(source_name)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data.get("data", [])
                log(f"✓ Loaded from checkpoint: {source_name} ({len(items)} items)")
                return items
        except Exception as e:
            log(f"Warning: Failed to load checkpoint {source_name}: {e}")
    return None

def process_speeches() -> List[Dict[str, Any]]:
    """Process Kim's New Year Speeches"""
    source_name = "speeches"
    
    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached
    
    log("Processing Kim's New Year Speeches...")
    speeches_dir = RESOURCES / "Kim's New Years Speeches"
    
    if not speeches_dir.exists():
        log(f"Warning: {speeches_dir} not found")
        return []
    
    items = []
    for txt_file in speeches_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if len(content) > 100:
                    items.append({
                        "text": content,
                        "source": f"kim_speech_{txt_file.stem}",
                        "type": "political_speech"
                    })
        except Exception as e:
            log(f"Error reading {txt_file}: {e}")
    
    log(f"Found {len(items)} speeches")
    save_checkpoint(source_name, items)
    return items

def process_parallel_data() -> List[Dict[str, Any]]:
    """Process parallel translation pairs"""
    source_name = "parallel"
    
    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached
    
    log("Processing parallel translation data...")
    parallel_dir = RESOURCES / "Parallel Boost"
    
    if not parallel_dir.exists():
        log(f"Warning: {parallel_dir} not found")
        return []
    
    items = []
    pair_count = 0
    for csv_file in parallel_dir.glob("*.csv"):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    north_text = _extract_field(row, PARALLEL_NK_KEYS)
                    south_text = _extract_field(row, PARALLEL_SK_KEYS)

                    if not north_text and not south_text:
                        continue

                    pair_count += 1
                    if north_text:
                        items.append({
                            "text": north_text,
                            "source": f"parallel_{csv_file.stem}",
                            "type": "north_korean_parallel"
                        })
                    if south_text:
                        items.append({
                            "text": south_text,
                            "source": f"parallel_{csv_file.stem}",
                            "type": "south_korean_parallel"
                        })
        except Exception as e:
            log(f"Error reading {csv_file}: {e}")
    
    log(f"Found {pair_count} parallel pairs ({len(items)} sentences)")
    save_checkpoint(source_name, items)
    return items

def process_dictionaries(max_entries: int = 10000) -> List[Dict[str, Any]]:
    """Process dictionary data"""
    source_name = f"dictionaries_{max_entries}"
    
    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached
    
    log(f"Processing dictionaries (max {max_entries} entries)...")
    dict_dir = RESOURCES / "Dictionaries"
    
    if not dict_dir.exists():
        log(f"Warning: {dict_dir} not found")
        return []
    
    items = []
    total_added = 0
    for csv_file in dict_dir.glob("*.csv"):
        if total_added >= max_entries:
            break
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if total_added >= max_entries:
                        break

                    # Extract Korean text from dictionary entries
                    text_parts = []
                    for key in row.keys():
                        raw_val = row[key]
                        if raw_val is None:
                            continue
                        val = raw_val.strip()
                        if val and re.search(r'[가-힣]', val):
                            text_parts.append(val)

                    if not text_parts:
                        continue

                    combined = " ".join(text_parts)
                    if len(combined) <= 20:
                        continue

                    items.append({
                        "text": combined,
                        "source": f"dictionary_{csv_file.stem}",
                        "type": "dictionary_entry"
                    })
                    total_added += 1
        except Exception as e:
            log(f"Error reading {csv_file}: {e}")
    
    log(f"Found {len(items)} dictionary entries (capped at {max_entries})")
    save_checkpoint(source_name, items)
    return items

def process_gyeoremal() -> List[Dict[str, Any]]:
    """Process gyeoremal dialect comparison data"""
    source_name = "gyeoremal"
    
    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached
    
    log("Processing gyeoremal dialect data...")
    gyeoremal_dir = RESOURCES / "gyeoremal"
    
    if not gyeoremal_dir.exists():
        log(f"Warning: {gyeoremal_dir} not found")
        return []
    
    items = []
    for csv_file in gyeoremal_dir.glob("*.csv"):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add both dialects with context
                    for key in row.keys():
                        val = row[key].strip()
                        if val and re.search(r'[가-힣]', val) and len(val) > 10:
                            items.append({
                                "text": val,
                                "source": f"gyeoremal_{csv_file.stem}",
                                "type": "dialect_comparison"
                            })
        except Exception as e:
            log(f"Error reading {csv_file}: {e}")
    
    log(f"Found {len(items)} dialect items")
    save_checkpoint(source_name, items)
    return items

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF with PyPDF2 fallback"""
    if not PDF_AVAILABLE:
        return ""
    
    try:
        # Try PyMuPDF first (better for Korean)
        doc = fitz.open(str(pdf_path))
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        full_text = "\n".join(text_parts)
        
        # Fallback to PyPDF2 if PyMuPDF extracted nothing
        if not full_text.strip():
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
                full_text = "\n".join(text_parts)
        
        return full_text
        
    except Exception as e:
        log(f"Error extracting text from {pdf_path.name}: {e}")
        return ""

def clean_pdf_text(text: str) -> str:
    """Clean PDF text by removing page numbers, headers, footers, and artifacts"""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that are just page numbers
        if re.match(r'^\d+$', line):
            continue
        
        # Skip common page number patterns
        if re.match(r'^[- ]*\d+[- ]*$', line):
            continue
        
        # Skip very short lines (likely artifacts)
        if len(line) < 5:
            continue
        
        # Skip lines with mostly non-Korean characters (headers/footers)
        korean_chars = len(re.findall(r'[가-힣]', line))
        total_chars = len(re.sub(r'\s', '', line))
        if total_chars > 0 and korean_chars / total_chars < 0.3:
            continue
        
        cleaned_lines.append(line)
    
    # Join and normalize whitespace
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()


def load_pdf_cache() -> Optional[Dict[str, List[str]]]:
    """Return cached PDF sentences from data/local/pdf_cache.json if present."""
    if not PDF_CACHE_PATH.exists():
        return None
    try:
        with open(PDF_CACHE_PATH, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        log(f"Warning: Failed to load PDF cache {PDF_CACHE_PATH}: {exc}")
    return None


def append_pdf_chunks(
    items: List[Dict[str, Any]],
    text: str,
    folder_type: str,
    pdf_stem: str,
    sentences: Optional[List[str]] = None,
) -> None:
    """Chunk PDF text similar to the original OCR-based pipeline."""
    if not text:
        return
    max_chunk_size = 2000
    if len(text) <= max_chunk_size and (sentences is None or len(sentences) <= 1):
        items.append(
            {
                "text": text,
                "source": f"pdf_{folder_type}_{pdf_stem}",
                "type": f"pdf_{folder_type}",
            }
        )
        return

    if sentences is None:
        sentences = re.split(r'[.!?]\s+', text)
    current_chunk: List[str] = []
    current_length = 0
    chunk_index = 0
    for sentence in sentences:
        if not sentence:
            continue
        if current_length + len(sentence) > max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 100:
                items.append(
                    {
                        "text": chunk_text,
                        "source": f"pdf_{folder_type}_{pdf_stem}",
                        "type": f"pdf_{folder_type}",
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) > 100:
            items.append(
                {
                    "text": chunk_text,
                    "source": f"pdf_{folder_type}_{pdf_stem}",
                    "type": f"pdf_{folder_type}",
                    "chunk_index": chunk_index,
                }
            )

def process_pdfs() -> List[Dict[str, Any]]:
    """Process PDFs from both With The Century and PDFs folders"""
    source_name = "pdfs"

    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached

    pdf_cache = load_pdf_cache()
    if pdf_cache:
        log(f"Processing PDFs via cached sentences ({len(pdf_cache)} entries in {PDF_CACHE_PATH})")
        items: List[Dict[str, Any]] = []
        for pdf_id, sentences in pdf_cache.items():
            if not sentences:
                continue
            folder_prefix = pdf_id.split('/', 1)[0]
            pdf_name = pdf_id.split('/', 1)[1] if '/' in pdf_id else pdf_id
            folder_type = "with_century" if "century" in folder_prefix.lower() else "regulations"
            text = ' '.join(sentences)
            if len(text) < 100:
                continue
            korean_chars = len(re.findall(r'[가-힣]', text))
            total_chars = len(re.sub(r'\s', '', text))
            if total_chars == 0 or korean_chars / max(total_chars, 1) < 0.5:
                continue
            append_pdf_chunks(items, text, folder_type, Path(pdf_name).stem, sentences)
        log(f"Found {len(items)} text chunks from cached PDFs")
        save_checkpoint(source_name, items)
        return items

    if not PDF_AVAILABLE:
        log("Warning: PDF libraries not installed. Skipping PDF processing.")
        log("Install with: pip install PyPDF2 PyMuPDF")
        return []

    log("Processing PDFs...")
    
    items = []
    pdf_folders = [
        (RESOURCES / "With The Century", "with_century"),
        (RESOURCES / "PDFs", "regulations")
    ]
    
    for pdf_dir, folder_type in pdf_folders:
        if not pdf_dir.exists():
            log(f"Warning: {pdf_dir} not found")
            continue
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        log(f"Found {len(pdf_files)} PDFs in {pdf_dir.name}")
        
        for pdf_file in pdf_files:
            try:
                # Extract text
                raw_text = extract_pdf_text(pdf_file)
                if not raw_text:
                    log(f"No text extracted from {pdf_file.name}")
                    continue
                
                # Clean the text
                cleaned_text = clean_pdf_text(raw_text)
                
                # Validate minimum Korean content
                if len(cleaned_text) < 100:
                    log(f"Text too short after cleaning: {pdf_file.name}")
                    continue
                
                korean_chars = len(re.findall(r'[가-힣]', cleaned_text))
                total_chars = len(re.sub(r'\s', '', cleaned_text))
                
                if total_chars == 0 or korean_chars / total_chars < 0.5:
                    log(f"Insufficient Korean content: {pdf_file.name}")
                    continue
                
                append_pdf_chunks(items, cleaned_text, folder_type, pdf_file.stem)
                
            except Exception as e:
                log(f"Error processing {pdf_file.name}: {e}")
    
    log(f"Found {len(items)} text chunks from PDFs")
    save_checkpoint(source_name, items)
    return items

def process_rodong() -> List[Dict[str, Any]]:
    """Process Rodong Sinmun training data from DPRK-BERT-Public"""
    source_name = "rodong"
    
    # Check for existing checkpoint
    cached = load_checkpoint(source_name)
    if cached is not None:
        return cached
    
    rodong_file = DPRK_BERT_PUBLIC / "rodong_train.json"
    
    if not rodong_file.exists():
        log(f"Warning: Rodong data not found at {rodong_file}")
        return []
    
    items = []
    
    try:
        with open(rodong_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Data format: {"data": [{"data": "text", "date": "...", "title": "..."}, ...]}
        rodong_items = data.get('data', [])
        
        for idx, item in enumerate(rodong_items):
            text = item.get('data', '').strip()
            if not text or len(text) < 50:
                continue
            
            items.append({
                "text": text,
                "source": "rodong",
                "type": "newspaper",
                "date": item.get('date', ''),
                "title": item.get('title', ''),
                "index": idx
            })
        
        log(f"Processed {len(items)} Rodong Sinmun articles")
    
    except Exception as e:
        log(f"Error processing Rodong data: {e}")
        return []
    
    save_checkpoint(source_name, items)
    return items

def create_training_dataset():
    """Combine all local data sources into training dataset"""
    log("=" * 70)
    log("Creating training dataset from local sources only")
    log("=" * 70)
    
    all_data = []
    
    # Process each data source
    all_data.extend(process_speeches())
    all_data.extend(process_parallel_data())
    all_data.extend(process_dictionaries(max_entries=10000))
    all_data.extend(process_gyeoremal())
    all_data.extend(process_pdfs())
    all_data.extend(process_rodong())  # Add Rodong Sinmun data
    
    log(f"\nTotal items collected: {len(all_data)}")

    if len(all_data) == 0:
        log("ERROR: No data found!")
        return False

    for record in all_data:
        if "data" not in record and "text" in record:
            record["data"] = record["text"]

    # Split into train/validation (90/10)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Create output directory
    output_dir = PROJECT_ROOT / "local_training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in DPRK-BERT format
    train_file = output_dir / "train.json"
    val_file = output_dir / "validation.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump({"data": train_data}, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump({"data": val_data}, f, ensure_ascii=False, indent=2)
    
    log(f"\n✅ Training data saved:")
    log(f"   Train: {train_file} ({len(train_data)} items)")
    log(f"   Validation: {val_file} ({len(val_data)} items)")
    
    # Print statistics
    log("\n📊 Data sources breakdown:")
    sources = {}
    for item in all_data:
        src = item.get('source', 'unknown').split('_')[0]
        sources[src] = sources.get(src, 0) + 1
    
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        log(f"   {src}: {count} items")
    
    return True

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("🇰🇵 DPRK-BERT Training with Local Data Only")
    print("=" * 70)
    print("\nSources used:")
    print("  ✅ Kim's New Year Speeches")
    print("  ✅ Parallel translation pairs (NK/SK)")
    print("  ✅ NK dictionaries (10,000 entries)")
    print("  ✅ Gyeoremal dialect comparisons")
    print("  ✅ PDFs (With The Century, regulations)" if PDF_AVAILABLE else "  ⚠️  PDFs (libraries not installed)")
    print("  ✅ Rodong Sinmun articles (original DPRK-BERT data)")
    print("\nSources EXCLUDED:")
    print("  ❌ Web scraping (removed)")
    print("\nCheckpointing:")
    print(f"  📁 {CHECKPOINT_DIR}")
    print("  💾 Previously processed data will be reused")
    print("  🔄 Delete checkpoint files to reprocess from scratch")
    print("=" * 70)
    print()
    
    if not create_training_dataset():
        return 1
    
    print("\n" + "=" * 70)
    print("✅ Dataset ready!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. cd {DPRK_BERT_DIR}")
    print("2. python mlm_trainer.py \\")
    print("     --mode train \\")
    print("     --train_file ../local_training_data/train.json \\")
    print("     --validation_file ../local_training_data/validation.json \\")
    print("     --num_train_epochs 10")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
