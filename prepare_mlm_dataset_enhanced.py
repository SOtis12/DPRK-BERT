"""
Enhanced version of prepare_mlm_dataset.py that handles all DPRK data sources:
- Original functionality (Rodong Sinmun, New Year speeches)
- NEW: PDF processing (With The Century, regulations)
- NEW: Dictionary processing (9.17M entries from NK phone)  
- NEW: CSV parallel data (parallel boost/seed files)
- NEW: Web scraped data integration
- NEW: Data quality filtering and deduplication

This extends the existing DPRK-BERT prepare_mlm_dataset.py with multi-source support.
"""

import os
import json
import csv
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import argparse
import config

# Import original functions
from prepare_mlm_dataset import prepare_newyear_data, get_all_day_data, data_store_field

# PDF processing
try:
    import fitz  # PyMuPDF - better for Korean text extraction
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import cleaner
from cleaner import Cleaner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_enhanced_args():
    """Enhanced argument parser for multi-source data processing"""
    parser = argparse.ArgumentParser(description="Enhanced MLM dataset preparation for DPRK-BERT")
    
    # Original arguments
    parser.add_argument(
        "--input_type",
        type=str,
        default="enhanced",
        choices=["rodong", "new_year", "enhanced"],
        help="Type of data for preparation",
    )
    parser.add_argument(
        "--apply_split",
        action="store_true",
        default=False,
        help="Whether to split the dataset into train/validation",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Split ratio when split is true",
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default="../Resources",
        help="Source folder containing all data",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="enhanced_mlm_training_data",
        help="Save folder for dataset",
    )
    
    # Enhanced arguments for multiple sources
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["speeches", "parallel", "dictionaries", "pdfs", "web"],
        help="Data sources to include: speeches, parallel, dictionaries, pdfs, web, rodong",
    )
    parser.add_argument(
        "--max_dict_entries",
        type=int,
        default=50000,
        help="Maximum dictionary entries to process (to avoid memory issues)",
    )
    parser.add_argument(
        "--scraped_folders",
        nargs="*",
        default=[],
        help="Paths to scraped data folders (bert_data)",
    )
    parser.add_argument(
        "--quality_filter",
        action="store_true",
        default=True,
        help="Apply quality filtering to data",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true", 
        default=True,
        help="Remove duplicate entries",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=20,
        help="Minimum text length for quality filtering",
    )
    parser.add_argument(
        "--max_text_length", 
        type=int,
        default=1000,
        help="Maximum text length for quality filtering",
    )
    
    args = parser.parse_args()
    return args

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (better for Korean)"""
    if not PDF_AVAILABLE:
        logger.error("PDF processing libraries not available")
        return ""
        
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        full_text = "\n".join(text_parts)
        
        # Fallback to PyPDF2 if no text extracted
        if not full_text.strip():
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(text)
                    full_text = "\n".join(text_parts)
            except:
                pass
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_pdf_folder(source_folder: Path, folder_name: str, save_folder: str, cleaner: Cleaner) -> List[Dict[str, Any]]:
    """Process all PDFs in a folder"""
    pdf_folder = source_folder / folder_name
    if not pdf_folder.exists():
        logger.warning(f"PDF folder not found: {pdf_folder}")
        return []
    
    pdf_files = list(pdf_folder.glob("*.pdf"))
    logger.info(f"Processing {len(pdf_files)} PDFs from {folder_name}")
    
    data = []
    for pdf_file in tqdm(pdf_files, desc=f"Processing {folder_name} PDFs"):
        try:
            text = extract_pdf_text(pdf_file)
            if text and len(text.strip()) > 100:
                cleaned_text = cleaner.clean(text)
                
                if is_valid_korean_text(cleaned_text):
                    # Split long texts into smaller chunks for BERT
                    chunks = split_text_into_chunks(cleaned_text, max_length=800)
                    
                    for i, chunk in enumerate(chunks):
                        if len(chunk.strip()) > 50:
                            doc_data = {
                                "id": f"{folder_name}_{pdf_file.stem}_chunk_{i}",
                                "data": chunk,
                                "source": folder_name.lower(),
                                "type": "pdf_document",
                                "filename": pdf_file.name,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                            data.append(doc_data)
                            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
    
    logger.info(f"Extracted {len(data)} chunks from {folder_name} PDFs")
    return data

def process_dictionary_files(source_folder: Path, max_entries: int, cleaner: Cleaner) -> List[Dict[str, Any]]:
    """Process dictionary CSV files"""
    dict_folder = source_folder / "Dictionaries"
    if not dict_folder.exists():
        logger.warning(f"Dictionary folder not found: {dict_folder}")
        return []
    
    csv_files = list(dict_folder.glob("*.csv"))
    logger.info(f"Processing {len(csv_files)} dictionary files")
    
    data = []
    total_processed = 0
    
    for csv_file in csv_files:
        if total_processed >= max_entries:
            logger.info(f"Reached max dictionary entries: {max_entries}")
            break
            
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # Try to read as CSV
                reader = csv.reader(f)
                header = next(reader, None)
                
                for row_num, row in enumerate(reader):
                    if total_processed >= max_entries:
                        break
                        
                    if len(row) >= 2:
                        term = row[0].strip()
                        definition = row[1].strip()
                        
                        if term and definition:
                            # Create training text from dictionary entry
                            if is_korean_text(definition):
                                text = f"{term}: {definition}"
                            else:
                                text = f"{term} = {definition}"
                            
                            cleaned_text = cleaner.clean(text)
                            if len(cleaned_text.strip()) > 10 and is_valid_korean_text(cleaned_text):
                                entry_data = {
                                    "id": f"dict_{csv_file.stem}_{row_num}",
                                    "data": cleaned_text,
                                    "source": "dictionary",
                                    "type": "dictionary_entry",
                                    "filename": csv_file.name
                                }
                                data.append(entry_data)
                                total_processed += 1
                                
        except Exception as e:
            logger.error(f"Error processing dictionary {csv_file}: {e}")
    
    logger.info(f"Processed {len(data)} dictionary entries")
    return data

def process_parallel_data(source_folder: Path, cleaner: Cleaner) -> List[Dict[str, Any]]:
    """Process parallel boost/seed CSV files"""
    parallel_folder = source_folder / "Parallel Boost"
    if not parallel_folder.exists():
        logger.warning(f"Parallel folder not found: {parallel_folder}")
        return []
    
    csv_files = list(parallel_folder.glob("*.csv"))
    logger.info(f"Processing {len(csv_files)} parallel files")
    
    data = []
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row_num, row in enumerate(reader):
                    try:
                        south_korean = row.get('south_korean', '').strip()
                        north_korean = row.get('north_korean', '').strip()
                        
                        if south_korean and north_korean:
                            # Add both variants as separate training examples
                            for text, variant in [(south_korean, 'south'), (north_korean, 'north')]:
                                cleaned_text = cleaner.clean(text)
                                if is_valid_korean_text(cleaned_text):
                                    parallel_data = {
                                        "id": f"parallel_{variant}_{csv_file.stem}_{row_num}",
                                        "data": cleaned_text,
                                        "source": "parallel",
                                        "type": f"parallel_{variant}",
                                        "filename": csv_file.name
                                    }
                                    data.append(parallel_data)
                                    
                    except Exception as e:
                        logger.warning(f"Error processing row {row_num} in {csv_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing parallel file {csv_file}: {e}")
    
    logger.info(f"Processed {len(data)} parallel entries")
    return data

def process_enhanced_speeches(source_folder: Path, save_folder: str, split: float, cleaner: Cleaner) -> List[Dict[str, Any]]:
    """Enhanced speech processing with better text chunking"""
    speeches_folder = source_folder / "Kim's New Years Speeches"
    if not speeches_folder.exists():
        logger.warning(f"Speeches folder not found: {speeches_folder}")
        return []
    
    txt_files = list(speeches_folder.glob("*.txt"))
    logger.info(f"Processing {len(txt_files)} speech files")
    
    data = []
    for txt_file in tqdm(txt_files, desc="Processing speeches"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            cleaned_text = cleaner.clean(text)
            if cleaned_text.strip():
                # Split speech into paragraphs for better training
                paragraphs = split_text_into_chunks(cleaned_text, max_length=500)
                year = txt_file.stem
                
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph.strip()) > 50:
                        speech_data = {
                            "id": f"speech_{year}_para_{i}",
                            "data": paragraph,
                            "source": "new_year_speech",
                            "type": "political_speech",
                            "year": int(year) if year.isdigit() else None,
                            "filename": txt_file.name,
                            "paragraph_index": i
                        }
                        data.append(speech_data)
                        
        except Exception as e:
            logger.error(f"Error processing speech {txt_file}: {e}")
    
    logger.info(f"Processed {len(data)} speech paragraphs")
    return data

def process_web_scraped_data(scraped_folders: List[str], cleaner: Cleaner) -> List[Dict[str, Any]]:
    """Process web scraped BERT data"""
    data = []
    
    for folder_path in scraped_folders:
        folder = Path(folder_path)
        bert_folder = folder / "bert_data"
        
        if bert_folder.exists():
            bert_files = list(bert_folder.glob("*_bert_mlm.txt"))
            logger.info(f"Processing {len(bert_files)} BERT files from {folder.name}")
            
            for bert_file in tqdm(bert_files, desc=f"Processing {folder.name}"):
                try:
                    with open(bert_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    if text.strip():
                        # Split into sentences
                        sentences = split_into_sentences(text)
                        
                        for i, sentence in enumerate(sentences):
                            cleaned_sentence = cleaner.clean(sentence)
                            if len(cleaned_sentence.strip()) > 20 and is_valid_korean_text(cleaned_sentence):
                                sentence_data = {
                                    "id": f"web_{bert_file.stem}_{i}",
                                    "data": cleaned_sentence,
                                    "source": "web_scraped",
                                    "type": "web_article",
                                    "scraper_run": folder.name,
                                    "original_file": bert_file.name
                                }
                                data.append(sentence_data)
                                
                except Exception as e:
                    logger.error(f"Error processing {bert_file}: {e}")
    
    logger.info(f"Processed {len(data)} web sentences")
    return data

# Utility functions
def is_korean_text(text: str) -> bool:
    """Check if text contains Korean characters"""
    return bool(re.search(r'[가-힣]', text))

def is_valid_korean_text(text: str) -> bool:
    """Check if text contains substantial Korean content"""
    if not text.strip():
        return False
    
    korean_chars = len(re.findall(r'[가-힣]', text))
    total_chars = len(re.sub(r'\s', '', text))
    
    if total_chars == 0:
        return False
    
    korean_ratio = korean_chars / total_chars
    return korean_ratio > 0.3 and korean_chars > 10

def split_text_into_chunks(text: str, max_length: int = 500) -> List[str]:
    """Split text into smaller chunks for BERT training"""
    # Split by sentences first
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_length and current_chunk:
            # Start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """Split Korean text into sentences"""
    sentence_endings = r'[.!?。！？]+'
    sentences = re.split(sentence_endings, text)
    
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:
            clean_sentences.append(sentence)
    
    return clean_sentences

def quality_filter_data(data: List[Dict[str, Any]], min_length: int, max_length: int) -> List[Dict[str, Any]]:
    """Filter data by quality criteria"""
    filtered_data = []
    
    for item in tqdm(data, desc="Quality filtering"):
        text = item['data']
        
        # Length check
        if not (min_length <= len(text) <= max_length):
            continue
        
        # Korean content check
        if not is_valid_korean_text(text):
            continue
        
        # Repetition check
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:  # Too repetitive
                continue
        
        filtered_data.append(item)
    
    logger.info(f"Quality filtered: kept {len(filtered_data)} of {len(data)} items")
    return filtered_data

def deduplicate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate entries"""
    seen_texts = set()
    unique_data = []
    
    for item in tqdm(data, desc="Deduplicating"):
        normalized_text = re.sub(r'\s+', ' ', item['data'].strip().lower())
        
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            unique_data.append(item)
    
    logger.info(f"Deduplication: kept {len(unique_data)} of {len(data)} items")
    return unique_data

def prepare_enhanced_data(source_folder: str, save_folder: str, sources: List[str], args) -> None:
    """Main function for enhanced data preparation"""
    source_folder = Path(source_folder)
    cleaner = Cleaner()
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    all_data = []
    
    # Process each data source
    if "speeches" in sources:
        logger.info("Processing New Year speeches...")
        speech_data = process_enhanced_speeches(source_folder, save_folder, args.split_ratio, cleaner)
        all_data.extend(speech_data)
    
    if "parallel" in sources:
        logger.info("Processing parallel data...")
        parallel_data = process_parallel_data(source_folder, cleaner)
        all_data.extend(parallel_data)
    
    if "dictionaries" in sources:
        logger.info("Processing dictionaries...")
        dict_data = process_dictionary_files(source_folder, args.max_dict_entries, cleaner)
        all_data.extend(dict_data)
    
    if "pdfs" in sources:
        logger.info("Processing PDFs...")
        # Process both PDF folders
        pdf_data = process_pdf_folder(source_folder, "PDFs", save_folder, cleaner)
        all_data.extend(pdf_data)
        
        century_data = process_pdf_folder(source_folder, "With The Century", save_folder, cleaner)
        all_data.extend(century_data)
    
    if "web" in sources and args.scraped_folders:
        logger.info("Processing web scraped data...")
        web_data = process_web_scraped_data(args.scraped_folders, cleaner)
        all_data.extend(web_data)
    
    if "rodong" in sources:
        logger.info("Processing Rodong data...")
        # Use original function for Rodong data
        rodong_folder = source_folder.parent / "rodong_data"  # Adjust path as needed
        if rodong_folder.exists():
            try:
                get_all_day_data(str(rodong_folder), save_folder, args.split_ratio)
            except Exception as e:
                logger.warning(f"Could not process Rodong data: {e}")
    
    logger.info(f"Collected {len(all_data)} total items from all sources")
    
    # Apply quality filtering and deduplication
    if args.quality_filter:
        all_data = quality_filter_data(all_data, args.min_text_length, args.max_text_length)
    
    if args.deduplicate:
        all_data = deduplicate_data(all_data)
    
    # Split and save data
    if args.apply_split and len(all_data) > 1:
        import random
        random.shuffle(all_data)
        
        split_point = int(len(all_data) * args.split_ratio)
        train_data = all_data[:split_point]
        val_data = all_data[split_point:]
        
        # Save train and validation
        datasets = [
            ('train.json', train_data),
            ('validation.json', val_data),
            ('all.json', all_data)
        ]
    else:
        # Save as single dataset
        datasets = [('all.json', all_data)]
    
    for filename, dataset in datasets:
        if dataset:
            filepath = os.path.join(save_folder, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({data_store_field: dataset}, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(dataset)} items to {filepath}")
    
    # Save metadata
    metadata = {
        "total_items": len(all_data),
        "sources_processed": sources,
        "quality_filtering": args.quality_filter,
        "deduplication": args.deduplicate,
        "split_ratio": args.split_ratio if args.apply_split else None,
        "processing_args": vars(args)
    }
    
    metadata_path = os.path.join(save_folder, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Enhanced dataset preparation complete. Metadata saved to {metadata_path}")

def main():
    """Main function - enhanced version"""
    args = parse_enhanced_args()
    
    if args.input_type == "enhanced":
        prepare_enhanced_data(args.source_folder, args.save_folder, args.sources, args)
    elif args.input_type == "rodong":
        get_all_day_data(args.source_folder, args.save_folder, split=args.split_ratio if args.apply_split else 1)
    elif args.input_type == "new_year":
        prepare_newyear_data(args.source_folder, args.save_folder, split=args.split_ratio if args.apply_split else 1)

if __name__ == "__main__":
    main()