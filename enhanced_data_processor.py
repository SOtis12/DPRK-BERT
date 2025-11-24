"""
Enhanced data processor for DPRK-BERT that handles multiple data sources:
- PDF text extraction (With The Century, regulations PDFs)  
- Dictionary processing (9.17M entries from NK phone)
- CSV parallel data (parallel boost/seed files)
- Kim's New Year speeches
- Web scraped data
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

# PDF processing
try:
    import PyPDF2
    import fitz  # PyMuPDF - better for Korean text extraction
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF libraries not available. Install with: pip install PyPDF2 PyMuPDF")

# Korean text processing
try:
    from konlpy.tag import Hannanum
    import hanja
    KOREAN_NLP_AVAILABLE = True
except ImportError:
    KOREAN_NLP_AVAILABLE = False
    print("Warning: Korean NLP libraries not available. Install with: pip install konlpy hanja")

# Import cleaner from DPRK-BERT-master
import sys
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "DPRK-BERT-master"))

try:
    from cleaner import Cleaner
except ImportError:
    # Fallback minimal cleaner if DPRK-BERT cleaner not available
    class Cleaner:
        def clean(self, text):
            return re.sub(r'\s+', ' ', text.strip())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """Enhanced data processor for multiple DPRK data sources"""
    
    def __init__(self, cleaner=None):
        self.cleaner = cleaner or Cleaner()
        self.data_store_field = "data"
        self.processed_stats = {
            "total_documents": 0,
            "total_sentences": 0,
            "sources": {},
            "quality_filtered": 0,
            "duplicates_removed": 0
        }
    
    def process_pdf_folder(self, pdf_folder: Path, source_name: str = "pdf") -> List[Dict[str, Any]]:
        """Extract text from all PDFs in a folder"""
        if not PDF_AVAILABLE:
            logger.error("PDF processing libraries not available")
            return []
        
        pdf_folder = Path(pdf_folder)
        pdf_files = list(pdf_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_folder}")
        
        data = []
        for pdf_file in tqdm(pdf_files, desc=f"Processing {source_name} PDFs"):
            try:
                text = self._extract_pdf_text(pdf_file)
                if text and len(text.strip()) > 100:  # Filter very short texts
                    # Clean and validate text
                    cleaned_text = self.cleaner.clean(text)
                    if self._is_valid_korean_text(cleaned_text):
                        doc_data = {
                            "id": f"{source_name}_{pdf_file.stem}",
                            "data": cleaned_text,
                            "source": source_name,
                            "type": "pdf_document",
                            "filename": pdf_file.name,
                            "file_size": pdf_file.stat().st_size,
                            "char_count": len(cleaned_text)
                        }
                        data.append(doc_data)
                    else:
                        logger.warning(f"Invalid Korean text in {pdf_file.name}")
                else:
                    logger.warning(f"No substantial text extracted from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        logger.info(f"Successfully processed {len(data)} PDFs from {source_name}")
        self.processed_stats["sources"][source_name] = len(data)
        return data
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF (better for Korean)"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            full_text = "\n".join(text_parts)
            
            # Fallback to PyPDF2 if PyMuPDF fails
            if not full_text.strip():
                full_text = self._extract_pdf_text_pypdf2(pdf_path)
            
            return full_text
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}: {e}, trying PyPDF2")
            return self._extract_pdf_text_pypdf2(pdf_path)
    
    def _extract_pdf_text_pypdf2(self, pdf_path: Path) -> str:
        """Fallback PDF text extraction using PyPDF2"""
        try:
            text_parts = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PyPDF2 also failed for {pdf_path}: {e}")
            return ""
    
    def process_dictionary_folder(self, dict_folder: Path, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process dictionary CSV files from NK phone"""
        dict_folder = Path(dict_folder)
        csv_files = list(dict_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} dictionary CSV files")
        
        data = []
        total_entries = 0
        
        for csv_file in tqdm(csv_files, desc="Processing dictionary files"):
            try:
                entries = self._process_dictionary_csv(csv_file, max_entries)
                data.extend(entries)
                total_entries += len(entries)
                
                if max_entries and total_entries >= max_entries:
                    logger.info(f"Reached max entries limit: {max_entries}")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing dictionary {csv_file}: {e}")
        
        logger.info(f"Processed {len(data)} dictionary entries")
        self.processed_stats["sources"]["dictionary"] = len(data)
        return data
    
    def _process_dictionary_csv(self, csv_file: Path, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process individual dictionary CSV file"""
        entries = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                # Try to detect CSV format
                sample = f.read(1024)
                f.seek(0)
                
                # Use sniffer to detect delimiter
                sniffer = csv.Sniffer()
                delimiter = ','
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except:
                    pass
                
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                
                for row_num, row in enumerate(reader):
                    if max_entries and len(entries) >= max_entries:
                        break
                        
                    if len(row) >= 2:
                        # Assume format: [term, definition] or [english, korean]
                        term = row[0].strip()
                        definition = row[1].strip()
                        
                        if term and definition:
                            # Create training text from dictionary entry
                            if self._is_korean_text(definition):
                                # Korean definition - create definition text
                                text = f"{term}: {definition}"
                            else:
                                # Parallel translation - create translation text  
                                text = f"{term} = {definition}"
                            
                            cleaned_text = self.cleaner.clean(text)
                            if len(cleaned_text.strip()) > 10:
                                entry_data = {
                                    "id": f"dict_{csv_file.stem}_{row_num}",
                                    "data": cleaned_text,
                                    "source": "dictionary",
                                    "type": "dictionary_entry",
                                    "term": term,
                                    "definition": definition,
                                    "filename": csv_file.name
                                }
                                entries.append(entry_data)
        
        except Exception as e:
            logger.error(f"Error reading CSV {csv_file}: {e}")
        
        return entries
    
    def process_parallel_csv_files(self, parallel_files: List[Path]) -> List[Dict[str, Any]]:
        """Process parallel boost/seed CSV files"""
        data = []
        
        for csv_file in parallel_files:
            logger.info(f"Processing parallel data: {csv_file}")
            try:
                entries = self._process_parallel_csv(csv_file)
                data.extend(entries)
                logger.info(f"Added {len(entries)} parallel entries from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        self.processed_stats["sources"]["parallel"] = len(data)
        return data
    
    def _process_parallel_csv(self, csv_file: Path) -> List[Dict[str, Any]]:
        """Process individual parallel CSV file"""
        entries = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader):
                try:
                    # Expected format: id,domain,tags,south_korean,north_korean
                    south_korean = row.get('south_korean', '').strip()
                    north_korean = row.get('north_korean', '').strip()
                    domain = row.get('domain', '').strip()
                    tags = row.get('tags', '').strip()
                    
                    if south_korean and north_korean:
                        # Create separate entries for each variant
                        # South Korean version
                        sk_data = {
                            "id": f"parallel_sk_{csv_file.stem}_{row_num}",
                            "data": self.cleaner.clean(south_korean),
                            "source": "parallel",
                            "type": "parallel_south",
                            "domain": domain,
                            "tags": tags,
                            "parallel_pair": north_korean
                        }
                        
                        # North Korean version  
                        nk_data = {
                            "id": f"parallel_nk_{csv_file.stem}_{row_num}",
                            "data": self.cleaner.clean(north_korean),
                            "source": "parallel", 
                            "type": "parallel_north",
                            "domain": domain,
                            "tags": tags,
                            "parallel_pair": south_korean
                        }
                        
                        entries.extend([sk_data, nk_data])
                        
                except Exception as e:
                    logger.warning(f"Error processing row {row_num} in {csv_file}: {e}")
        
        return entries
    
    def process_new_year_speeches(self, speeches_folder: Path) -> List[Dict[str, Any]]:
        """Process Kim's New Year speeches"""
        speeches_folder = Path(speeches_folder)
        txt_files = list(speeches_folder.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} speech files")
        
        data = []
        for txt_file in tqdm(txt_files, desc="Processing speeches"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                cleaned_text = self.cleaner.clean(text)
                if cleaned_text.strip():
                    year = txt_file.stem  # Filename should be year.txt
                    speech_data = {
                        "id": f"speech_{year}",
                        "data": cleaned_text,
                        "source": "new_year_speech",
                        "type": "political_speech",
                        "year": int(year) if year.isdigit() else None,
                        "filename": txt_file.name,
                        "char_count": len(cleaned_text)
                    }
                    data.append(speech_data)
                    
            except Exception as e:
                logger.error(f"Error processing speech {txt_file}: {e}")
        
        logger.info(f"Processed {len(data)} speeches")
        self.processed_stats["sources"]["speeches"] = len(data)
        return data
    
    def process_web_scraped_data(self, scraped_folders: List[Path]) -> List[Dict[str, Any]]:
        """Process web scraped BERT data"""
        data = []
        
        for folder in scraped_folders:
            folder = Path(folder)
            bert_data_folder = folder / "bert_data"
            
            if bert_data_folder.exists():
                bert_files = list(bert_data_folder.glob("*_bert_mlm.txt"))
                logger.info(f"Found {len(bert_files)} BERT files in {folder}")
                
                for bert_file in tqdm(bert_files, desc=f"Processing {folder.name}"):
                    try:
                        with open(bert_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        if text.strip():
                            # Split into sentences for better training
                            sentences = self._split_into_sentences(text)
                            
                            for i, sentence in enumerate(sentences):
                                if len(sentence.strip()) > 20:  # Filter very short sentences
                                    cleaned_sentence = self.cleaner.clean(sentence)
                                    if self._is_valid_korean_text(cleaned_sentence):
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
        
        logger.info(f"Processed {len(data)} web scraped sentences")
        self.processed_stats["sources"]["web"] = len(data)
        return data
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split Korean text into sentences"""
        # Korean sentence endings
        sentence_endings = r'[.!?。！？]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _is_valid_korean_text(self, text: str) -> bool:
        """Check if text contains substantial Korean content"""
        if not text.strip():
            return False
        
        # Count Korean characters (Hangul)
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))  # Exclude spaces
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio > 0.3 and korean_chars > 10  # At least 30% Korean and 10+ Korean chars
    
    def _is_korean_text(self, text: str) -> bool:
        """Simple check if text contains Korean characters"""
        return bool(re.search(r'[가-힣]', text))
    
    def deduplicate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entries based on text content"""
        seen_texts = set()
        unique_data = []
        duplicates = 0
        
        for item in tqdm(data, desc="Deduplicating"):
            # Normalize text for comparison
            normalized_text = re.sub(r'\s+', ' ', item['data'].strip().lower())
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_data.append(item)
            else:
                duplicates += 1
        
        self.processed_stats["duplicates_removed"] = duplicates
        logger.info(f"Removed {duplicates} duplicates, kept {len(unique_data)} unique items")
        return unique_data
    
    def quality_filter_data(self, data: List[Dict[str, Any]], min_length: int = 20, max_length: int = 1000) -> List[Dict[str, Any]]:
        """Filter data by quality criteria"""
        filtered_data = []
        filtered_count = 0
        
        for item in tqdm(data, desc="Quality filtering"):
            text = item['data']
            
            # Length check
            if not (min_length <= len(text) <= max_length):
                filtered_count += 1
                continue
            
            # Korean content check
            if not self._is_valid_korean_text(text):
                filtered_count += 1
                continue
            
            # Additional quality checks
            if self._passes_quality_checks(text):
                filtered_data.append(item)
            else:
                filtered_count += 1
        
        self.processed_stats["quality_filtered"] = filtered_count
        logger.info(f"Quality filtered {filtered_count} items, kept {len(filtered_data)} high-quality items")
        return filtered_data
    
    def _passes_quality_checks(self, text: str) -> bool:
        """Additional quality checks for text"""
        # Check for excessive repetition
        words = text.split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:  # Too repetitive
                return False
        
        # Check for excessive punctuation or symbols
        non_alnum_chars = len(re.findall(r'[^\w\s가-힣]', text))
        if non_alnum_chars > len(text) * 0.3:  # More than 30% punctuation
            return False
        
        return True
    
    def compile_final_dataset(self, all_data: List[Dict[str, Any]], output_dir: Path, split_ratio: float = 0.9) -> Dict[str, Any]:
        """Compile final dataset with train/validation split"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update stats
        self.processed_stats["total_documents"] = len(all_data)
        self.processed_stats["total_sentences"] = sum(1 for item in all_data if 'sentence' in item.get('type', ''))
        
        # Shuffle data
        import random
        shuffled_data = all_data.copy()
        random.shuffle(shuffled_data)
        
        # Split data
        split_point = int(len(shuffled_data) * split_ratio)
        train_data = shuffled_data[:split_point]
        val_data = shuffled_data[split_point:]
        
        # Save datasets
        datasets = {
            'train.json': {"data": train_data},
            'validation.json': {"data": val_data},
            'all.json': {"data": shuffled_data}
        }
        
        for filename, dataset in datasets.items():
            output_file = output_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(dataset['data'])} items to {output_file}")
        
        # Save metadata
        metadata = {
            "processing_stats": self.processed_stats,
            "dataset_info": {
                "total_items": len(all_data),
                "train_items": len(train_data),
                "validation_items": len(val_data),
                "split_ratio": split_ratio
            },
            "sources_breakdown": self.processed_stats["sources"]
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset compilation complete. Metadata saved to {metadata_file}")
        return metadata


def main():
    """Main function for standalone processing"""
    parser = argparse.ArgumentParser(description="Enhanced DPRK-BERT data processor")
    parser.add_argument("--resources-folder", type=Path, required=True, help="Path to Resources folder")
    parser.add_argument("--scraped-folders", type=Path, nargs="*", help="Paths to scraped data folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for processed data")
    parser.add_argument("--max-dict-entries", type=int, default=100000, help="Max dictionary entries to process")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--skip-pdfs", action="store_true", help="Skip PDF processing")
    parser.add_argument("--skip-dictionaries", action="store_true", help="Skip dictionary processing")
    
    args = parser.parse_args()
    
    processor = EnhancedDataProcessor()
    all_data = []
    
    resources_folder = Path(args.resources_folder)
    
    # Process PDFs
    if not args.skip_pdfs:
        pdf_folders = [
            (resources_folder / "PDFs", "regulations"),
            (resources_folder / "With The Century", "century")
        ]
        
        for pdf_folder, source_name in pdf_folders:
            if pdf_folder.exists():
                pdf_data = processor.process_pdf_folder(pdf_folder, source_name)
                all_data.extend(pdf_data)
    
    # Process dictionaries
    if not args.skip_dictionaries:
        dict_folder = resources_folder / "Dictionaries"
        if dict_folder.exists():
            dict_data = processor.process_dictionary_folder(dict_folder, args.max_dict_entries)
            all_data.extend(dict_data)
    
    # Process parallel data
    parallel_folder = resources_folder / "Parallel Boost"
    if parallel_folder.exists():
        parallel_files = list(parallel_folder.glob("*.csv"))
        if parallel_files:
            parallel_data = processor.process_parallel_csv_files(parallel_files)
            all_data.extend(parallel_data)
    
    # Process speeches
    speeches_folder = resources_folder / "Kim's New Years Speeches"
    if speeches_folder.exists():
        speech_data = processor.process_new_year_speeches(speeches_folder)
        all_data.extend(speech_data)
    
    # Process web scraped data
    if args.scraped_folders:
        web_data = processor.process_web_scraped_data(args.scraped_folders)
        all_data.extend(web_data)
    
    # Quality processing
    logger.info("Starting data quality processing...")
    all_data = processor.quality_filter_data(all_data)
    all_data = processor.deduplicate_data(all_data)
    
    # Compile final dataset
    metadata = processor.compile_final_dataset(all_data, args.output_dir, args.split_ratio)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total documents processed: {metadata['processing_stats']['total_documents']}")
    print(f"Sources breakdown: {metadata['sources_breakdown']}")
    print(f"Quality filtered: {metadata['processing_stats']['quality_filtered']}")
    print(f"Duplicates removed: {metadata['processing_stats']['duplicates_removed']}")
    print(f"Final training items: {metadata['dataset_info']['train_items']}")
    print(f"Final validation items: {metadata['dataset_info']['validation_items']}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()