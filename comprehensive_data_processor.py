#!/usr/bin/env python3
"""
Comprehensive Data Processor for Enhanced DPRK-BERT Training
============================================================

This script processes all available North Korean dialect resources to create
the most comprehensive training dataset possible for BERT model enhancement.

Resources processed:
- Kim's New Year Speeches (NK dialect text)
- PDFs (NK regulations and documents) 
- With The Century (NK historical texts)
- Parallel Boost (NK/SK parallel sentences)
- Dictionaries (NK phone dictionary data)
- Gyeoremal (dialect differences with examples)

Focus: Maximum efficiency through parallel translation pairs and quality validation
"""

import os
import json
import csv
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import PyPDF2
import pdfplumber
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveDataProcessor:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.resources_path = self.base_path / "Resources"
        self.output_path = self.base_path / "enhanced_training_data"
        self.output_path.mkdir(exist_ok=True)
        
        # Data containers
        self.training_data = []
        self.parallel_pairs = []
        self.quality_issues = []
        
        logger.info(f"Initialized processor with base path: {self.base_path}")
        
    def process_kims_speeches(self) -> List[Dict]:
        """Process Kim's New Year speeches for NK dialect training data"""
        speeches_path = self.resources_path / "Kim's New Years Speeches"
        speeches_data = []
        
        logger.info("Processing Kim's New Year speeches...")
        
        for speech_file in speeches_path.glob("*.txt"):
            try:
                with open(speech_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Split into sentences using Korean sentence endings
                sentences = re.split(r'[.!?।]', content)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                
                year = speech_file.stem
                for i, sentence in enumerate(sentences):
                    if self._is_quality_sentence(sentence):
                        speeches_data.append({
                            'text': sentence,
                            'source': f'kim_speech_{year}',
                            'dialect_type': 'north_korean',
                            'year': year,
                            'sentence_id': i,
                            'data_type': 'political_speech'
                        })
                
                logger.info(f"Processed {len(sentences)} sentences from {year} speech")
                
            except Exception as e:
                logger.error(f"Error processing {speech_file}: {e}")
                self.quality_issues.append(f"Speech processing error: {speech_file} - {e}")
        
        logger.info(f"Total sentences from speeches: {len(speeches_data)}")
        return speeches_data
    
    def process_parallel_boost(self) -> List[Dict]:
        """Process Parallel Boost CSV files for NK/SK parallel translation pairs"""
        parallel_path = self.resources_path / "Parallel Boost"
        parallel_data = []
        
        logger.info("Processing Parallel Boost translation pairs...")
        
        for csv_file in parallel_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Processing {csv_file.name} with columns: {list(df.columns)}")
                
                # Try to identify NK/SK columns based on common patterns
                nk_cols = [col for col in df.columns if any(x in col.lower() for x in ['north', 'dprk', 'nk', '북'])]
                sk_cols = [col for col in df.columns if any(x in col.lower() for x in ['south', 'sk', 'rok', '남'])]
                
                if nk_cols and sk_cols:
                    nk_col, sk_col = nk_cols[0], sk_cols[0]
                    
                    for idx, row in df.iterrows():
                        nk_text = str(row[nk_col]).strip()
                        sk_text = str(row[sk_col]).strip()
                        
                        if self._is_quality_sentence(nk_text) and self._is_quality_sentence(sk_text):
                            # Add parallel pair
                            pair_id = f"parallel_{csv_file.stem}_{idx}"
                            
                            parallel_data.extend([
                                {
                                    'text': nk_text,
                                    'source': f'parallel_boost_{csv_file.stem}',
                                    'dialect_type': 'north_korean',
                                    'pair_id': pair_id,
                                    'parallel_text': sk_text,
                                    'data_type': 'parallel_translation'
                                },
                                {
                                    'text': sk_text,
                                    'source': f'parallel_boost_{csv_file.stem}',
                                    'dialect_type': 'south_korean', 
                                    'pair_id': pair_id,
                                    'parallel_text': nk_text,
                                    'data_type': 'parallel_translation'
                                }
                            ])
                            
                            # Add to parallel pairs for specialized training
                            self.parallel_pairs.append({
                                'north_korean': nk_text,
                                'south_korean': sk_text,
                                'source': csv_file.stem,
                                'pair_id': pair_id
                            })
                
                logger.info(f"Extracted {len([x for x in parallel_data if csv_file.stem in x['source']])} items from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                self.quality_issues.append(f"Parallel Boost error: {csv_file} - {e}")
        
        logger.info(f"Total parallel translation items: {len(parallel_data)}")
        logger.info(f"Total parallel pairs: {len(self.parallel_pairs)}")
        return parallel_data
    
    def process_pdfs(self) -> List[Dict]:
        """Process PDF documents with quality cleaning (remove page numbers, headers, etc.)"""
        pdf_path = self.resources_path / "PDFs"
        pdf_data = []
        
        logger.info("Processing PDF documents...")
        
        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                extracted_text = self._extract_pdf_text(pdf_file)
                cleaned_text = self._clean_pdf_text(extracted_text)
                
                # Split into sentences
                sentences = re.split(r'[.!?।]', cleaned_text)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
                
                for i, sentence in enumerate(sentences):
                    if self._is_quality_sentence(sentence) and not self._is_pdf_artifact(sentence):
                        pdf_data.append({
                            'text': sentence,
                            'source': f'pdf_{pdf_file.stem}',
                            'dialect_type': 'north_korean',
                            'document': pdf_file.name,
                            'sentence_id': i,
                            'data_type': 'legal_document'
                        })
                
                logger.info(f"Processed {len(sentences)} sentences from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {e}")
                self.quality_issues.append(f"PDF processing error: {pdf_file} - {e}")
        
        logger.info(f"Total sentences from PDFs: {len(pdf_data)}")
        return pdf_data
    
    def process_with_century(self) -> List[Dict]:
        """Process 'With The Century' historical documents"""
        century_path = self.resources_path / "With The Century" 
        century_data = []
        
        logger.info("Processing 'With The Century' documents...")
        
        for pdf_file in century_path.glob("*.pdf"):
            try:
                # These are large historical documents, process in chunks
                extracted_text = self._extract_pdf_text(pdf_file, max_pages=50)  # Limit for memory
                cleaned_text = self._clean_pdf_text(extracted_text)
                
                # Split into sentences  
                sentences = re.split(r'[.!?।]', cleaned_text)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
                
                for i, sentence in enumerate(sentences[:1000]):  # Limit sentences per document
                    if self._is_quality_sentence(sentence) and not self._is_pdf_artifact(sentence):
                        century_data.append({
                            'text': sentence,
                            'source': f'century_{pdf_file.stem}',
                            'dialect_type': 'north_korean', 
                            'document': pdf_file.name,
                            'sentence_id': i,
                            'data_type': 'historical_text'
                        })
                
                logger.info(f"Processed {len(sentences)} sentences from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing Century PDF {pdf_file}: {e}")
                self.quality_issues.append(f"Century PDF error: {pdf_file} - {e}")
        
        logger.info(f"Total sentences from With The Century: {len(century_data)}")
        return century_data
    
    def process_dictionaries(self) -> List[Dict]:
        """Process NK phone dictionary data"""
        dict_path = self.resources_path / "Dictionaries"
        dict_data = []
        
        logger.info("Processing dictionary data...")
        
        for csv_file in dict_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Processing dictionary {csv_file.name} with columns: {list(df.columns)}")
                
                # Process dictionary entries as training data
                for idx, row in df.iterrows():
                    # Extract words and definitions
                    for col in df.columns:
                        value = str(row[col]).strip()
                        if self._is_quality_sentence(value) and len(value) > 5:
                            dict_data.append({
                                'text': value,
                                'source': f'dictionary_{csv_file.stem}',
                                'dialect_type': 'north_korean',
                                'field': col,
                                'entry_id': idx,
                                'data_type': 'dictionary_entry'
                            })
                
                logger.info(f"Processed {len([x for x in dict_data if csv_file.stem in x['source']])} entries from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing dictionary {csv_file}: {e}")
                self.quality_issues.append(f"Dictionary error: {csv_file} - {e}")
        
        logger.info(f"Total dictionary entries: {len(dict_data)}")
        return dict_data
    
    def integrate_gyeoremal_data(self) -> List[Dict]:
        """Integrate existing gyeoremal data as parallel translation training"""
        gyeoremal_path = self.resources_path / "gyeoremal"
        gyeoremal_data = []
        
        logger.info("Integrating gyeoremal dialect data...")
        
        # Load the BERT training data
        bert_data_file = gyeoremal_path / "gyeoremal_bert_training_data.json"
        if bert_data_file.exists():
            with open(bert_data_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                gyeoremal_data.extend(existing_data)
        
        # Process meaning differences for parallel pairs
        meaning_file = gyeoremal_path / "meaning_differences.csv"
        if meaning_file.exists():
            df = pd.read_csv(meaning_file)
            for idx, row in df.iterrows():
                nk_meaning = str(row['north_meaning']).strip()
                sk_meaning = str(row['south_meaning']).strip()
                
                if self._is_quality_sentence(nk_meaning) and self._is_quality_sentence(sk_meaning):
                    pair_id = f"gyeoremal_meaning_{idx}"
                    
                    # Add parallel pair for dialect translation training
                    self.parallel_pairs.append({
                        'north_korean': nk_meaning,
                        'south_korean': sk_meaning,
                        'source': 'gyeoremal_meanings',
                        'word': row['word'],
                        'pair_id': pair_id
                    })
        
        logger.info(f"Integrated {len(gyeoremal_data)} gyeoremal training items")
        return gyeoremal_data
    
    def _extract_pdf_text(self, pdf_path: Path, max_pages: Optional[int] = None) -> str:
        """Extract text from PDF with error handling"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = min(len(pdf.pages), max_pages) if max_pages else len(pdf.pages)
                
                for i in range(pages_to_process):
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    pages_to_process = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)
                    
                    for i in range(pages_to_process):
                        page = reader.pages[i]
                        text += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e2}")
        
        return text
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF text by removing artifacts, page numbers, headers, etc."""
        if not text:
            return ""
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'^-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
        
        # Remove headers/footers (repeated lines)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip obvious artifacts
            if len(line) < 3:
                continue
            if line.isdigit():
                continue
            if re.match(r'^[^\w\u4e00-\u9fff\uac00-\ud7af]*$', line):  # Only punctuation
                continue
            if line.count('_') > len(line) // 2:  # Mostly underscores
                continue
                
            cleaned_lines.append(line)
        
        # Join and normalize whitespace
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def _is_quality_sentence(self, text: str) -> bool:
        """Check if text meets quality standards for training"""
        if not text or len(text.strip()) < 5:
            return False
        
        # Must contain Korean characters
        if not re.search(r'[\uac00-\ud7af]', text):
            return False
        
        # Reasonable length
        if len(text) > 500:
            return False
            
        # Not mostly punctuation or numbers
        alphanumeric_ratio = len(re.findall(r'[\w\uac00-\ud7af]', text)) / len(text)
        if alphanumeric_ratio < 0.6:
            return False
        
        return True
    
    def _is_pdf_artifact(self, text: str) -> bool:
        """Check if text is likely a PDF artifact (page numbers, headers, etc.)"""
        # Common PDF artifacts
        artifacts = [
            r'^\d+$',  # Just a number
            r'^page \d+',  # Page indicators
            r'^\d{4}-\d{2}-\d{2}',  # Dates
            r'^제\d+조',  # Article numbers
            r'^[A-Z]{2,}\s*$',  # All caps short words
        ]
        
        for pattern in artifacts:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        return False
    
    def save_comprehensive_dataset(self):
        """Save all processed data in formats optimized for BERT training"""
        
        # Collect all data
        all_data = []
        all_data.extend(self.process_kims_speeches())
        all_data.extend(self.process_parallel_boost())
        all_data.extend(self.process_pdfs())
        all_data.extend(self.process_with_century())
        all_data.extend(self.process_dictionaries())
        all_data.extend(self.integrate_gyeoremal_data())
        
        # Save comprehensive training dataset
        output_file = self.output_path / "comprehensive_bert_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Save parallel pairs separately for specialized training
        parallel_file = self.output_path / "parallel_translation_pairs.json"
        with open(parallel_file, 'w', encoding='utf-8') as f:
            json.dump(self.parallel_pairs, f, ensure_ascii=False, indent=2)
        
        # Save processing report
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'total_training_items': len(all_data),
            'parallel_translation_pairs': len(self.parallel_pairs),
            'data_sources': {
                'kims_speeches': len([x for x in all_data if 'kim_speech' in x['source']]),
                'parallel_boost': len([x for x in all_data if 'parallel_boost' in x['source']]),
                'pdfs': len([x for x in all_data if x['source'].startswith('pdf_')]),
                'with_century': len([x for x in all_data if 'century' in x['source']]),
                'dictionaries': len([x for x in all_data if 'dictionary' in x['source']]),
                'gyeoremal': len([x for x in all_data if 'gyeoremal' in x['source']])
            },
            'quality_issues': self.quality_issues
        }
        
        report_file = self.output_path / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE DATA PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"📊 Total training items: {len(all_data):,}")
        logger.info(f"🔄 Parallel translation pairs: {len(self.parallel_pairs):,}")
        logger.info(f"📁 Output directory: {self.output_path}")
        logger.info(f"📋 Processing report: {report_file}")
        logger.info(f"⚠️ Quality issues found: {len(self.quality_issues)}")
        
        return report


if __name__ == "__main__":
    processor = ComprehensiveDataProcessor()
    report = processor.save_comprehensive_dataset()
    
    print(f"\n🎉 Processing complete! Generated {report['total_training_items']:,} training items")
    print(f"🔄 Created {report['parallel_translation_pairs']:,} parallel translation pairs")
    print(f"📁 Data saved in: enhanced_training_data/")