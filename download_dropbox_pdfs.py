#!/usr/bin/env python3
"""
Download PDFs from Dropbox URLs and save to Resources/PDFs directory
Integrates with existing PDF processing framework
"""

import sys
import csv
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs
import re

# Paths
PROJECT_ROOT = Path(__file__).parent
RESOURCES_PDFS = PROJECT_ROOT / "Resources" / "PDFs"
CSV_FILE = Path("/Users/samuel/Downloads/dropbox_urls_deduped.csv")
DOWNLOAD_LOG = PROJECT_ROOT / "dropbox_download_log.json"

# Download settings
BATCH_SIZE = 50  # Download in batches to manage memory
DELAY_BETWEEN_DOWNLOADS = 1  # seconds
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

def log(msg: str):
    """Print log message"""
    print(f"[INFO] {msg}")

def convert_dropbox_url_to_direct(url: str) -> str:
    """
    Convert Dropbox sharing URL to direct download URL
    Changes dl=0 to dl=1 and removes unnecessary parameters
    """
    # Replace dl=0 with dl=1 for direct download
    if 'dl=0' in url:
        url = url.replace('dl=0', 'dl=1')
    elif 'dl=' not in url:
        # Add dl=1 if not present
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}dl=1"
    
    return url

def extract_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from Dropbox URL"""
    # Try to extract from URL path
    match = re.search(r'/([^/]+\.pdf)', url, re.IGNORECASE)
    if match:
        filename = match.group(1)
        # Clean up the filename
        filename = filename.split('?')[0]  # Remove query params
        return filename
    
    # Fallback: generate from URL hash
    url_hash = str(abs(hash(url)))[:10]
    return f"dropbox_{url_hash}.pdf"

def download_pdf(url: str, output_path: Path, retries: int = MAX_RETRIES) -> bool:
    """
    Download a PDF from Dropbox URL
    Returns True if successful, False otherwise
    """
    direct_url = convert_dropbox_url_to_direct(url)
    
    for attempt in range(retries):
        try:
            log(f"Downloading {output_path.name} (attempt {attempt + 1}/{retries})")
            
            response = requests.get(
                direct_url,
                timeout=TIMEOUT,
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            )
            
            # Check if successful
            if response.status_code == 200:
                # Check if content is actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    log(f"Warning: Unexpected content type: {content_type}")
                
                # Save the file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size
                file_size = output_path.stat().st_size
                if file_size < 1024:  # Less than 1KB is suspicious
                    log(f"Warning: File size very small ({file_size} bytes)")
                    output_path.unlink()  # Delete suspicious file
                    return False
                
                log(f"✓ Downloaded {output_path.name} ({file_size / 1024:.1f} KB)")
                return True
            
            elif response.status_code == 429:  # Too many requests
                wait_time = 30 * (attempt + 1)
                log(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            
            else:
                log(f"Failed: HTTP {response.status_code}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
        
        except requests.exceptions.Timeout:
            log(f"Timeout on attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(5)
        
        except Exception as e:
            log(f"Error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    
    return False

def load_download_log() -> Dict[str, bool]:
    """Load log of previously downloaded files"""
    if DOWNLOAD_LOG.exists():
        import json
        with open(DOWNLOAD_LOG, 'r') as f:
            return json.load(f)
    return {}

def save_download_log(log_data: Dict[str, bool]):
    """Save download log"""
    import json
    with open(DOWNLOAD_LOG, 'w') as f:
        json.dump(log_data, f, indent=2)

def main():
    """Main download function"""
    print("=" * 70)
    print("Dropbox PDF Downloader for DPRK-BERT")
    print("=" * 70)
    print()
    
    # Check if CSV exists
    if not CSV_FILE.exists():
        log(f"ERROR: CSV file not found: {CSV_FILE}")
        return 1
    
    # Create output directory
    RESOURCES_PDFS.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {RESOURCES_PDFS}")
    
    # Read URLs from CSV
    urls = []
    log(f"Reading URLs from {CSV_FILE}")
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and row['url'].strip():
                    # Handle multi-line URLs (CSV corruption)
                    url = row['url'].strip()
                    # If URL looks incomplete, skip
                    if url and url.startswith('http'):
                        urls.append(url)
    except Exception as e:
        log(f"Error reading CSV: {e}")
        return 1
    
    log(f"Found {len(urls)} URLs to process")
    
    if len(urls) == 0:
        log("ERROR: No valid URLs found in CSV")
        return 1
    
    # Load download log
    download_log = load_download_log()
    
    # Process URLs
    successful = 0
    failed = 0
    skipped = 0
    
    for i, url in enumerate(urls, 1):
        # Extract filename
        filename = extract_filename_from_url(url)
        if not filename:
            log(f"[{i}/{len(urls)}] Skipping: Could not extract filename")
            failed += 1
            continue
        
        output_path = RESOURCES_PDFS / filename
        
        # Check if already downloaded
        if output_path.exists():
            file_size = output_path.stat().st_size
            if file_size > 1024:  # Valid file
                log(f"[{i}/{len(urls)}] Skipping {filename} (already exists, {file_size / 1024:.1f} KB)")
                skipped += 1
                continue
        
        # Check download log
        if url in download_log and download_log[url]:
            log(f"[{i}/{len(urls)}] Skipping {filename} (in download log)")
            skipped += 1
            continue
        
        # Download
        log(f"[{i}/{len(urls)}] Processing: {filename}")
        success = download_pdf(url, output_path)
        
        if success:
            successful += 1
            download_log[url] = True
        else:
            failed += 1
            download_log[url] = False
        
        # Save log periodically
        if i % 10 == 0:
            save_download_log(download_log)
        
        # Rate limiting
        if i < len(urls):
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
        
        # Progress update every 20 downloads
        if i % 20 == 0:
            log(f"Progress: {i}/{len(urls)} - Success: {successful}, Failed: {failed}, Skipped: {skipped}")
    
    # Final save
    save_download_log(download_log)
    
    # Summary
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total URLs:      {len(urls)}")
    print(f"✓ Downloaded:    {successful}")
    print(f"✓ Skipped:       {skipped}")
    print(f"✗ Failed:        {failed}")
    print(f"Output location: {RESOURCES_PDFS}")
    print()
    
    # Count PDF files
    pdf_count = len(list(RESOURCES_PDFS.glob("*.pdf")))
    print(f"Total PDFs in {RESOURCES_PDFS.name}: {pdf_count}")
    print("=" * 70)
    
    if successful > 0:
        print()
        print("Next steps:")
        print("1. Run PDF processing:")
        print("   python3 train_with_local_data.py")
        print()
        print("2. This will automatically:")
        print("   - Extract Korean text from PDFs")
        print("   - Clean and validate content")
        print("   - Add to training dataset")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
