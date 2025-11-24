#!/usr/bin/env python3
"""
Test the Dropbox downloader with a small sample before full download
"""

import sys
from pathlib import Path
import csv

CSV_FILE = Path("/Users/samuel/Downloads/dropbox_urls_deduped.csv")

def main():
    print("Testing Dropbox URL CSV file...")
    print()
    
    if not CSV_FILE.exists():
        print(f"❌ CSV file not found: {CSV_FILE}")
        return 1
    
    # Read first few URLs
    urls = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:  # First 5 URLs
                    break
                if 'url' in row and row['url'].strip():
                    urls.append(row['url'].strip())
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return 1
    
    print(f"✓ CSV file readable")
    print(f"✓ First few URLs:")
    for i, url in enumerate(urls, 1):
        # Extract filename
        import re
        match = re.search(r'/([^/]+\.pdf)', url, re.IGNORECASE)
        if match:
            filename = match.group(1).split('?')[0]
            print(f"  {i}. {filename}")
            print(f"     URL: {url[:80]}...")
        else:
            print(f"  {i}. URL: {url[:80]}...")
    
    print()
    print("Ready to download!")
    print()
    print("To download all PDFs:")
    print("  python3 download_dropbox_pdfs.py")
    print()
    print("This will:")
    print("  1. Download ~813 PDFs to Resources/PDFs/")
    print("  2. Skip already downloaded files")
    print("  3. Track progress in dropbox_download_log.json")
    print("  4. Take approximately 15-30 minutes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
