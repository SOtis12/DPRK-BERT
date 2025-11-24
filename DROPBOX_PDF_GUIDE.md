# Dropbox PDF Download and Processing Guide

## Overview
Download 813 Tongil Sinbo (통일신보) PDFs from Dropbox and automatically process them for BERT training.

## Quick Start

### 1. Download PDFs from Dropbox
```bash
python3 download_dropbox_pdfs.py
```

This will:
- Download ~813 PDFs to `Resources/PDFs/`
- Skip already downloaded files
- Track progress in `dropbox_download_log.json`
- Take approximately 15-30 minutes
- Handle rate limiting automatically

### 2. Process PDFs for Training
```bash
python3 train_with_local_data.py
```

This will:
- Extract Korean text from all PDFs (including new Dropbox downloads)
- Clean and validate content
- Remove page numbers, headers, footers
- Create training dataset in `local_training_data/`

## What is Tongil Sinbo?

**통일신보 (Tongil Sinbo)** = "Unification News"
- North Korean newspaper published in Japan
- Weekly publication with NK perspective
- Rich source of North Korean dialect and terminology
- Valuable for NK/SK translation training

## Download Features

### Smart URL Handling
- Converts Dropbox sharing URLs to direct download URLs
- Extracts filenames from URLs automatically
- Handles malformed URLs gracefully

### Resume Capability
- Tracks all downloads in `dropbox_download_log.json`
- Skips already downloaded files
- Can be interrupted and resumed anytime
- Validates file sizes (rejects < 1KB files)

### Rate Limiting Protection
- 1 second delay between downloads
- Automatic backoff on HTTP 429 (rate limit)
- Configurable retry logic (3 attempts per file)
- 60 second timeout per download

### Progress Tracking
```
[INFO] [42/813] Processing: tongil-sinbo-2023-0204.pdf
[INFO] Downloading tongil-sinbo-2023-0204.pdf (attempt 1/3)
[INFO] ✓ Downloaded tongil-sinbo-2023-0204.pdf (342.5 KB)
[INFO] Progress: 40/813 - Success: 38, Failed: 2, Skipped: 0
```

## Processing Features

### PDF Text Extraction (Already Implemented)
The existing `train_with_local_data.py` script includes:

1. **Dual Extraction**
   - PyMuPDF (fitz) - primary (better for Korean)
   - PyPDF2 - fallback

2. **Text Cleaning**
   - Remove page numbers (standalone digits)
   - Remove headers/footers (low Korean content)
   - Remove short artifacts (< 5 characters)
   - Validate 50%+ Korean character ratio

3. **Smart Chunking**
   - Long PDFs split into ~2000 char chunks
   - Sentence-aware splitting
   - Preserves context

4. **Quality Validation**
   - Minimum 100 characters after cleaning
   - Minimum 50% Korean content
   - Filter corrupted/invalid PDFs

## File Structure

```
improved_dprk_bert/
├── Resources/
│   └── PDFs/
│       ├── tongil-sinbo-2024-01.pdf       ← Downloaded here
│       ├── tongil-sinbo-2024-0106.pdf
│       ├── Tongil-Sinbo-2023-0101.pdf
│       └── ... (~813 files)
├── download_dropbox_pdfs.py               ← Downloader script
├── dropbox_download_log.json              ← Progress tracking
├── train_with_local_data.py               ← PDF processor
└── data_processing_checkpoints/
    └── pdfs.json                          ← Processing cache
```

## Usage Examples

### Download All PDFs
```bash
# Full download (15-30 minutes)
python3 download_dropbox_pdfs.py
```

### Resume Interrupted Download
```bash
# Just run again - it will skip completed downloads
python3 download_dropbox_pdfs.py
```

### Process PDFs for Training
```bash
# This processes ALL PDFs in Resources/PDFs/
python3 train_with_local_data.py
```

### Check Download Status
```bash
# Count downloaded PDFs
ls -1 Resources/PDFs/*.pdf | wc -l

# Check total size
du -sh Resources/PDFs/
```

### Clear and Restart
```bash
# Clear download log to re-download everything
rm dropbox_download_log.json

# Clear PDF processing cache to reprocess
rm data_processing_checkpoints/pdfs.json

# Clear all downloaded PDFs
rm Resources/PDFs/*.pdf
```

## Expected Output

### Download Summary
```
======================================================================
DOWNLOAD SUMMARY
======================================================================
Total URLs:      813
✓ Downloaded:    805
✓ Skipped:       8
✗ Failed:        0
Output location: /path/to/Resources/PDFs

Total PDFs in PDFs: 813
======================================================================

Next steps:
1. Run PDF processing:
   python3 train_with_local_data.py

2. This will automatically:
   - Extract Korean text from PDFs
   - Clean and validate content
   - Add to training dataset
```

### Processing Summary
```
[INFO] Processing PDFs...
[INFO] Found 813 PDFs in PDFs
[INFO] Found 156 PDFs in With The Century
[INFO] Found 1247 text chunks from PDFs
[INFO] ✓ Checkpoint saved: pdfs (1247 items)

📊 Data sources breakdown:
   kim: 7 items
   parallel: 1924 items
   dictionary: 10000 items
   gyeoremal: 248 items
   pdf: 1247 items  ← Including Tongil Sinbo!
```

## Troubleshooting

### "No valid URLs found in CSV"
**Fix:** Check CSV file path in script matches your location
```python
CSV_FILE = Path("/Users/samuel/Downloads/dropbox_urls_deduped.csv")
```

### "Rate limited. Waiting 30 seconds..."
**Expected:** Dropbox has rate limits. Script handles this automatically.

### Downloads fail repeatedly
**Fix:** Check internet connection or try:
```bash
# Test a single URL manually
curl -L "https://www.dropbox.com/scl/fi/p0no5cs7neqvvpliskhjn/tongil-sinbo-2024-01.pdf?dl=1" -o test.pdf
```

### PDFs download but no text extracted
**Check:**
1. PDF libraries installed: `pip install PyPDF2 PyMuPDF`
2. PDFs are text-based (not scanned images)
3. Check individual PDF manually

### Out of disk space
**Estimate:** 813 PDFs × ~300KB average = ~240MB needed

## Configuration

### Adjust Download Settings
Edit `download_dropbox_pdfs.py`:
```python
BATCH_SIZE = 50              # Downloads per batch
DELAY_BETWEEN_DOWNLOADS = 1  # Seconds between downloads
MAX_RETRIES = 3              # Retry attempts
TIMEOUT = 60                 # Download timeout
```

### Adjust Processing Settings
Edit `train_with_local_data.py`:
```python
max_chunk_size = 2000        # Characters per chunk
korean_ratio_min = 0.5       # Minimum Korean content (50%)
min_text_length = 100        # Minimum chars after cleaning
```

## Data Quality

### Automatic Validation
Each PDF is validated for:
- ✅ Valid PDF file format
- ✅ Successful text extraction
- ✅ Minimum 100 characters
- ✅ At least 50% Korean characters
- ✅ No excessive page numbers/artifacts

### Manual Inspection
```bash
# View first processed PDF chunk
python3 -c "import json; data=json.load(open('local_training_data/train.json')); pdfs=[d for d in data['data'] if 'pdf' in d.get('source','')]; print(pdfs[0] if pdfs else 'No PDFs')"
```

## Integration with Training

The downloaded and processed PDFs automatically integrate:

```bash
# 1. Download PDFs
python3 download_dropbox_pdfs.py

# 2. Process all data sources (including PDFs)
python3 train_with_local_data.py

# 3. Train BERT model
cd DPRK-BERT-master
python mlm_trainer.py \
  --mode train \
  --train_file ../local_training_data/train.json \
  --validation_file ../local_training_data/validation.json \
  --num_train_epochs 10
```

## Performance

### Download Speed
- ~813 PDFs in 15-30 minutes
- ~0.5-1 MB/s average
- Depends on network and Dropbox limits

### Processing Speed
- ~813 PDFs processed in 5-10 minutes
- Cached after first run (instant resume)
- PyMuPDF faster than PyPDF2

### Storage Requirements
- PDFs: ~240 MB
- Extracted text (JSON): ~50-100 MB
- Total: ~300-400 MB

## Next Steps

1. **Download PDFs:**
   ```bash
   python3 download_dropbox_pdfs.py
   ```

2. **Verify downloads:**
   ```bash
   ls -lh Resources/PDFs/ | head -20
   ```

3. **Process for training:**
   ```bash
   python3 train_with_local_data.py
   ```

4. **Train model:**
   ```bash
   cd DPRK-BERT-master
   python mlm_trainer.py --mode train \
     --train_file ../local_training_data/train.json \
     --validation_file ../local_training_data/validation.json
   ```

---

**Status:** ✅ Ready to download and process  
**Source:** 813 Tongil Sinbo PDFs from Dropbox  
**Output:** High-quality NK dialect training data
