# Fintech Text Analysis — Keyword Frequency Pipeline

This project implements a keyword frequency analysis pipeline for fintech-related terms in banks' annual reports. It extracts text from PDFs using multiple fallback methods (including OCR), searches for predefined fintech keywords, and outputs frequency statistics.

---

## Project Structure

```
project-root/
│
├── keywords frequencies/
│   └── kf_calculation.py   # Main pipeline script
├── requirements.txt         # Python dependencies
├── info/                    # Documentation and reference materials
│   ├── function_overview.txt
│   ├── methodology.txt
│   ├── outputs_info
│   └── The_Impact_of_Fintech_Innovation_on_Bank.pdf
├── data/                    # Place raw PDF annual reports here
└── outputs/                 # All processed results
    ├── corpus_status.csv    # Extracted text with extraction method status
    ├── keyword_freq_long.csv
    └── keyword_freq_wide.csv
```

---

## Requirements

- Python 3.9+
- Tesseract OCR (optional, for scanned PDFs)
  - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## Preparing the Data

### PDF Reports
- Place all annual reports in the `keywords frequencies/data/` folder
- Filenames must follow this format: `BankName_YYYY.pdf`
- Examples: `Maybank_2023.pdf`, `CIMB_2024.pdf`

---

## Running the Pipeline

From the project root, run:

```bash
python "keywords frequencies/kf_calculation.py"
```

Or navigate to the keywords frequencies folder:

```bash
cd "keywords frequencies"
python kf_calculation.py
```

### What the Script Does:

1. **Step 1**: Setup project folders (`data/`, `outputs/`)
2. **Step 2**: Extract text from PDFs using multi-library fallback:
   - Tries pypdf → pdfplumber → PyMuPDF → OCR (pytesseract)
   - Adds a 'status' column showing which method worked
   - Saves corpus with status to `corpus_status.csv`
3. **Step 3**: Load keyword dictionary (FTII/FTOI terms)
4. **Step 4**: Preprocess text (normalize, count words)
5. **Step 5**: Count keyword frequencies and save results

Logs will appear in the console showing progress for each step and a final summary of extraction methods used.

---

## PDF Processing

The script uses a robust multi-library fallback approach:

1. **pypdf**: Fast, works for most modern PDFs
2. **pdfplumber**: Better text extraction for complex layouts
3. **PyMuPDF**: Alternative library for difficult PDFs
4. **OCR (pytesseract + pdf2image)**: Extracts text from scanned/image-based PDFs

Each PDF gets a status indicator showing which method successfully extracted the text.

---

## Keyword Dictionary

The pipeline searches for two groups of fintech keywords:

### FTII (Fintech Input/Technology):
- artificial intelligence, ai, face recognition, voice recognition
- blockchain, alliance chain, distributed ledger
- cloud computing, cloud service, cloud platform
- big data, data flow, data mining, data visualization

### FTOI (Fintech Output/Innovation):
- online payment, mobile payment, qr code payment, digital wallet
- online loan, online finance, lending platform
- customer portrait, predictive model, credit evaluation, anti-fraud
- online banking, mobile banking, internet banking, bank app

---

## Outputs

### corpus_status.csv
- Columns: bank, year, filename, raw_text, status
- Shows extraction status for each PDF

### keyword_freq_long.csv
- Long format: bank, year, group, keyword, count, rel_freq, status
- Shows absolute count and relative frequency (count/word_count) for each keyword

### keyword_freq_wide.csv
- Wide format: bank, year, FTII__keyword1, FTII__keyword2, ..., FTOI__keyword1, ...
- One row per bank-year with all keywords as columns

---

## Example Workflow

1. Drop PDF files into `keywords frequencies/data/`:
   - `Maybank_2023.pdf`
   - `Maybank_2024.pdf`
   - `CIMB_2023.pdf`
   - `CIMB_2024.pdf`

2. Run `python "keywords frequencies/kf_calculation.py"`

3. Check results in `keywords frequencies/outputs/`:
   - Review extraction status in `corpus_status.csv`
   - Analyze keyword frequencies in `keyword_freq_long.csv` and `keyword_freq_wide.csv`

---

## Notes & Limitations

- The script suppresses library warnings for cleaner output
- PDFs that fail all extraction methods will have empty text and status="Failed"
- For scanned PDFs, make sure Tesseract OCR is installed on your system
- OCR processing is slower than direct text extraction
- Filenames must strictly follow the `BankName_YYYY.pdf` format

---

## Troubleshooting

**OCR not working?**
- Install Tesseract OCR system package
- On Windows, add Tesseract to your PATH environment variable

**No output files?**
- Check that PDF filenames follow the correct format
- Verify PDFs are in the `data/` folder (created automatically)
- Check console output for error messages

**Empty text extracted?**
- Some PDFs may have security restrictions
- Try opening the PDF manually to verify it contains text
- For image-based PDFs, ensure OCR is properly configured

---

## Reference Materials

See the `info/` folder for additional documentation:
- `function_overview.txt`: Detailed function descriptions
- `methodology.txt`: Research methodology
- PDF: Original research paper on fintech innovation impact

---

## Future Extensions

This pipeline currently performs keyword frequency analysis (Steps 1-5).
For a complete replication of the methodology in the reference paper, additional steps would include:
- Min-max normalization of frequencies
- Entropy weighting calculation
- Composite index construction (FTII, FTOI, FTI)
- Panel regression analysis with financial data
