=========================================
README – Fintech Keyword Frequency Script
=========================================

This script extracts text from PDF annual reports, counts how often fintech-related keywords appear, 
and saves the results into CSV files.

-----------------------------------------
1. Requirements
-----------------------------------------
Install Python 3.9+ and the required packages:

    pip install pandas PyPDF2

-----------------------------------------
2. Project Structure
-----------------------------------------
When you first run the script, it will create two folders beside the .py file:

    data/     → put your PDF files here (BankName_YYYY.pdf)
    outputs/  → results will be written here

-----------------------------------------
3. Input Files
-----------------------------------------
Place your PDF reports inside the `data/` folder.

Filename format must be:

    BankName_YYYY.pdf

Examples:
    Maybank_2019.pdf
    CIMB_2020.pdf

-----------------------------------------
4. What the Script Does
-----------------------------------------
Steps performed:
1. Setup → ensures data/ and outputs/ folders exist
2. Read all PDFs from data/
3. Extract text and normalize (lowercase, remove extra spaces)
4. Count keyword frequencies (absolute and relative to word count)
5. Save results into CSVs

-----------------------------------------
5. Output Files
-----------------------------------------
Results are saved into the `outputs/` folder:

- keyword_freq_long.csv
    • One row per bank-year-keyword
    • Columns: bank, year, group (FTII/FTOI), keyword, count, rel_freq

- keyword_freq_wide.csv
    • One row per bank-year
    • Columns: bank, year, FTII__keyword, FTOI__keyword (relative frequencies)

-----------------------------------------
6. How to Run
-----------------------------------------
From the terminal, run:

    python main_keywords_only.py

Make sure your PDFs are already placed in the `data/` folder.

-----------------------------------------
7. Notes
-----------------------------------------
- If no valid PDFs are found, the script will log a warning.
- If keywords are not found in any text, `keyword_freq_wide.csv` may be empty.
- The keyword dictionary is built-in (FTII = input/technology, FTOI = output/innovation).
