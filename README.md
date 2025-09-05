# Fintech Text Analysis Pipeline

This project replicates the methodology from the paper *“The Impact of Fintech Innovation on Bank Performance”*.  
It processes banks’ annual reports, builds Fintech Innovation indices (FTII, FTOI, FTI), and runs panel regressions to measure their impact on financial performance.

---

## Project Structure

project-root/
│
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
├── data/                   # Place raw PDF annual reports here
├── outputs/                # All processed results and analysis outputs
│   ├── corpus.csv
│   ├── corpus_clean.csv
│   ├── keyword_freq_long.csv
│   ├── keyword_freq_wide.csv
│   ├── keyword_freq_normalized.csv
│   ├── entropy_weights.csv
│   ├── fintech_indices_long.csv
│   ├── fintech_indices_wide.csv
│   ├── bank_financials.csv   # (provided/maintained manually)
│   ├── analysis_dataset.csv
│   ├── regression_pooled_ols.txt
│   ├── regression_fixed_effects.txt
│   ├── regression_random_effects.txt
│   ├── regression_pcse.txt
│   ├── desc_stats.csv
│   ├── corr_matrix.csv
│   └── vif.csv (if enough rows)

---

## Requirements

- Python 3.9+
- Install dependencies with:

    pip install -r requirements.txt

Dependencies are listed in `requirements.txt`.

---

## Preparing the Data

1. PDF Reports
   - Place all annual reports in the `data/` folder.
   - Filenames must follow this format:
     BankName_YYYY.pdf
     Example: Maybank_2023.pdf, CIMB_2024.pdf

2. Bank Financials CSV
   - A separate file `bank_financials.csv` must exist in `outputs/`.
   - This contains real financial/control variables used in regression (ROA, ROE, Assets, etc.).
   - Required columns:

     bank,year,ROA,ROE,Assets,YearsInBusiness,BookValue,CAR,EarningsAbility,LiquidityRatio,Growth,Inflation,Islamic,StateOwned

   - Ensure:
     - `bank` matches the PDF filename prefix exactly (e.g., Maybank).
     - `year` matches the year in the filename.

---

## Running the Pipeline

From the project root, run:

    python main.py

The script will:

1. Step 1: Setup project folders (`data/`, `outputs/`)
2. Step 2: Extract text from PDFs → corpus.csv
3. Step 3: Load keyword dictionary (FTII/FTOI terms)
4. Step 4: Preprocess text → corpus_clean.csv
5. Step 5: Count keyword frequencies → keyword_freq_long.csv, keyword_freq_wide.csv
6. Step 6: Apply min–max normalization → keyword_freq_normalized.csv
7. Step 7: Compute entropy weights → entropy_weights.csv
8. Step 8: Build indices (FTII, FTOI, FTI) → fintech_indices_long.csv, fintech_indices_wide.csv
9. Step 9: Merge with financials → analysis_dataset.csv
10. Step 10: Run regressions (Pooled OLS, FE, RE, PCSE) → summaries saved in outputs/
11. Step 11: Diagnostics (descriptives, correlations, VIF, Hausman test) → CSV outputs

Logs will appear in the console to show progress for each step.

---

## Outputs You Get

- Keyword-level results
  - Frequency counts (absolute + normalized)
  - Entropy weights

- Indices
  - FTII (input tech), FTOI (output innovation), FTI (combined)

- Analysis dataset
  - Merged fintech indices + financial metrics

- Regression results
  - Pooled OLS, Fixed Effects, Random Effects, PCSE
  - Diagnostics (descriptive stats, correlation, VIF, Hausman)

---

## Notes & Limitations

- If FE/RE regressions fail with "exog does not have full column rank", it means the dataset is too small (too few banks/years). Add more annual reports and rows in `bank_financials.csv`.
- PCSE and Pooled OLS will still run even with small data, but results won’t be meaningful with <10 rows.
- For robust replication like the paper, aim for 10 banks × 10 years (≈100 rows).

---

## Example Workflow

1. Drop Maybank_2023.pdf, Maybank_2024.pdf, CIMB_2023.pdf, CIMB_2024.pdf into `data/`.
2. Add corresponding rows in `bank_financials.csv`.
3. Run `python main.py`.
4. Check results in `outputs/`.
