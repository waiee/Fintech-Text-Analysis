# main.py

import os
import re
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader


# -----------------------------
# Step 1. Setup project folders
# -----------------------------
def setup_project(base_dir: Path) -> tuple[Path, Path]:
    """
    Setup base folders for data and outputs.
    Returns (data_dir, output_dir).
    """
    data_dir = base_dir / "data"       # store raw PDFs
    output_dir = base_dir / "outputs"  # store processed results
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print(f"Project structure ready.\n- Data folder: {data_dir}\n- Outputs folder: {output_dir}")
    return data_dir, output_dir


# -----------------------------
# Step 2. PDF → text extraction
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a single PDF file.
    Returns the entire text as a single string.
    """
    text_content = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text_content.append(page.extract_text() or "")
    except Exception as e:
        print(f"[ERROR] Could not read {pdf_path.name}: {e}")
    return " ".join(text_content)


def build_corpus(data_dir: Path) -> pd.DataFrame:
    """
    Read all PDF files from data_dir and build a corpus dataframe.
    Assumes filenames follow pattern: BankName_YYYY.pdf
    """
    records = []
    for pdf_file in data_dir.glob("*.pdf"):
        raw_text = extract_text_from_pdf(pdf_file)

        # Parse filename convention: BankName_YYYY.pdf
        try:
            bank_name, year_str = pdf_file.stem.rsplit("_", 1)
            year = int(year_str)
        except ValueError:
            print(f"[WARNING] Skipping file {pdf_file.name} (invalid naming)")
            continue

        records.append({
            "bank": bank_name,
            "year": year,
            "filename": pdf_file.name,
            "raw_text": raw_text
        })

    return pd.DataFrame(records)


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR, OUTPUT_DIR = setup_project(BASE_DIR)

    # Build corpus from PDF reports
    df_corpus = build_corpus(DATA_DIR)
    df_corpus.to_parquet(OUTPUT_DIR / "corpus.parquet", index=False)

    print(f"✅ Corpus built with {len(df_corpus)} documents.")
    print(df_corpus.head())
