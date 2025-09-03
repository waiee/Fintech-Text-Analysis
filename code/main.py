# main.py
"""
Fintech Text Analysis (Steps 1–3)
- Step 1: Setup project structure
- Step 2: PDF → Text extraction and corpus building (saved as CSV)
- Step 3: Keyword dictionary preparation
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from PyPDF2 import PdfReader


# -----------------------------
# Step 1. Setup project folders
# -----------------------------
def setup_project(base_dir: Path) -> Tuple[Path, Path]:
    """
    Setup base folders for data and outputs.
    Returns (data_dir, output_dir).
    """
    data_dir = base_dir / "data"       # store raw PDFs
    output_dir = base_dir / "outputs"  # store processed results
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print("Project structure ready.")
    print(f"- Data folder:   {data_dir}")
    print(f"- Outputs folder:{output_dir}")
    return data_dir, output_dir


# -----------------------------
# Step 2. PDF → text extraction
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a single PDF file.
    Returns the entire text as a single string (best-effort).
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
    Columns: bank, year, filename, raw_text
    """
    records = []

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("[INFO] No PDF files found in 'data'. Add files and rerun.")
        return pd.DataFrame(columns=["bank", "year", "filename", "raw_text"])

    for pdf_file in pdf_files:
        # Parse filename convention: BankName_YYYY.pdf
        try:
            bank_name, year_str = pdf_file.stem.rsplit("_", 1)
            year = int(year_str)
        except ValueError:
            print(f"[WARNING] Skipping file {pdf_file.name} (expected BankName_YYYY.pdf)")
            continue

        raw_text = extract_text_from_pdf(pdf_file)

        records.append({
            "bank": bank_name.strip(),
            "year": year,
            "filename": pdf_file.name,
            "raw_text": raw_text
        })

    df = pd.DataFrame(records)
    # Sort for predictable ordering
    if not df.empty:
        df = df.sort_values(by=["bank", "year"], kind="stable").reset_index(drop=True)
    return df


def save_corpus_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save corpus DataFrame as CSV (UTF-8).
    """
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Corpus saved → {output_path}")


# -----------------------------
# Step 3. Keyword dictionary
# -----------------------------
def get_fintech_keywords() -> dict:
    """
    Return dictionary of fintech keywords grouped into
    input (FTII) and output (FTOI) categories.
    """
    keywords = {
        "FTII": [  # FinTech Input Index (technology terms)
            "artificial intelligence", "ai", "face recognition",
            "voice recognition", "fingerprint recognition",
            "blockchain", "alliance chain", "distributed ledger",
            "asymmetric encryption",
            "cloud computing", "cloud service", "cloud platform",
            "cloud architecture",
            "big data", "data flow", "data mining", "data visualization"
        ],
        "FTOI": [  # FinTech Output Index (innovation terms)
            "online payment", "mobile payment", "qr code payment",
            "digital wallet",
            "online loan", "online finance", "lending platform",
            "inclusive credit",
            "customer portrait", "predictive model", "credit evaluation",
            "anti-fraud",
            "online banking", "mobile banking", "internet banking", "bank app"
        ]
    }
    return keywords


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR, OUTPUT_DIR = setup_project(BASE_DIR)

    # Step 2: Build and save corpus
    df_corpus = build_corpus(DATA_DIR)
    save_corpus_csv(df_corpus, OUTPUT_DIR / "corpus.csv")

    # Step 3: Load fintech keywords
    fintech_keywords = get_fintech_keywords()
    print("✅ Keyword dictionary prepared:")
    for category, words in fintech_keywords.items():
        print(f"- {category}: {len(words)} keywords")
