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
# Step 4. Text preprocessing
# -----------------------------

import re

_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters/numbers and basic punctuation so phrase matching still works.
# We’ll normalize case and collapse whitespace; we won’t strip punctuation entirely yet,
# because some keywords are multi-word phrases and we want the text intact for regex counts.
def normalize_text(text: str) -> str:
    """
    Basic normalization:
      - Lowercase
      - Replace hyphen line breaks
      - Collapse whitespace
      - Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    # fix common PDF artifacts
    t = text.replace("-\n", "")            # join hyphenated line breaks
    t = t.replace("\n", " ")               # unify newlines to spaces
    t = t.lower()
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned text and simple word count.
    Columns added:
      - clean_text: normalized text for keyword search
      - word_count: total words in clean_text
    """
    if df.empty:
        return df.assign(clean_text=[], word_count=[])

    clean = df["raw_text"].apply(normalize_text)
    word_counts = clean.apply(lambda s: 0 if not s else len(s.split(" ")))
    out = df.copy()
    out["clean_text"] = clean
    out["word_count"] = word_counts
    return out


def save_clean_corpus_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned corpus DataFrame as CSV.
    Includes: bank, year, filename, word_count, clean_text
    (raw_text is excluded to keep file size manageable)
    """
    cols = ["bank", "year", "filename", "word_count", "clean_text"]
    df[cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Clean corpus saved → {output_path}")

# -----------------------------
# Step 4. Text preprocessing
# -----------------------------

import re

_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters/numbers and basic punctuation so phrase matching still works.
# We’ll normalize case and collapse whitespace; we won’t strip punctuation entirely yet,
# because some keywords are multi-word phrases and we want the text intact for regex counts.
def normalize_text(text: str) -> str:
    """
    Basic normalization:
      - Lowercase
      - Replace hyphen line breaks
      - Collapse whitespace
      - Strip leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    # fix common PDF artifacts
    t = text.replace("-\n", "")            # join hyphenated line breaks
    t = t.replace("\n", " ")               # unify newlines to spaces
    t = t.lower()
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()


def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned text and simple word count.
    Columns added:
      - clean_text: normalized text for keyword search
      - word_count: total words in clean_text
    """
    if df.empty:
        return df.assign(clean_text=[], word_count=[])

    clean = df["raw_text"].apply(normalize_text)
    word_counts = clean.apply(lambda s: 0 if not s else len(s.split(" ")))
    out = df.copy()
    out["clean_text"] = clean
    out["word_count"] = word_counts
    return out


def save_clean_corpus_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned corpus DataFrame as CSV.
    Includes: bank, year, filename, word_count, clean_text
    (raw_text is excluded to keep file size manageable)
    """
    cols = ["bank", "year", "filename", "word_count", "clean_text"]
    df[cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Clean corpus saved → {output_path}")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR, OUTPUT_DIR = setup_project(BASE_DIR)

    # Step 2: Build and save raw corpus (bank, year, filename, raw_text)
    df_corpus = build_corpus(DATA_DIR)
    save_corpus_csv(df_corpus, OUTPUT_DIR / "corpus.csv")

    # Step 3: Load fintech keywords
    fintech_keywords = get_fintech_keywords()
    print("Keyword dictionary prepared:")
    for category, words in fintech_keywords.items():
        print(f"- {category}: {len(words)} keywords")

    # Step 4: Preprocess text (normalized text + word counts)
    df_clean = preprocess_corpus(df_corpus)
    save_clean_corpus_csv(df_clean, OUTPUT_DIR / "corpus_clean.csv")

    # Small preview
    if not df_clean.empty:
        preview_cols = ["bank", "year", "filename", "word_count"]
        print(df_clean[preview_cols].head())

