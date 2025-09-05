# main_keywords_only.py
"""
Fintech Keyword Frequency (Steps 1–5 only)

What it does:
- Step 1: Create data/ and outputs/ folders beside this file
- Step 2: Read all PDFs in data/ (expects BankName_YYYY.pdf)
- Step 3: Prepare keyword dictionary (FTII / FTOI)
- Step 4: Normalize text (+ word counts)
- Step 5: Count keyword frequencies and save:
    - outputs/keyword_freq_long.csv
    - outputs/keyword_freq_wide.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re
import pandas as pd
from PyPDF2 import PdfReader

# -----------------------------
# Small logging helper
# -----------------------------
def log(step: str, msg: str, status: str = "INFO") -> None:
    icons = {"INFO": "ℹ️", "OK": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"[{step}] {icons.get(status, '')} {msg}")

# -----------------------------
# Step 1. Project setup
# -----------------------------
def setup_project(base_dir: Path) -> Tuple[Path, Path]:
    data_dir = base_dir / "data"
    out_dir = base_dir / "outputs"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    log("Step 1", f"Data folder → {data_dir}", "OK")
    log("Step 1", f"Outputs folder → {out_dir}", "OK")
    return data_dir, out_dir

# -----------------------------
# Step 2. PDF → Text
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(pdf_path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return " ".join(pages)
    except Exception as e:
        log("Step 2", f"Could not read {pdf_path.name}: {e}", "ERROR")
        return ""

def build_corpus(data_dir: Path) -> pd.DataFrame:
    log("Step 2", "Scanning PDFs in data/ ...", "INFO")
    rows = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        # Expect BankName_YYYY.pdf
        try:
            bank, y = pdf.stem.rsplit("_", 1)
            year = int(y)
        except ValueError:
            log("Step 2", f"Skip (bad name): {pdf.name} (expected BankName_YYYY.pdf)", "WARNING")
            continue
        txt = extract_text_from_pdf(pdf)
        rows.append({"bank": bank.strip(), "year": year, "filename": pdf.name, "raw_text": txt})
    df = pd.DataFrame(rows).sort_values(["bank", "year"]).reset_index(drop=True)
    if df.empty:
        log("Step 2", "No valid PDFs found.", "WARNING")
    else:
        log("Step 2", f"Corpus built with {len(df)} document(s)", "OK")
    return df

# -----------------------------
# Step 3. Keywords
# -----------------------------
def get_fintech_keywords() -> Dict[str, List[str]]:
    # FTII = input/technology; FTOI = output/innovation
    kws = {
        "FTII": [
            "artificial intelligence", "ai", "face recognition", "voice recognition",
            "fingerprint recognition", "blockchain", "alliance chain", "distributed ledger",
            "asymmetric encryption", "cloud computing", "cloud service", "cloud platform",
            "cloud architecture", "big data", "data flow", "data mining", "data visualization",
        ],
        "FTOI": [
            "online payment", "mobile payment", "qr code payment", "digital wallet",
            "online loan", "online finance", "lending platform", "inclusive credit",
            "customer portrait", "predictive model", "credit evaluation", "anti-fraud",
            "online banking", "mobile banking", "internet banking", "bank app",
        ],
    }
    total = sum(len(v) for v in kws.values())
    log("Step 3", f"Keyword dictionary ready with {total} term(s)", "OK")
    return kws

# -----------------------------
# Step 4. Normalize text
# -----------------------------
_WS = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("-\n", "")
    s = s.replace("\n", " ")
    s = s.lower()
    s = _WS.sub(" ", s)
    return s.strip()

def preprocess_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(clean_text=[], word_count=[])
    clean = df["raw_text"].apply(normalize_text)
    wc = clean.apply(lambda x: 0 if not x else len(x.split(" ")))
    out = df.copy()
    out["clean_text"] = clean
    out["word_count"] = wc
    log("Step 4", f"Preprocessed {len(out)} record(s)", "OK")
    return out

# -----------------------------
# Step 5. Count keyword freq
# -----------------------------
def _compile_patterns(kws_by_group: Dict[str, List[str]]) -> Dict[str, Dict[str, re.Pattern]]:
    compiled = {}
    for grp, words in kws_by_group.items():
        compiled[grp] = {}
        for kw in words:
            parts = [re.escape(t) for t in kw.split()]
            pat = r"\b" + r"\s+".join(parts) + r"\b"
            compiled[grp][kw] = re.compile(pat, flags=re.IGNORECASE)
    log("Step 5", "Compiled regex patterns for keywords", "OK")
    return compiled

def count_keywords(df_clean: pd.DataFrame, kws_by_group: Dict[str, List[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_clean.empty:
        cols = ["bank", "year", "group", "keyword", "count", "rel_freq"]
        return pd.DataFrame(columns=cols), pd.DataFrame()
    patterns = _compile_patterns(kws_by_group)
    rows = []
    for _, r in df_clean.iterrows():
        bank, year, text, wc = r["bank"], int(r["year"]), r.get("clean_text", "") or "", int(r.get("word_count", 0) or 0)
        for grp, pats in patterns.items():
            for kw, pat in pats.items():
                c = len(pat.findall(text))
                rel = (c / wc) if wc > 0 else 0.0
                rows.append({"bank": bank, "year": year, "group": grp, "keyword": kw, "count": c, "rel_freq": rel})
    long_df = pd.DataFrame(rows).sort_values(["bank", "year", "group", "keyword"]).reset_index(drop=True)

    if long_df.empty:
        log("Step 5", "No keyword matches found; wide table will be empty.", "WARNING")
        return long_df, pd.DataFrame()

    long_df["colname"] = long_df.apply(lambda x: f"{x['group']}__{x['keyword']}", axis=1)
    wide_df = (
        long_df.pivot_table(index=["bank", "year"], columns="colname", values="rel_freq", aggfunc="first")
        .reset_index()
    )
    wide_df = wide_df.loc[:, sorted(wide_df.columns, key=lambda c: (c not in ("bank", "year"), c))]
    log("Step 5", f"Built frequency tables for {wide_df.shape[0]} bank-year record(s)", "OK")
    return long_df, wide_df

def save_keyword_freq_outputs(freq_long: pd.DataFrame, freq_wide: pd.DataFrame, output_dir: Path) -> None:
    try:
        out_long = output_dir / "keyword_freq_long.csv"
        out_wide = output_dir / "keyword_freq_wide.csv"
        freq_long.to_csv(out_long, index=False, encoding="utf-8")
        log("Step 5", f"Keyword frequency (long) saved → {out_long}", "OK")
        if not freq_wide.empty:
            freq_wide.to_csv(out_wide, index=False, encoding="utf-8")
            log("Step 5", f"Keyword frequency (wide) saved → {out_wide}", "OK")
        else:
            log("Step 5", "Wide table not created (no matches).", "WARNING")
    except Exception as e:
        log("Step 5", f"Failed to save outputs: {e}", "ERROR")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    DATA_DIR, OUT_DIR = setup_project(BASE)

    # 2) PDFs → corpus
    df_corpus = build_corpus(DATA_DIR)

    # 3) Keywords
    keywords = get_fintech_keywords()

    # 4) Clean text & word counts
    df_clean = preprocess_corpus(df_corpus)

    # 5) Count & save
    freq_long, freq_wide = count_keywords(df_clean, keywords)
    save_keyword_freq_outputs(freq_long, freq_wide, OUT_DIR)

    log("Done", "Keyword frequency pipeline finished.", "OK")
