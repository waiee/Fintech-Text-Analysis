# main.py
"""
Fintech Text Analysis (Steps 1–5)
- Step 1: Setup project structure
- Step 2: PDF → Text extraction and corpus building (saved as CSV)
- Step 3: Keyword dictionary preparation
- Step 4: Text preprocessing (normalize + word counts)
- Step 5: Keyword frequency counting (absolute + normalized)
"""

from pathlib import Path
from typing import Tuple, Dict, List, Tuple as Tup
import re
import pandas as pd
from PyPDF2 import PdfReader


# -----------------------------
# Logging helper
# -----------------------------
def log_step(step: str, message: str, status: str = "INFO") -> None:
    """
    Print a formatted log message for a step.
    status: INFO | OK | ERROR | WARNING
    """
    icons = {
        "INFO": "ℹ️",
        "OK": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️"
    }
    icon = icons.get(status.upper(), "")
    print(f"[{step}] {icon} {message}")


# -----------------------------
# Step 1. Setup project folders
# -----------------------------
def setup_project(base_dir: Path) -> Tuple[Path, Path]:
    """
    Setup base folders for data and outputs.
    Returns (data_dir, output_dir).
    """
    log_step("Step 1", "Setting up project structure...", "INFO")

    data_dir = base_dir / "data"       # store raw PDFs
    output_dir = base_dir / "outputs"  # store processed results
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    log_step("Step 1", f"Data folder: {data_dir}", "OK")
    log_step("Step 1", f"Outputs folder: {output_dir}", "OK")
    log_step("Step 1", "Project structure ready", "OK")
    return data_dir, output_dir


# -----------------------------
# Step 2. PDF → text extraction
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a single PDF file (best-effort).
    """
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        for page in reader.pages:
            text_content.append(page.extract_text() or "")
        return " ".join(text_content)
    except Exception as e:
        log_step("Step 2", f"Could not read {pdf_path.name}: {e}", "ERROR")
        return ""


def build_corpus(data_dir: Path) -> pd.DataFrame:
    """
    Read all PDF files from data_dir and build a corpus dataframe.
    Assumes filenames follow pattern: BankName_YYYY.pdf
    Columns: bank, year, filename, raw_text
    """
    log_step("Step 2", "Building raw corpus from PDFs...", "INFO")

    records = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        log_step("Step 2", "No PDF files found in 'data'. Add files and rerun.", "WARNING")
        return pd.DataFrame(columns=["bank", "year", "filename", "raw_text"])

    for pdf_file in pdf_files:
        # Parse filename convention: BankName_YYYY.pdf
        try:
            bank_name, year_str = pdf_file.stem.rsplit("_", 1)
            year = int(year_str)
        except ValueError:
            log_step("Step 2", f"Skipping file {pdf_file.name} (expected BankName_YYYY.pdf)", "WARNING")
            continue

        raw_text = extract_text_from_pdf(pdf_file)
        records.append({
            "bank": bank_name.strip(),
            "year": year,
            "filename": pdf_file.name,
            "raw_text": raw_text
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=["bank", "year"], kind="stable").reset_index(drop=True)
        log_step("Step 2", f"Corpus built with {len(df)} document(s)", "OK")
    else:
        log_step("Step 2", "Corpus is empty after processing PDFs.", "WARNING")
    return df


def save_corpus_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save corpus DataFrame as CSV (UTF-8).
    """
    try:
        df.to_csv(output_path, index=False, encoding="utf-8")
        log_step("Step 2", f"Corpus saved → {output_path}", "OK")
    except Exception as e:
        log_step("Step 2", f"Failed to save corpus: {e}", "ERROR")


# -----------------------------
# Step 3. Keyword dictionary
# -----------------------------
def get_fintech_keywords() -> Dict[str, List[str]]:
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
    total = sum(len(v) for v in keywords.values())
    log_step("Step 3", f"Keyword dictionary ready with {total} term(s)", "OK")
    for grp, terms in keywords.items():
        log_step("Step 3", f"{grp}: {len(terms)} keyword(s)", "INFO")
    return keywords


# -----------------------------
# Step 4. Text preprocessing
# -----------------------------
_WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """
    Basic normalization:
      - Lowercase
      - Replace hyphen line breaks (fin-\ntech -> fintech)
      - Replace newlines with spaces
      - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    t = text.replace("-\n", "")
    t = t.replace("\n", " ")
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
    log_step("Step 4", "Preprocessing text (normalize + word counts)...", "INFO")
    if df.empty:
        log_step("Step 4", "Input corpus is empty; nothing to preprocess.", "WARNING")
        return df.assign(clean_text=[], word_count=[])

    clean = df["raw_text"].apply(normalize_text)
    word_counts = clean.apply(lambda s: 0 if not s else len(s.split(" ")))
    out = df.copy()
    out["clean_text"] = clean
    out["word_count"] = word_counts
    log_step("Step 4", f"Preprocessed {len(out)} record(s)", "OK")
    return out


def save_clean_corpus_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the cleaned corpus DataFrame as CSV.
    Includes: bank, year, filename, word_count, clean_text
    (raw_text is excluded to keep file size manageable)
    """
    try:
        cols = ["bank", "year", "filename", "word_count", "clean_text"]
        df[cols].to_csv(output_path, index=False, encoding="utf-8")
        log_step("Step 4", f"Clean corpus saved → {output_path}", "OK")
    except Exception as e:
        log_step("Step 4", f"Failed to save clean corpus: {e}", "ERROR")


# -----------------------------
# Step 5. Keyword frequency counting
# -----------------------------
def _compile_patterns(keywords_by_group: Dict[str, List[str]]) -> Dict[str, Dict[str, re.Pattern]]:
    """
    Compile case-insensitive regex patterns for each keyword, grouped by FTII/FTOI.
    - Uses word boundaries so 'ai' won't match inside 'chairman'.
    - Multi-word phrases are joined with \s+ to tolerate extra spaces/newlines.
    """
    compiled: Dict[str, Dict[str, re.Pattern]] = {}
    for group, words in keywords_by_group.items():
        group_patterns: Dict[str, re.Pattern] = {}
        for kw in words:
            terms = [re.escape(t) for t in kw.split()]
            pattern_str = r"\b" + r"\s+".join(terms) + r"\b"
            group_patterns[kw] = re.compile(pattern_str, flags=re.IGNORECASE)
        compiled[group] = group_patterns
    log_step("Step 5", "Compiled regex patterns for keywords", "OK")
    return compiled


def count_keywords_in_text(text: str, compiled: Dict[str, Dict[str, re.Pattern]]) -> Dict[Tup[str, str], int]:
    """
    Count occurrences of each keyword in 'text'.
    Returns a dict with keys (group, keyword) -> count.
    """
    counts: Dict[Tup[str, str], int] = {}
    text = text or ""
    for group, patterns in compiled.items():
        for kw, pat in patterns.items():
            hits = pat.findall(text)
            counts[(group, kw)] = len(hits)
    return counts


def build_keyword_frequencies(
    df_clean: pd.DataFrame,
    keywords_by_group: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (bank, year) document:
      - Count occurrences per keyword (absolute count)
      - Compute normalized frequency = count / word_count

    Returns:
      - freq_long: tidy long-form table (bank, year, group, keyword, count, rel_freq)
      - freq_wide: one row per (bank, year) with columns for each keyword's rel_freq
                   (prefixed with group, e.g., FTII__artificial intelligence)
    """
    log_step("Step 5", "Counting keyword frequencies...", "INFO")

    if df_clean.empty:
        log_step("Step 5", "Clean corpus is empty; skipping counting.", "WARNING")
        cols = ["bank", "year", "group", "keyword", "count", "rel_freq"]
        return pd.DataFrame(columns=cols), pd.DataFrame()

    compiled = _compile_patterns(keywords_by_group)

    rows = []
    for _, row in df_clean.iterrows():
        bank = row["bank"]
        year = int(row["year"])
        wc = int(row.get("word_count", 0)) or 0
        txt = row.get("clean_text", "") or ""

        counts = count_keywords_in_text(txt, compiled)
        for (group, kw), c in counts.items():
            rel = (c / wc) if wc > 0 else 0.0
            rows.append({
                "bank": bank,
                "year": year,
                "group": group,
                "keyword": kw,
                "count": c,
                "rel_freq": rel
            })

    freq_long = pd.DataFrame(rows).sort_values(["bank", "year", "group", "keyword"], kind="stable")

    # Build a wide table of rel_freqs for convenience in later steps
    if not freq_long.empty:
        freq_long["colname"] = freq_long.apply(
            lambda r: f"{r['group']}__{r['keyword']}", axis=1
        )
        freq_wide = (
            freq_long
            .pivot_table(index=["bank", "year"], columns="colname", values="rel_freq", aggfunc="first")
            .reset_index()
        )
        # Ensure stable column order
        freq_wide = freq_wide.loc[:, sorted(freq_wide.columns, key=lambda c: (c not in ("bank", "year"), c))]
        log_step("Step 5", f"Built frequency tables for {freq_wide.shape[0]} bank-year record(s)", "OK")
    else:
        freq_wide = pd.DataFrame()
        log_step("Step 5", "No keyword matches found across documents.", "WARNING")

    return freq_long, freq_wide


def save_keyword_freq_outputs(
    freq_long: pd.DataFrame,
    freq_wide: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Save keyword frequency results:
      - outputs/keyword_freq_long.csv  (tidy long format)
      - outputs/keyword_freq_wide.csv  (wide format of rel_freq only)
    """
    try:
        long_path = output_dir / "keyword_freq_long.csv"
        wide_path = output_dir / "keyword_freq_wide.csv"

        freq_long.to_csv(long_path, index=False, encoding="utf-8")
        log_step("Step 5", f"Keyword frequency (long) saved → {long_path}", "OK")

        if not freq_wide.empty:
            freq_wide.to_csv(wide_path, index=False, encoding="utf-8")
            log_step("Step 5", f"Keyword frequency (wide) saved → {wide_path}", "OK")
        else:
            log_step("Step 5", "No keyword matches found; wide table not created.", "WARNING")
    except Exception as e:
        log_step("Step 5", f"Failed to save keyword frequencies: {e}", "ERROR")

# -----------------------------
# Step 6. Min–max normalization
# -----------------------------
def min_max_normalize(freq_long: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min–max normalization per keyword across all banks/years.
    Input: freq_long with columns [bank, year, group, keyword, rel_freq]
    Output: adds 'norm_freq' column (scaled to [0,1] per keyword)
    """
    log_step("Step 6", "Applying min–max normalization...", "INFO")

    if freq_long.empty:
        log_step("Step 6", "No keyword frequencies available to normalize.", "WARNING")
        return freq_long.assign(norm_freq=[])

    df = freq_long.copy()
    df["norm_freq"] = 0.0

    for (group, keyword), sub in df.groupby(["group", "keyword"]):
        vals = sub["rel_freq"].values
        min_v, max_v = vals.min(), vals.max()
        if max_v > min_v:
            df.loc[sub.index, "norm_freq"] = (sub["rel_freq"] - min_v) / (max_v - min_v)
        else:
            # If all values are the same, set to 0.0
            df.loc[sub.index, "norm_freq"] = 0.0

    log_step("Step 6", f"Normalization completed for {df['keyword'].nunique()} keywords", "OK")
    return df


def save_normalized_freq(df_norm: pd.DataFrame, output_dir: Path) -> None:
    """
    Save normalized keyword frequencies.
    """
    try:
        out_path = output_dir / "keyword_freq_normalized.csv"
        df_norm.to_csv(out_path, index=False, encoding="utf-8")
        log_step("Step 6", f"Normalized keyword frequencies saved → {out_path}", "OK")
    except Exception as e:
        log_step("Step 6", f"Failed to save normalized frequencies: {e}", "ERROR")

# -----------------------------
# Step 7. Entropy & entropy weights
# -----------------------------
import math

def compute_entropy_weights(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute entropy (E_j) and entropy weights (W_j) per keyword, separately within FTII and FTOI.
    Input:
      - df_norm: long table with columns [bank, year, group, keyword, rel_freq, norm_freq]
    Output:
      - DataFrame with columns [group, keyword, entropy, weight_raw, weight_norm]
    """
    log_step("Step 7", "Computing entropy and entropy-weights per keyword...", "INFO")

    required_cols = {"group", "keyword", "norm_freq"}
    if df_norm.empty or not required_cols.issubset(df_norm.columns):
        log_step("Step 7", "Normalized frequency table missing or incomplete; cannot compute entropy.", "ERROR")
        return pd.DataFrame(columns=["group", "keyword", "entropy", "weight_raw", "weight_norm"])

    results = []
    # n is the number of bank-year rows per keyword (use consistent n across keywords)
    # We'll compute per-keyword using its actual count of rows (safer if panel is unbalanced).
    for (grp, kw), sub in df_norm.groupby(["group", "keyword"], sort=False):
        values = sub["norm_freq"].astype(float).values
        n = len(values)
        if n <= 1:
            # Not enough observations to compute dispersion; set max entropy → zero weight
            entropy = 1.0
        else:
            s = values.sum()
            if s <= 0:
                # All zeros → undefined distribution → maximum entropy by convention here
                entropy = 1.0
            else:
                # probabilities
                p = values / s
                # entropy with k = 1/ln(n); zero-prob terms contribute 0
                k = 1.0 / math.log(n)
                entropy_sum = 0.0
                for pi in p:
                    if pi > 0:
                        entropy_sum += pi * math.log(pi)
                entropy = -k * entropy_sum
                # clamp numerical drift
                entropy = max(0.0, min(1.0, entropy))

        # inverse-entropy raw weight
        w_raw = 1.0 - entropy
        results.append({"group": grp, "keyword": kw, "entropy": entropy, "weight_raw": w_raw})

    df_w = pd.DataFrame(results)

    # Normalize weights within each group (FTII separately from FTOI)
    weight_norms = []
    for grp, gsub in df_w.groupby("group", sort=False):
        total = gsub["weight_raw"].sum()
        if total > 0:
            w_norm = gsub["weight_raw"] / total
        else:
            # If all raw weights are zero (e.g., all-zeros keywords), assign equal weights
            equal = 1.0 / max(len(gsub), 1)
            w_norm = pd.Series([equal] * len(gsub), index=gsub.index)
        tmp = gsub.copy()
        tmp["weight_norm"] = w_norm
        weight_norms.append(tmp)

    df_weights = pd.concat(weight_norms, axis=0).reset_index(drop=True)
    log_step("Step 7", f"Computed entropy & weights for {len(df_weights)} keyword(s)", "OK")
    return df_weights


def save_entropy_weights(df_weights: pd.DataFrame, output_dir: Path) -> None:
    """
    Save entropy and normalized weights.
    """
    try:
        out_path = output_dir / "entropy_weights.csv"
        df_weights.to_csv(out_path, index=False, encoding="utf-8")
        log_step("Step 7", f"Entropy weights saved → {out_path}", "OK")
    except Exception as e:
        log_step("Step 7", f"Failed to save entropy weights: {e}", "ERROR")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # Step 1
    DATA_DIR, OUTPUT_DIR = setup_project(BASE_DIR)

    # Step 2
    df_corpus = build_corpus(DATA_DIR)
    save_corpus_csv(df_corpus, OUTPUT_DIR / "corpus.csv")

    # Step 3
    fintech_keywords = get_fintech_keywords()

    # Step 4
    df_clean = preprocess_corpus(df_corpus)
    save_clean_corpus_csv(df_clean, OUTPUT_DIR / "corpus_clean.csv")

    # Step 5
    freq_long, freq_wide = build_keyword_frequencies(df_clean, fintech_keywords)
    save_keyword_freq_outputs(freq_long, freq_wide, OUTPUT_DIR)

    # Step 6
    df_norm = min_max_normalize(freq_long)
    save_normalized_freq(df_norm, OUTPUT_DIR)

    # Step 7
    df_weights = compute_entropy_weights(df_norm)
    save_entropy_weights(df_weights, OUTPUT_DIR)

    # Optional preview
    if not freq_long.empty:
        log_step("Preview", f"First few keyword counts:\n{freq_long.head(8)}", "INFO")
