from __future__ import annotations
import pandas as pd
import re

def column_summary(series: pd.Series) -> str:
    col = series.name
    miss = int(series.isna().sum())
    n = len(series)
    miss_pct = 100.0 * miss / n if n else 0.0
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(series):
        kind = "enum (boolean)"
        nu = int(series.nunique(dropna=True))
        detail = f"{nu} distinct labels"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
        s = series.dropna()
        if s.empty:
            detail = "no non-missing values"
        else:
            detail = f"range [{s.min()}, {s.max()}]"
    else:
        kind = "enum"
        nu = int(series.nunique(dropna=True))
        detail = f"{nu} distinct labels"

    return (
        f"{col}\n"
        f"\tdtype={dtype} | {kind} | {detail}\n"
        f"\tmissing: {miss} ({miss_pct:.2f}%)\n"
    )

def print_dataset_summary(key: str, title: str, df: pd.DataFrame) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n{key} — {title}\n{sep}")
    print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    print("Column names:")
    for i, c in enumerate(df.columns, 1):
        print(f"\t{i}. {c}")
    print()

    miss_per_col = df.isna().sum()
    cols_with_miss = miss_per_col[miss_per_col > 0].sort_values(ascending=False)
    if cols_with_miss.empty:
        print("Missing values: none\n")
    else:
        print("Missing values (by column):")
        for c, cnt in cols_with_miss.items():
            pct = 100.0 * cnt / len(df)
            print(f"\t{c}: {cnt} ({pct:.2f}%)")
        print(f"\tTotal cells missing: {df.isna().sum().sum()}\n")

    print("Per-column type-value summary:")
    for c in df.columns:
        print(column_summary(df[c]), end="")

def extract_number(string: str) -> float | None:
    if pd.isna(string):
        return None

    clean_string = string.strip().lower()

    # \d+      : 1 or more digits
    # [.,:]    : followed by a dot, comma, or colon
    # \d+      : followed by 1 or more digits
    # |\d+     : OR just a standard whole number (fallback)
    match = re.search(r'\d+[.,:]\d+|\d+', clean_string)
    if match:
        num_str = match.group().replace(",", ".").replace(":", ".")
        return float(num_str)
    else:
        return None
        