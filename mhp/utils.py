from __future__ import annotations

import re
import pandas as pd


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


def style_pct(df: pd.DataFrame, cols: "all" | list[str] | None = None) -> pd.Styler:
    if cols is None:
        cols = [col for col in df.columns if "pct" in str(col).lower()]

    if cols == "all":
        return df.style.format("{:.2f}%")

    return df.style.format("{:.2f}%", subset=cols)


def extract_number(string: str) -> float | None:
    if pd.isna(string):
        return None

    clean_string = string.strip().lower()
    match = re.search(r"\d+[.,:]\d+|\d+", clean_string)
    if match:
        num_str = match.group().replace(",", ".").replace(":", ".")
        return float(num_str)
    return None


def age_int_to_bracket(age: int) -> str:
    if age < 18:
        return "<18"
    elif age <= 22:
        return "18-22"
    elif age <= 26:
        return "23-26"
    elif age <= 30:
        return "27-30"
    else:
        return ">30"


def cgpa_float_to_band(cgpa: float | None) -> str:
    if pd.isna(cgpa) or cgpa <= 0 or cgpa > 4.0:
        return "other"
    elif cgpa < 2.5:
        return "<2.50"
    elif cgpa < 3.0:
        return "2.50-2.99"
    elif cgpa < 3.4:
        return "3.00-3.39"
    elif cgpa < 3.8:
        return "3.40-3.79"
    else:
        return "3.80-4.00"


def semester_int_to_academic_year(semester: int | None) -> str | None:
    if pd.isna(semester):
        return "unknown"
    elif semester <= 2:
        return "first"
    elif semester <= 4:
        return "second"
    elif semester <= 6:
        return "third"
    elif semester <= 8:
        return "fourth"
    else:
        return "other"


def dass21_depression_score_to_level(score: int | None) -> str | None:
    if pd.isna(score):
        return None
    elif score == 0:
        return "none"
    elif score <= 9:
        return "minimal"
    elif score <= 13:
        return "mild"
    elif score <= 20:
        return "moderate"
    elif score <= 27:
        return "moderately_severe"
    else:
        return "severe"


def dass21_anxiety_score_to_level(score: int | None) -> str | None:
    if pd.isna(score):
        return None
    elif score == 0:
        return "none"
    elif score <= 7:
        return "minimal"
    elif score <= 9:
        return "mild"
    elif score <= 14:
        return "moderate"
    elif score <= 19:
        return "moderately_severe"
    else:
        return "severe"


def dass21_stress_score_to_level(score: int | None) -> str | None:
    if pd.isna(score):
        return None
    elif score == 0:
        return "none"
    elif score <= 14:
        return "minimal"
    elif score <= 18:
        return "mild"
    elif score <= 25:
        return "moderate"
    elif score <= 33:
        return "moderately_severe"
    else:
        return "severe"

