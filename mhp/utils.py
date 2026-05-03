from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Sequence
from statsmodels.miscmodels.ordinal_model import OrderedResults, OrderedModel
from scipy.stats import chi2_contingency


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


def style_pct(
    df: pd.DataFrame, cols: "all" | list[str] | None = None
) -> pd.Styler:
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


def modal_value(series: pd.Series) -> str:
    return str(series.mode(dropna=True).iloc[0])


def build_analysis_df(
    df: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    cols = [outcome, *predictors, *controls]
    output = df.loc[:, cols].dropna().copy()

    return output


def build_design_matrix(
    df: pd.DataFrame,
    predictors: Sequence[str],
    controls: Sequence[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    x_num = df.loc[:, predictors].astype(float)

    if controls:
        x_cat = pd.get_dummies(
            df.loc[:, controls], drop_first=True, dtype=float
        )
        return pd.concat([x_num, x_cat], axis=1)

    return x_num


def fit_ordinal_logit(
    df: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    controls: Sequence[str] | None = None,
) -> tuple[OrderedResults, pd.DataFrame, list[str]]:
    controls = list(controls or [])
    analysis_df = build_analysis_df(df, outcome, predictors, controls)

    y = analysis_df[outcome].cat.codes
    x = build_design_matrix(analysis_df, predictors, controls)

    model = OrderedModel(y, x, dists="logit")
    result = model.fit(method="bfgs", disp=False)

    return result, analysis_df, x.columns.tolist()


def tidy_or_table(
    result: OrderedResults,
    feature_names: Sequence[str],
    clean_labels: dict[str, str],
) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    slope_idx = [name for name in params.index if name in feature_names]

    output = pd.DataFrame(
        {
            "term": slope_idx,
            "coef_log_odds": params.loc[slope_idx].values,
            "odds_ratio": np.exp(params.loc[slope_idx].values),
            "ci_low": np.exp(conf.loc[slope_idx, 0].values),
            "ci_high": np.exp(conf.loc[slope_idx, 1].values),
            "p_value": pvals.loc[slope_idx].values,
        }
    )

    output["label"] = output["term"].map(clean_labels).fillna(output["term"])
    return output.sort_values("p_value").reset_index(drop=True)


def cramers_v_from_table(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(ct)
    n = float(ct.to_numpy().sum())
    r, k = ct.shape
    if min(r, k) <= 1 or n == 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))
