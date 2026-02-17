from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# Map Bloomberg column names to maturities in years
TENOR_MAP = {
    # Short end (weeks / months)
    "BPSWS1Z Curncy": 1 / 52,      # 1 week
    "BPSWS2Z Curncy": 2 / 52,      # 2 weeks
    "BPSWSA Curncy": 1 / 12,       # 1 month
    "BPSWSB Curncy": 2 / 12,       # 2 months
    "BPSWSC Curncy": 3 / 12,       # 3 months
    "BPSWSD Curncy": 4 / 12,       # 4 months
    "BPSWSE Curncy": 5 / 12,       # 5 months
    "BPSWSF Curncy": 6 / 12,       # 6 months
    "BPSWSG Curncy": 7 / 12,       # 7 months
    "BPSWSH Curncy": 8 / 12,       # 8 months
    "BPSWSI Curncy": 9 / 12,       # 9 months
    "BPSWSJ Curncy": 10 / 12,      # 10 months
    "BPSWSK Curncy": 11 / 12,      # 11 months

    # Standard yearly tenors (MAIN PCA INPUT)
    "BPSWS1 Curncy": 1,
    "BPSWS2 Curncy": 2,
    "BPSWS3 Curncy": 3,
    "BPSWS4 Curncy": 4,
    "BPSWS5 Curncy": 5,
    "BPSWS6 Curncy": 6,
    "BPSWS7 Curncy": 7,
    "BPSWS8 Curncy": 8,
    "BPSWS9 Curncy": 9,
    "BPSWS10 Curncy": 10,
    "BPSWS12 Curncy": 12,
    "BPSWS15 Curncy": 15,
    "BPSWS20 Curncy": 20,
    "BPSWS25 Curncy": 25,
    "BPSWS30 Curncy": 30,
    "BPSWS40 Curncy": 40,
    "BPSWS50 Curncy": 50,
}


def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    # If already has datetime index, return.
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Prefer explicit datetime-typed column if present.
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"])
    if len(dt_cols.columns) > 0:
        c = dt_cols.columns[0]
        df2 = df.copy()
        df2.index = pd.to_datetime(df2[c])
        return df2.drop(columns=[c])

    # Fallback: try parsing the first column to datetimes.
    first = df.columns[0]
    parsed = pd.to_datetime(df[first], errors="coerce")
    if parsed.notna().any():
        df2 = df.copy()
        df2.index = parsed
        return df2.drop(columns=[first])

    return df


def interpolate_to_grid(
    df: pd.DataFrame,
    target_maturities: Iterable[float] = (1, 2, 3, 5, 7, 10, 15, 20, 30),
    tenor_map: dict | None = None,
    *,
    scale: float = 0.01,
    allow_extrapolation: bool = False,
    min_valid: int = 10,
) -> pd.DataFrame:
    """Vectorised interpolation onto target maturities.

    Drops rows that have fewer than ``min_valid`` non-NaN source rates (they
    cannot be reliably interpolated). Returns a DataFrame indexed by date with
    float columns matching target maturities.
    """
    if tenor_map is None:
        tenor_map = TENOR_MAP

    df = df.copy()
    df = _ensure_date_index(df)
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    source_cols = [c for c in df.columns if c in tenor_map]
    if not source_cols:
        raise ValueError("No known tenor columns found in input DataFrame.")

    col_to_mat = {c: float(tenor_map[c]) for c in source_cols}
    rates = df[source_cols].astype(float) * float(scale)
    rates.columns = [col_to_mat[c] for c in source_cols]
    rates = rates.reindex(sorted(rates.columns), axis=1)

    # Drop rows that have fewer than `min_valid` non-NaN source points
    valid_count = rates.notna().sum(axis=1)
    if (valid_count < min_valid).any():
        rates = rates.loc[valid_count >= min_valid]

    target = sorted(float(x) for x in target_maturities)
    all_cols = sorted(set(rates.columns).union(target))

    rates_reindexed = rates.reindex(columns=all_cols)
    if allow_extrapolation:
        rates_interp = rates_reindexed.interpolate(axis=1, method="linear", limit_direction="both")
    else:
        rates_interp = rates_reindexed.interpolate(axis=1, method="linear", limit_area="inside")

    return rates_interp[target]


def compute_forward_rates(spot_curves: pd.DataFrame) -> pd.DataFrame:
    """Compute forward rates between adjacent maturities on the grid.

    For consecutive maturities T_{i-1} and T_i on the grid, the forward
    rate over [T_{i-1}, T_i] is:

        f_i = (T_i * S(T_i) - T_{i-1} * S(T_{i-1})) / (T_i - T_{i-1})

    This uses the linear zero-rate approximation appropriate for OIS rates.

    Parameters
    ----------
    spot_curves : pd.DataFrame
        Interpolated spot curves (dates x maturities in decimal).

    Returns
    -------
    pd.DataFrame of forward rates, indexed by date, columns are the
    right-endpoint maturity of each forward interval.
    """
    mats = sorted(float(c) for c in spot_curves.columns)
    result = pd.DataFrame(index=spot_curves.index)

    for i in range(1, len(mats)):
        t_prev, t_cur = mats[i - 1], mats[i]
        dt = t_cur - t_prev
        result[t_cur] = (
            t_cur * spot_curves[t_cur] - t_prev * spot_curves[t_prev]
        ) / dt

    return result
