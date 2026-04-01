from __future__ import annotations
from typing import Iterable
import pandas as pd


# Map Bloomberg column names to maturities in years
TENOR_MAP = {
    # Short end (weeks / months)
    "BPSWS1Z Curncy": 1.0 / 52.0,      # 1 week
    "BPSWS2Z Curncy": 2.0 / 52.0,      # 2 weeks
    "BPSWSA Curncy": 1.0 / 12.0,       # 1 month
    "BPSWSB Curncy": 2.0 / 12.0,       # 2 months
    # BPSWSC (3mo) and BPSWSD (4mo) not present in dataset

    "BPSWSE Curncy": 5.0 / 12.0,       # 5 months
    "BPSWSF Curncy": 6.0 / 12.0,       # 6 months
    "BPSWSG Curncy": 7.0 / 12.0,       # 7 months
    "BPSWSH Curncy": 8.0 / 12.0,       # 8 months
    "BPSWSI Curncy": 9.0 / 12.0,       # 9 months
    "BPSWSJ Curncy": 10.0 / 12.0,      # 10 months
    "BPSWSK Curncy": 11.0 / 12.0,      # 11 months

    # Standard yearly tenors (MAIN PCA INPUT)
    "BPSWS1 Curncy": 1.0,
    "BPSWS1F Curncy": 18.0 / 12.0,    # 18 months
    "BPSWS2 Curncy": 2.0,
    "BPSWS3 Curncy": 3.0,
    "BPSWS4 Curncy": 4.0,
    "BPSWS5 Curncy": 5.0,
    "BPSWS6 Curncy": 6.0,
    "BPSWS7 Curncy": 7.0,
    "BPSWS8 Curncy": 8.0,
    "BPSWS9 Curncy": 9.0,
    "BPSWS10 Curncy": 10.0,
    "BPSWS12 Curncy": 12.0,
    "BPSWS15 Curncy": 15.0,
    "BPSWS20 Curncy": 20.0,
    "BPSWS25 Curncy": 25.0,
    "BPSWS30 Curncy": 30.0,
    "BPSWS40 Curncy": 40.0,
    "BPSWS50 Curncy": 50.0,
}

def interpolate_to_grid(
    df: pd.DataFrame,
    target_maturities: Iterable[float],
    *,
    scale: float = 0.01,
    min_valid: int = 10,
) -> pd.DataFrame:
    
    tenor_map = TENOR_MAP

    # Deduplicate and sort index
    df = df.copy()
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    source_cols = [c for c in df.columns if c in tenor_map]
    col_to_mat = {c: tenor_map[c] for c in source_cols}

    rates = df[source_cols].astype(float) * scale
    rates.columns = [col_to_mat[c] for c in source_cols]
    rates = rates.reindex(sorted(rates.columns), axis=1)

    # Drop rows that have fewer than `min_valid` non-NaN source points
    valid_count = rates.notna().sum(axis=1)
    rates = rates.loc[valid_count >= min_valid]

    target = sorted(target_maturities)
    all_cols = sorted(set(rates.columns).union(target))
    rates_reindexed = rates.reindex(columns=all_cols)

    rates_interp = rates_reindexed.interpolate(axis=1, method="linear", limit_area="inside")

    return rates_interp[target]


def compute_forward_rates(spot_curves: pd.DataFrame) -> pd.DataFrame:
    """Compute forward rates between adjacent maturities on the grid """
    
    mats = sorted(float(c) for c in spot_curves.columns)
    result = pd.DataFrame(index=spot_curves.index)

    for i in range(1, len(mats)):
        t_prev, t_cur = mats[i - 1], mats[i]
        dt = t_cur - t_prev
        result[float(t_cur)] = (
            t_cur * spot_curves[t_cur] - t_prev * spot_curves[t_prev]
        ) / dt

    return result
