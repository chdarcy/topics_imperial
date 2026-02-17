"""Rolling-window PCA butterfly strategy on yield curves.

Constructs a 3-tenor butterfly that neutralises level and slope PCs
while maintaining unit exposure to the curvature PC.  PCs are identified
by their loading shape (not assumed to be in order).  Weights are
re-estimated each day using a rolling estimation window.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from pca import pca_on_curves


# ── PC identification ──────────────────────────────────────────────────

def identify_pcs(loadings: pd.DataFrame) -> dict[str, str]:
    """Classify PCs as 'level', 'slope', or 'curvature' by loading shape.

    Uses structural properties (more robust than template correlation):
      - level:     loadings are most uniform (lowest relative variation)
      - slope:     loadings have highest |correlation with maturity|
      - curvature: the remaining PC

    Returns a mapping like {'PC1': 'level', 'PC2': 'slope', 'PC3': 'curvature'}.
    """
    mats = np.array(loadings.index, dtype=float)
    pc_cols = list(loadings.columns)

    # Score each PC for "level-ness": how uniform are the loadings?
    # Use coefficient of variation: lower = more uniform = more level-like
    level_scores = {}
    for pc in pc_cols:
        v = loadings[pc].values
        mean_abs = np.abs(v).mean()
        if mean_abs < 1e-12:
            level_scores[pc] = 999.0  # degenerate
        else:
            level_scores[pc] = np.std(np.abs(v)) / mean_abs

    # Score each PC for "slope-ness": correlation with maturity
    slope_scores = {}
    for pc in pc_cols:
        v = loadings[pc].values
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(v, mats)[0, 1]
        slope_scores[pc] = abs(corr) if not np.isnan(corr) else 0.0

    assignment: dict[str, str] = {}

    # 1. Level = most uniform loadings (lowest variation score)
    remaining = set(pc_cols)
    level_pc = min(remaining, key=lambda pc: level_scores[pc])
    assignment[level_pc] = "level"
    remaining.discard(level_pc)

    # 2. Slope = highest |correlation with maturity| among remaining
    slope_pc = max(remaining, key=lambda pc: slope_scores[pc])
    assignment[slope_pc] = "slope"
    remaining.discard(slope_pc)

    # 3. Curvature = whatever is left
    for pc in remaining:
        assignment[pc] = "curvature"

    return assignment


def _enforce_sign_convention(
    loadings: pd.DataFrame,
    pc_roles: dict[str, str],
) -> pd.DataFrame:
    """Flip eigenvector signs so that loadings have a stable orientation.

    level:     sum of loadings is positive
    slope:     loading increases with maturity (long end > short end)
    curvature: loading at the median maturity is positive (belly is positive)
    """
    loadings = loadings.copy()
    mats = loadings.index.tolist()

    for pc_col, role in pc_roles.items():
        if pc_col not in loadings.columns:
            continue
        v = loadings[pc_col]
        if role == "level" and v.sum() < 0:
            loadings[pc_col] *= -1
        elif role == "slope":
            if v.iloc[-1] - v.iloc[0] < 0:
                loadings[pc_col] *= -1
        elif role == "curvature":
            mid = mats[len(mats) // 2]
            if v.loc[mid] < 0:
                loadings[pc_col] *= -1

    return loadings


# ── Weight solver ──────────────────────────────────────────────────────

def solve_butterfly_weights(
    loadings: pd.DataFrame,
    pc_roles: dict[str, str],
    tenors: tuple[float, float, float] = (3.0, 7.0, 15.0),
) -> np.ndarray | None:
    """Solve the 3x3 system for butterfly weights.

    Given PCA loadings and their role assignments, find weights
    w = [w_short, w_belly, w_long] such that the portfolio has zero
    exposure to level and slope PCs, and unit exposure to curvature.

    Returns None if the system is singular or ill-conditioned.
    """
    # Order columns as [level, slope, curvature]
    role_to_pc = {v: k for k, v in pc_roles.items()}
    ordered_cols = []
    for role in ("level", "slope", "curvature"):
        if role not in role_to_pc:
            return None
        ordered_cols.append(role_to_pc[role])

    tenor_list = list(tenors)

    # L is (3 tenors x 3 PCs) in order [level, slope, curvature]
    L = loadings.loc[tenor_list, ordered_cols].values.astype(float)

    cond = np.linalg.cond(L)
    if cond > 1e10:
        warnings.warn(f"Loading matrix poorly conditioned (cond={cond:.1e})")
        return None

    # b = [0, 0, 1]: zero level, zero slope, unit curvature
    b = np.array([0.0, 0.0, 1.0])

    try:
        w = np.linalg.solve(L.T, b)
    except np.linalg.LinAlgError:
        return None

    return w


# ── Signal-based trading ───────────────────────────────────────────────

def apply_trading_signals(
    strategy_results: dict,
    curves: pd.DataFrame,
    tenors: tuple[float, float, float] = (3.0, 7.0, 15.0),
    zscore_window: int = 63,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
    stop_loss_bps: float = 50.0,
    take_profit_bps: float = 30.0,
    pnl_scale: float = 10_000.0,
) -> dict:
    """Apply z-score mean-reversion signals to the butterfly strategy.

    Tracks the butterfly spread level (w' @ rates) and its rolling z-score.
    Entry when |z| > entry_threshold, exit when |z| < exit_threshold or
    when stop-loss/take-profit hit.

    Returns a dict augmenting strategy_results with signal-based PnL.
    """
    weights = strategy_results["weights"]
    tenor_list = list(tenors)

    # Compute butterfly spread level: w' @ rates at tenors
    spread_records = []
    for date, row in weights.iterrows():
        w = row[["w_3", "w_7", "w_15"]].values.astype(float)
        if date in curves.index:
            rates = curves.loc[date, tenor_list].values.astype(float)
            spread_val = float(np.dot(w, rates)) * pnl_scale
            spread_records.append({"date": date, "spread": spread_val})

    spread_df = pd.DataFrame(spread_records).set_index("date")
    spread = spread_df["spread"]

    # Rolling z-score of the spread
    rolling_mean = spread.rolling(zscore_window, min_periods=20).mean()
    rolling_std = spread.rolling(zscore_window, min_periods=20).std()
    zscore = (spread - rolling_mean) / rolling_std

    # Generate positions: +1 = long butterfly, -1 = short, 0 = flat
    position = pd.Series(0.0, index=weights.index)
    trade_pnl = pd.Series(0.0, index=strategy_results["daily_pnl"].index)
    raw_pnl = strategy_results["daily_pnl"]

    cumulative_trade_pnl = 0.0
    entry_pnl = 0.0  # cumulative PnL since last entry

    dates = weights.index.tolist()
    pnl_dates = raw_pnl.index.tolist()

    prev_pos = 0.0

    for idx, date in enumerate(dates):
        z = zscore.get(date, np.nan)
        if np.isnan(z):
            position.iloc[idx] = prev_pos
            continue

        new_pos = prev_pos

        if prev_pos == 0:
            # Entry logic
            if z > entry_threshold:
                new_pos = -1.0  # spread is rich, sell butterfly
            elif z < -entry_threshold:
                new_pos = 1.0   # spread is cheap, buy butterfly
        else:
            # Exit logic: z-score reversion
            if prev_pos > 0 and z > -exit_threshold:
                new_pos = 0.0
            elif prev_pos < 0 and z < exit_threshold:
                new_pos = 0.0

            # Stop-loss / take-profit on trade PnL
            if prev_pos != 0 and abs(entry_pnl) > 0:
                if entry_pnl < -stop_loss_bps:
                    new_pos = 0.0
                elif entry_pnl > take_profit_bps:
                    new_pos = 0.0

        position.iloc[idx] = new_pos

        # Apply position to next day's PnL
        if idx < len(pnl_dates):
            pnl_date = pnl_dates[idx] if idx < len(pnl_dates) else None
            if pnl_date is not None and pnl_date in raw_pnl.index:
                daily = raw_pnl.loc[pnl_date] * new_pos
                trade_pnl.loc[pnl_date] = daily
                if new_pos != 0:
                    entry_pnl += daily
                if new_pos == 0 and prev_pos != 0:
                    entry_pnl = 0.0  # reset on exit
                if prev_pos == 0 and new_pos != 0:
                    entry_pnl = 0.0  # reset on new entry

        prev_pos = new_pos

    return {
        **strategy_results,
        "signal_position": position,
        "signal_daily_pnl": trade_pnl,
        "signal_cumulative_pnl": trade_pnl.cumsum(),
        "spread": spread,
        "zscore": zscore,
    }


# ── Rolling PCA butterfly (main loop) ─────────────────────────────────

def rolling_pca_butterfly(
    curves: pd.DataFrame,
    window: int = 252,
    tenors: tuple[float, float, float] = (3.0, 7.0, 15.0),
    n_components: int = 3,
    standardize: bool = True,
    pnl_scale: float = 10_000.0,
) -> dict:
    """Run rolling-window PCA and compute butterfly strategy PnL.

    PCs are identified by loading shape each window (not assumed in order).

    Parameters
    ----------
    curves : pd.DataFrame
        Interpolated yield curves (dates x maturities in decimal).
    window : int
        Estimation window in business days.
    tenors : tuple
        Butterfly tenors (short wing, belly, long wing).
    n_components : int
        Number of PCA components to extract.
    standardize : bool
        Whether to z-score before PCA.
    pnl_scale : float
        Multiplier for PnL (10_000 converts decimal to bps).

    Returns
    -------
    dict with keys: weights, daily_pnl, cumulative_pnl, rolling_loadings,
    diagnostics, pc_scores.
    """
    dates = curves.index.tolist()
    n = len(dates)
    tenor_list = list(tenors)

    weights_records: list[dict] = []
    pnl_records: list[dict] = []
    score_records: list[dict] = []
    loadings_samples: dict[pd.Timestamp, pd.DataFrame] = {}
    diag_records: list[dict] = []

    skipped = 0

    for i in range(window, n - 1):
        window_curves = curves.iloc[i - window: i + 1]

        try:
            res = pca_on_curves(
                window_curves, n_components=n_components, standardize=standardize
            )
        except ValueError:
            skipped += 1
            continue

        loadings = res["loadings"]
        scores = res["scores"]

        # Identify which PC is level, slope, curvature
        pc_roles = identify_pcs(loadings)
        if "curvature" not in pc_roles.values():
            skipped += 1
            continue

        loadings = _enforce_sign_convention(loadings, pc_roles)

        # Check that all butterfly tenors are in the loadings index
        if not all(t in loadings.index for t in tenor_list):
            skipped += 1
            continue

        w = solve_butterfly_weights(loadings, pc_roles=pc_roles, tenors=tenors)
        if w is None:
            skipped += 1
            continue

        trade_date = dates[i]
        pnl_date = dates[i + 1]

        # Daily rate change at butterfly tenors
        delta = curves.loc[pnl_date, tenor_list] - curves.loc[trade_date, tenor_list]
        if delta.isna().any():
            skipped += 1
            continue

        daily_pnl = float(np.dot(w, delta.values)) * pnl_scale

        weights_records.append(
            {"date": trade_date, "w_3": w[0], "w_7": w[1], "w_15": w[2]}
        )
        pnl_records.append({"date": pnl_date, "daily_pnl": daily_pnl})

        # Store the last day's PC scores for factor correlation analysis
        # Project the pnl_date's rate change onto the full loading vectors
        role_to_pc = {v: k for k, v in pc_roles.items()}
        if pnl_date in curves.index and trade_date in curves.index:
            full_delta = (curves.loc[pnl_date] - curves.loc[trade_date]).values
            # Only use maturities present in loadings
            common_mats = [m for m in curves.columns if m in loadings.index]
            if len(common_mats) == len(loadings.index):
                full_delta_aligned = (
                    curves.loc[pnl_date, common_mats] - curves.loc[trade_date, common_mats]
                ).values * pnl_scale
                level_pc = role_to_pc.get("level", "PC1")
                slope_pc = role_to_pc.get("slope", "PC2")
                curv_pc = role_to_pc.get("curvature", "PC3")
                score_records.append({
                    "date": pnl_date,
                    "level_score": float(full_delta_aligned @ loadings[level_pc].values),
                    "slope_score": float(full_delta_aligned @ loadings[slope_pc].values),
                    "curvature_score": float(full_delta_aligned @ loadings[curv_pc].values),
                })

        # Store loadings periodically (every 21 biz days ~ monthly)
        if len(loadings_samples) == 0 or (i - window) % 21 == 0:
            loadings_samples[trade_date] = loadings

        # Diagnostics
        L_sub = loadings.loc[tenor_list].values.astype(float)
        cond = np.linalg.cond(L_sub)

        role_to_pc = {v: k for k, v in pc_roles.items()}
        level_pc = role_to_pc.get("level", "PC1")
        slope_pc = role_to_pc.get("slope", "PC2")
        curv_pc = role_to_pc.get("curvature", "PC3")

        diag_records.append(
            {
                "date": trade_date,
                "cond_number": cond,
                "expl_var_pc1": res["explained_variance_ratio"][0],
                "expl_var_pc2": res["explained_variance_ratio"][1],
                "expl_var_pc3": res["explained_variance_ratio"][2],
                "level_pc": level_pc,
                "slope_pc": slope_pc,
                "curvature_pc": curv_pc,
                # Loadings at butterfly tenors (using identified roles)
                "level_3y": loadings.loc[tenors[0], level_pc],
                "level_7y": loadings.loc[tenors[1], level_pc],
                "level_15y": loadings.loc[tenors[2], level_pc],
                "slope_3y": loadings.loc[tenors[0], slope_pc],
                "slope_7y": loadings.loc[tenors[1], slope_pc],
                "slope_15y": loadings.loc[tenors[2], slope_pc],
                "curv_3y": loadings.loc[tenors[0], curv_pc],
                "curv_7y": loadings.loc[tenors[1], curv_pc],
                "curv_15y": loadings.loc[tenors[2], curv_pc],
            }
        )

    if skipped:
        warnings.warn(f"Skipped {skipped} dates due to PCA failures or NaN data")

    weights_df = pd.DataFrame(weights_records).set_index("date")
    pnl_df = pd.DataFrame(pnl_records).set_index("date")
    daily_pnl = pnl_df["daily_pnl"]
    cumulative_pnl = daily_pnl.cumsum()
    diagnostics = pd.DataFrame(diag_records).set_index("date")
    pc_scores = pd.DataFrame(score_records).set_index("date") if score_records else pd.DataFrame()

    return {
        "weights": weights_df,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "rolling_loadings": loadings_samples,
        "diagnostics": diagnostics,
        "pc_scores": pc_scores,
    }
