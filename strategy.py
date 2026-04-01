from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from pca import pca_on_curves


# ── PC identification ──────────────────────────────────────────────────

def identify_pcs(loadings: pd.DataFrame) -> dict[str, str]:
    """Classify PCs as 'level', 'slope', or 'curvature' by loading shape """
    
    mats = np.array(loadings.index, dtype=float)
    pc_cols = list(loadings.columns)

    # Score each PC for "level-ness": how uniform are the loadings?
    level_scores = {pc: np.std(np.abs(loadings[pc].values)) / np.abs(loadings[pc].values).mean() for pc in pc_cols}

    # Score each PC for "slope-ness": correlation with maturity
    slope_scores = {pc: abs(np.corrcoef(loadings[pc].values, mats)[0, 1]) for pc in pc_cols}

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
    """Flip eigenvector signs. level: +, slope:long end > short end, curvature: belly +) """
    
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
    tenors: tuple[float, float, float],
) -> np.ndarray | None:

    # Order columns as [level, slope, curvature]
    role_to_pc = {v: k for k, v in pc_roles.items()}
    ordered_cols = [role_to_pc[role] for role in ("level", "slope", "curvature")]
    tenor_list = list(tenors)
    # L is (3 tenors x 3 PCs) in order [level, slope, curvature]
    L = loadings.loc[tenor_list, ordered_cols].values.astype(float)
    b = np.array([0.0, 0.0, 1.0])
    w = np.linalg.solve(L.T, b)
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
    """Apply z-score mean-reversion signals to the butterfly strategy"""

    weights = strategy_results["weights"]
    tenor_list = list(tenors)
    w_keys = [f"w_{t:g}" for t in tenors]

    # Compute butterfly spread level: w' @ rates at tenors
    spread_records = []
    for date, row in weights.iterrows():
        w = row[w_keys].values.astype(float)
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


# ── Strategy variants ─────────────────────────────────────────────────

def apply_vol_scaling(
    strategy_results: dict,
    vol_window: int = 63,
    vol_target_bps: float = 5.0,
) -> dict:
    """Volatility-scaled butterfly: size the position inversely proportional
    to recent realised volatility so that the ex-ante daily risk is constant.

    Literature: standard in rates RV desks (Golub & Tilman 2000, Ch.6;
    Hariparsad & Maré 2023 use duration-neutral + vol-normalised sizing).
    """
    raw_pnl = strategy_results["daily_pnl"]
    rolling_vol = raw_pnl.rolling(vol_window, min_periods=20).std()

    scale = vol_target_bps / rolling_vol.replace(0, np.nan)
    scale = scale.clip(upper=5.0)  # cap leverage at 5x

    scaled_pnl = raw_pnl * scale.shift(1)  # use lagged vol
    scaled_pnl = scaled_pnl.dropna()

    return {
        **strategy_results,
        "variant_daily_pnl": scaled_pnl,
        "variant_cumulative_pnl": scaled_pnl.cumsum(),
        "variant_scale": scale,
    }


def apply_momentum_signal(
    strategy_results: dict,
    lookback: int = 21,
    holding_period: int = 5,
) -> dict:
    """Curvature momentum / trend-following: go with the recent trend
    in the butterfly spread rather than betting on mean-reversion.

    If the raw butterfly PnL has been positive over the last `lookback`
    days, stay long; if negative, go short. Rebalance every
    `holding_period` days.

    Literature: Dreher, Gräb & Kostka (2020) "From carry trades to curvy
    trades" show curvature factor has positive autocorrelation at short
    horizons in EUR; Suimon et al. (2020) use trend signals on JGB.
    """
    raw_pnl = strategy_results["daily_pnl"]
    cumret = raw_pnl.rolling(lookback, min_periods=10).sum()

    # Signal: +1 if trailing PnL > 0, -1 otherwise
    signal = np.sign(cumret)

    # Only rebalance every `holding_period` days to reduce turnover
    held_signal = signal.copy()
    for i in range(len(held_signal)):
        if i % holding_period != 0:
            held_signal.iloc[i] = held_signal.iloc[i - 1] if i > 0 else 0.0

    mom_pnl = raw_pnl * held_signal.shift(1)
    mom_pnl = mom_pnl.dropna()

    return {
        **strategy_results,
        "variant_daily_pnl": mom_pnl,
        "variant_cumulative_pnl": mom_pnl.cumsum(),
        "variant_signal": held_signal,
    }


def apply_carry_overlay(
    strategy_results: dict,
    curves: pd.DataFrame,
    tenors: tuple[float, float, float] = (3.0, 7.0, 15.0),
    carry_window: int = 21,
    pnl_scale: float = 10_000.0,
) -> dict:
    """Carry-adjusted curvature: combine the z-score mean-reversion signal
    with a carry (roll-down) signal.  Only enter when both signals agree.

    The carry of the butterfly is estimated as the 21-day trailing average
    daily PnL — a positive carry means curvature is earning positive theta.

    Literature: Dreher, Gräb & Kostka (2020) "curvy trades" combine carry
    and value signals on the curvature factor; De Vere (2021) combines
    macro views with butterfly carry in UST.
    """
    raw_pnl = strategy_results["daily_pnl"]

    # Carry signal: sign of trailing average PnL
    carry = raw_pnl.rolling(carry_window, min_periods=10).mean()
    carry_signal = np.sign(carry)

    # Z-score mean-reversion signal (reuse spread zscore)
    weights = strategy_results["weights"]
    tenor_list = list(tenors)
    w_keys = [f"w_{t:g}" for t in tenors]

    spread_records = []
    for date, row in weights.iterrows():
        w = row[w_keys].values.astype(float)
        if date in curves.index:
            rates = curves.loc[date, tenor_list].values.astype(float)
            spread_val = float(np.dot(w, rates)) * pnl_scale
            spread_records.append({"date": date, "spread": spread_val})

    spread_df = pd.DataFrame(spread_records).set_index("date")
    spread = spread_df["spread"]
    rolling_mean = spread.rolling(63, min_periods=20).mean()
    rolling_std = spread.rolling(63, min_periods=20).std()
    zscore = (spread - rolling_mean) / rolling_std

    # Reversion signal: sell if z > 1, buy if z < -1
    mr_signal = pd.Series(0.0, index=zscore.index)
    mr_signal[zscore > 1.0] = -1.0
    mr_signal[zscore < -1.0] = 1.0

    # Combined: trade only when carry and reversion agree
    # If they disagree, stay flat (conservative)
    combined = pd.Series(0.0, index=raw_pnl.index)
    common = combined.index.intersection(carry_signal.index).intersection(mr_signal.index)
    for dt in common:
        c = carry_signal.get(dt, 0)
        m = mr_signal.get(dt, 0)
        if c != 0 and m != 0 and np.sign(c) == np.sign(m):
            combined[dt] = m
        elif m != 0:
            combined[dt] = m * 0.5  # half size if only reversion agrees

    carry_pnl = raw_pnl * combined.shift(1)
    carry_pnl = carry_pnl.dropna()

    return {
        **strategy_results,
        "variant_daily_pnl": carry_pnl,
        "variant_cumulative_pnl": carry_pnl.cumsum(),
        "variant_carry_signal": carry_signal,
        "variant_mr_signal": mr_signal,
    }


# ── Rolling PCA butterfly (main loop) ─────────────────────────────────

def rolling_pca_butterfly(
    curves: pd.DataFrame,
    window: int = 252,
    tenors: tuple[float, float, float] = (3.0, 7.0, 15.0),
    n_components: int = 3,
    pnl_scale: float = 10_000.0,
) -> dict:
    """Run rolling-window PCA and compute butterfly strategy PnL """
    dates = curves.index.tolist()
    n = len(dates)
    tenor_list = list(tenors)

    weights_records: list[dict] = []
    pnl_records: list[dict] = []
    score_records: list[dict] = []
    loadings_samples: dict[pd.Timestamp, pd.DataFrame] = {}
    diag_records: list[dict] = []

    for i in range(window, n - 1):
        window_curves = curves.iloc[i - window: i + 1]
        res = pca_on_curves(window_curves, n_components=n_components)
        loadings = res["loadings"]
        pc_roles = identify_pcs(loadings)
        loadings = _enforce_sign_convention(loadings, pc_roles)
        w = solve_butterfly_weights(loadings, pc_roles=pc_roles, tenors=tenors)
        trade_date = dates[i]
        pnl_date = dates[i + 1]
        delta = curves.loc[pnl_date, tenor_list] - curves.loc[trade_date, tenor_list]
        daily_pnl = float(np.dot(w, delta.values)) * pnl_scale
        w_keys = [f"w_{t:g}" for t in tenors]
        weights_records.append({"date": trade_date, **dict(zip(w_keys, w))})
        pnl_records.append({"date": pnl_date, "daily_pnl": daily_pnl})
        role_to_pc = {v: k for k, v in pc_roles.items()}
        common_mats = [m for m in curves.columns if m in loadings.index]
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
        if len(loadings_samples) == 0 or (i - window) % 21 == 0:
            loadings_samples[trade_date] = loadings
        L_sub = loadings.loc[tenor_list].values.astype(float)
        cond = np.linalg.cond(L_sub)
        tenor_labels = [f"{t:g}y" for t in tenors]
        diag_records.append({
            "date": trade_date,
            "cond_number": cond,
            "expl_var_pc1": res["explained_variance_ratio"][0],
            "expl_var_pc2": res["explained_variance_ratio"][1],
            "expl_var_pc3": res["explained_variance_ratio"][2],
            "level_pc": level_pc,
            "slope_pc": slope_pc,
            "curvature_pc": curv_pc,
            f"level_{tenor_labels[0]}": loadings.loc[tenors[0], level_pc],
            f"level_{tenor_labels[1]}": loadings.loc[tenors[1], level_pc],
            f"level_{tenor_labels[2]}": loadings.loc[tenors[2], level_pc],
            f"slope_{tenor_labels[0]}": loadings.loc[tenors[0], slope_pc],
            f"slope_{tenor_labels[1]}": loadings.loc[tenors[1], slope_pc],
            f"slope_{tenor_labels[2]}": loadings.loc[tenors[2], slope_pc],
            f"curv_{tenor_labels[0]}": loadings.loc[tenors[0], curv_pc],
            f"curv_{tenor_labels[1]}": loadings.loc[tenors[1], curv_pc],
            f"curv_{tenor_labels[2]}": loadings.loc[tenors[2], curv_pc],
        })
    weights_df = pd.DataFrame(weights_records).set_index("date")
    pnl_df = pd.DataFrame(pnl_records).set_index("date")
    daily_pnl = pnl_df["daily_pnl"]
    cumulative_pnl = daily_pnl.cumsum()
    diagnostics = pd.DataFrame(diag_records).set_index("date")
    pc_scores = pd.DataFrame(score_records).set_index("date")
    return {
        "weights": weights_df,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
        "rolling_loadings": loadings_samples,
        "diagnostics": diagnostics,
        "pc_scores": pc_scores,
    }
