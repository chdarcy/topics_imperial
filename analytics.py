"""Performance analytics and plotting for the PCA butterfly strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


# BOE regime boundaries for shading
BOE_REGIMES = [
    ("2023-01-01", "2023-08-03", "Hiking", "#ffcccc"),
    ("2023-08-03", "2024-08-01", "Hold", "#cce5ff"),
    ("2024-08-01", "2026-02-01", "Cutting", "#ccffcc"),
]


def compute_performance_metrics(
    daily_pnl: pd.Series,
    annualization_factor: int = 252,
) -> dict:
    """Compute strategy performance statistics."""
    pnl = daily_pnl.dropna()
    if len(pnl) == 0:
        return {}

    total = pnl.sum()
    avg = pnl.mean()
    vol = pnl.std(ddof=1)
    ann_return = avg * annualization_factor
    ann_vol = vol * np.sqrt(annualization_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = running_max - cum
    max_dd = drawdown.max()
    if max_dd > 0:
        dd_end_idx = drawdown.idxmax()
        dd_start_idx = cum.loc[:dd_end_idx].idxmax()
    else:
        dd_start_idx = dd_end_idx = pnl.index[0]

    return {
        "total_pnl_bps": total,
        "avg_daily_pnl_bps": avg,
        "annualized_return_bps": ann_return,
        "annualized_vol_bps": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown_bps": max_dd,
        "max_drawdown_start": dd_start_idx,
        "max_drawdown_end": dd_end_idx,
        "hit_rate": (pnl > 0).mean(),
        "num_days": len(pnl),
        "skewness": float(pnl.skew()),
        "kurtosis": float(pnl.kurtosis()),
    }


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover as sum of absolute weight changes."""
    return weights.diff().abs().sum(axis=1)


def generate_trade_log(
    weights: pd.DataFrame,
    daily_pnl: pd.Series,
) -> pd.DataFrame:
    """Create a trade log with weights, PnL, and turnover."""
    log = weights.copy()
    log["turnover"] = compute_turnover(weights)
    log["daily_pnl_bps"] = daily_pnl.reindex(log.index)
    if log["daily_pnl_bps"].isna().all():
        log["daily_pnl_bps"] = daily_pnl.values[: len(log)]
    log["cumulative_pnl_bps"] = log["daily_pnl_bps"].cumsum()
    log.index.name = "date"
    return log


# ── Factor correlation analysis (requirement 6) ───────────────────────

def factor_correlation_analysis(
    daily_pnl: pd.Series,
    pc_scores: pd.DataFrame,
) -> dict:
    """Analyse correlation between strategy PnL and PC factor scores.

    The PnL should be uncorrelated with level/slope scores and strongly
    correlated with curvature scores.

    Returns a dict with correlations, R-squared from regression, and
    a summary DataFrame.
    """
    # Align on common dates
    common = daily_pnl.index.intersection(pc_scores.index)
    pnl = daily_pnl.loc[common].values
    scores = pc_scores.loc[common]

    if len(common) < 10:
        return {"error": "Too few overlapping dates for correlation analysis"}

    correlations = {}
    for col in scores.columns:
        s = scores[col].values
        corr = np.corrcoef(pnl, s)[0, 1]
        correlations[col] = corr

    # OLS regression: PnL = a + b1*level + b2*slope + b3*curvature + epsilon
    X = scores.values
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        betas, residuals, _, _ = np.linalg.lstsq(X_with_const, pnl, rcond=None)
        y_hat = X_with_const @ betas
        ss_res = np.sum((pnl - y_hat) ** 2)
        ss_tot = np.sum((pnl - pnl.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        betas = np.zeros(X_with_const.shape[1])
        r_squared = 0.0

    return {
        "correlations": correlations,
        "r_squared": r_squared,
        "betas": {
            "intercept": betas[0],
            **{col: betas[i + 1] for i, col in enumerate(scores.columns)},
        },
        "n_obs": len(common),
    }


# ── Plotting helpers ──────────────────────────────────────────────────

def _add_regime_shading(ax: plt.Axes) -> None:
    """Add BOE regime shading to an axes."""
    for start, end, label, color in BOE_REGIMES:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            alpha=0.15,
            color=color,
            label=label,
        )


def _format_date_axis(axes) -> None:
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


# ── Main plotting function ────────────────────────────────────────────

def plot_results(
    strategy_results: dict,
    output_dir: Optional[Path] = None,
    label: str = "",
) -> None:
    """Generate all strategy plots.

    Creates 3 figures:
      1. Strategy performance (cumPnL, weights, turnover, rolling Sharpe)
      2. PCA dynamics (loadings by role, explained variance, condition number,
         PC role assignments)
      3. Factor correlation (PnL vs each factor score scatter + time series)
    If signal-based results are present, overlays signal PnL on figure 1.
    """
    weights = strategy_results["weights"]
    daily_pnl = strategy_results["daily_pnl"]
    cumulative_pnl = strategy_results["cumulative_pnl"]
    diagnostics = strategy_results["diagnostics"]
    pc_scores = strategy_results.get("pc_scores", pd.DataFrame())

    suffix = f"_{label}" if label else ""
    title_suffix = f" ({label})" if label else ""

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Strategy Performance ──────────────────────────────
    has_signal = "signal_cumulative_pnl" in strategy_results
    n_panels = 5 if has_signal else 4
    fig1, axes1 = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    fig1.suptitle(f"PCA Butterfly Strategy (3s7s15s){title_suffix} — Performance", fontsize=14)

    # Panel A: Cumulative PnL
    ax = axes1[0]
    ax.plot(cumulative_pnl.index, cumulative_pnl.values, color="black", linewidth=1.2,
            label="Static (always-on)")
    if has_signal:
        sig_cum = strategy_results["signal_cumulative_pnl"]
        ax.plot(sig_cum.index, sig_cum.values, color="tab:red", linewidth=1.2,
                label="Signal-based")
    ax.set_ylabel("Cumulative PnL (bps)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    _add_regime_shading(ax)
    ax.legend(loc="upper left", fontsize=8, ncol=5)

    # Panel B: Weights
    ax = axes1[1]
    for col, lbl in [("w_3", "3y"), ("w_7", "7y"), ("w_15", "15y")]:
        ax.plot(weights.index, weights[col], label=lbl, linewidth=0.8)
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", fontsize=8)
    _add_regime_shading(ax)

    # Panel C: Turnover
    ax = axes1[2]
    turnover = compute_turnover(weights)
    ax.bar(turnover.index, turnover.values, width=1, color="steelblue", alpha=0.6)
    ax.set_ylabel("Turnover")
    _add_regime_shading(ax)

    # Panel D: Rolling 63-day Sharpe
    ax = axes1[3]
    rolling_mean = daily_pnl.rolling(63, min_periods=30).mean()
    rolling_std = daily_pnl.rolling(63, min_periods=30).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="darkgreen", linewidth=0.8,
            label="Static")
    if has_signal:
        sig_pnl = strategy_results["signal_daily_pnl"]
        sig_rm = sig_pnl.rolling(63, min_periods=30).mean()
        sig_rs = sig_pnl.rolling(63, min_periods=30).std()
        sig_sharpe = (sig_rm / sig_rs) * np.sqrt(252)
        ax.plot(sig_sharpe.index, sig_sharpe.values, color="tab:red", linewidth=0.8,
                label="Signal")
        ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Rolling Sharpe (63d)")
    _add_regime_shading(ax)

    # Panel E: Signal z-score and position (if present)
    if has_signal:
        ax = axes1[4]
        zscore = strategy_results["zscore"]
        position = strategy_results["signal_position"]
        ax.plot(zscore.index, zscore.values, color="tab:purple", linewidth=0.7, label="Z-score")
        ax.fill_between(position.index, position.values * 2, alpha=0.3, color="tab:orange",
                        label="Position (scaled)")
        ax.axhline(1.5, color="grey", linewidth=0.5, linestyle="--")
        ax.axhline(-1.5, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Z-score / Position")
        ax.legend(loc="upper left", fontsize=8)
        _add_regime_shading(ax)

    ax = axes1[-1]
    ax.set_xlabel("Date")
    _format_date_axis(axes1)
    fig1.tight_layout()
    fig1.autofmt_xdate()

    if output_dir:
        fig1.savefig(output_dir / f"strategy_performance{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: PCA Dynamics ──────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle(f"PCA Butterfly Strategy (3s7s15s){title_suffix} — PCA Dynamics", fontsize=14)

    # Panel A: Loadings at butterfly tenors (by identified role)
    ax = axes2[0]
    colors = {"3y": "tab:blue", "7y": "tab:orange", "15y": "tab:green"}
    for role, ls in [("level", "-"), ("slope", "--"), ("curv", ":")]:
        for tenor_label in ["3y", "7y", "15y"]:
            col = f"{role}_{tenor_label}"
            if col in diagnostics.columns:
                ax.plot(
                    diagnostics.index, diagnostics[col],
                    color=colors[tenor_label], linestyle=ls, linewidth=0.7,
                )
    legend_elements = [
        Line2D([0], [0], color="grey", linestyle="-", label="Level"),
        Line2D([0], [0], color="grey", linestyle="--", label="Slope"),
        Line2D([0], [0], color="grey", linestyle=":", label="Curvature"),
        Line2D([0], [0], color="tab:blue", linestyle="-", label="3y"),
        Line2D([0], [0], color="tab:orange", linestyle="-", label="7y"),
        Line2D([0], [0], color="tab:green", linestyle="-", label="15y"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, ncol=6)
    ax.set_ylabel("Loading (by identified role)")

    # Panel B: Explained variance
    ax = axes2[1]
    for pc_num in [1, 2, 3]:
        col = f"expl_var_pc{pc_num}"
        if col in diagnostics.columns:
            ax.plot(diagnostics.index, diagnostics[col] * 100, label=f"PC{pc_num}",
                    linewidth=0.8)
    ax.set_ylabel("Explained Variance (%)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel C: Condition number
    ax = axes2[2]
    ax.semilogy(diagnostics.index, diagnostics["cond_number"], color="crimson", linewidth=0.8)
    ax.set_ylabel("Condition Number (log)")
    ax.set_xlabel("Date")
    ax.axhline(100, color="grey", linewidth=0.5, linestyle="--", label="Threshold=100")
    ax.legend(fontsize=8)

    _format_date_axis(axes2)
    fig2.tight_layout()
    fig2.autofmt_xdate()

    if output_dir:
        fig2.savefig(output_dir / f"pca_dynamics{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: Factor Correlation Analysis ──────────────────────
    if not pc_scores.empty:
        corr_result = factor_correlation_analysis(daily_pnl, pc_scores)

        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.suptitle(
            f"PnL vs Factor Scores{title_suffix} — "
            f"R² = {corr_result['r_squared']:.3f}",
            fontsize=13,
        )

        common = daily_pnl.index.intersection(pc_scores.index)
        pnl_vals = daily_pnl.loc[common].values

        for idx, (col, color) in enumerate(
            [("level_score", "tab:blue"), ("slope_score", "tab:orange"),
             ("curvature_score", "tab:green")]
        ):
            if col not in pc_scores.columns:
                continue
            ax = axes3[idx]
            score_vals = pc_scores.loc[common, col].values
            ax.scatter(score_vals, pnl_vals, alpha=0.3, s=8, color=color)
            corr = corr_result["correlations"].get(col, 0)
            ax.set_title(f"{col.replace('_score', '').title()}\ncorr = {corr:.3f}")
            ax.set_xlabel("Factor Score (bps)")
            ax.set_ylabel("Strategy PnL (bps)")
            ax.axhline(0, color="grey", linewidth=0.3)
            ax.axvline(0, color="grey", linewidth=0.3)
            # Add regression line
            if len(score_vals) > 2:
                m, b = np.polyfit(score_vals, pnl_vals, 1)
                x_range = np.linspace(score_vals.min(), score_vals.max(), 50)
                ax.plot(x_range, m * x_range + b, color="black", linewidth=1,
                        linestyle="--")

        fig3.tight_layout()
        if output_dir:
            fig3.savefig(output_dir / f"factor_correlation{suffix}.png", dpi=150,
                         bbox_inches="tight")
        plt.close(fig3)

    print(f"Plots saved to {output_dir}" if output_dir else "Plots displayed")


def plot_regime_mismatch(
    curves: pd.DataFrame,
    window: int = 252,
    output_dir: Optional[Path] = None,
) -> None:
    """Plot estimation vs trading window covariance eigenvalues over time.

    For each rolling date, computes eigenvalues of the estimation window
    (prior 252 days) and the next 63-day forward window. Large divergence
    between them indicates regime mismatch — the estimation period is not
    representative of the trading period.
    """
    from pca import pca_on_curves

    dates = curves.index.tolist()
    n = len(dates)
    forward_window = 63

    records = []

    for i in range(window, n - forward_window):
        # Estimation window eigenvalues
        est_curves = curves.iloc[i - window: i + 1]
        fwd_curves = curves.iloc[i: i + forward_window + 1]

        try:
            est_res = pca_on_curves(est_curves, n_components=3, standardize=True)
            fwd_res = pca_on_curves(fwd_curves, n_components=3, standardize=True)
        except ValueError:
            continue

        records.append({
            "date": dates[i],
            "est_ev1": est_res["eigenvalues"][0],
            "est_ev2": est_res["eigenvalues"][1],
            "est_ev3": est_res["eigenvalues"][2],
            "fwd_ev1": fwd_res["eigenvalues"][0],
            "fwd_ev2": fwd_res["eigenvalues"][1],
            "fwd_ev3": fwd_res["eigenvalues"][2],
        })

    if not records:
        return

    df = pd.DataFrame(records).set_index("date")

    # Compute mismatch metric: ratio of forward to estimation eigenvalues
    df["mismatch_pc1"] = df["fwd_ev1"] / df["est_ev1"]
    df["mismatch_pc2"] = df["fwd_ev2"] / df["est_ev2"]
    df["mismatch_pc3"] = df["fwd_ev3"] / df["est_ev3"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Regime Mismatch: Estimation vs Trading Window Eigenvalues", fontsize=14)

    # Panel A: Eigenvalues over time
    ax = axes[0]
    for pc, color in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")]:
        ax.plot(df.index, df[f"est_ev{pc}"], color=color, linewidth=0.8,
                label=f"PC{pc} est.", linestyle="-")
        ax.plot(df.index, df[f"fwd_ev{pc}"], color=color, linewidth=0.8,
                label=f"PC{pc} fwd.", linestyle="--")
    ax.set_ylabel("Eigenvalue")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    _add_regime_shading(ax)

    # Panel B: Mismatch ratio (fwd / est)
    ax = axes[1]
    for pc, color in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")]:
        ax.plot(df.index, df[f"mismatch_pc{pc}"], color=color, linewidth=0.8,
                label=f"PC{pc}")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Eigenvalue Ratio\n(trading / estimation)")
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", fontsize=8)
    _add_regime_shading(ax)

    # Panel C: Total variance mismatch
    ax = axes[2]
    total_est = df["est_ev1"] + df["est_ev2"] + df["est_ev3"]
    total_fwd = df["fwd_ev1"] + df["fwd_ev2"] + df["fwd_ev3"]
    ax.plot(df.index, total_est, color="black", linewidth=0.8, label="Estimation window")
    ax.plot(df.index, total_fwd, color="red", linewidth=0.8, label="Trading window (63d fwd)")
    ax.set_ylabel("Total Variance\n(sum of eigenvalues)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8)
    _add_regime_shading(ax)

    _format_date_axis(axes)
    fig.tight_layout()
    fig.autofmt_xdate()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "regime_mismatch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Regime mismatch chart saved to {output_dir}/regime_mismatch.png")
