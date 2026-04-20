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
    return weights.diff().abs().sum(axis=1)



# ── Factor correlation analysis (requirement 6) ───────────────────────

def factor_correlation_analysis(
    daily_pnl: pd.Series,
    pc_scores: pd.DataFrame,
) -> dict:
    
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




# ── Main plotting function ────────────────────────────────────────────

def plot_results(
    strategy_results: dict,
    output_dir: Optional[Path] = None,
    label: str = "",
) -> None:
    """Generate factor correlation plot for the strategy."""
    
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

    # Derive tenor label for figure titles from weight column names
    w_cols = [c for c in weights.columns if c.startswith("w_")]
    tenor_tag = "s".join(c.replace("w_", "") for c in w_cols) + "s" if w_cols else ""

    # ── Figure 1: Factor Correlation Analysis ──────────────────────
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


# ── Variant comparison plots ──────────────────────────────────────────

def plot_variant_cumulative_pnl(
    curves_data: pd.DataFrame,
    strategy_results: dict,
    tenors: tuple,
    label: str,
    output_dir: Optional[Path] = None,
) -> None:
    """Overlay cumulative PnL of all variants for a single curve configuration."""

    from strategy import (
        apply_vol_scaling, apply_momentum_signal, apply_carry_overlay,
    )
    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Static
    cum = strategy_results["cumulative_pnl"]
    ax.plot(cum.index, cum.values, color="black", linewidth=1.5, label="Static")

    # Signal MR
    if "signal_cumulative_pnl" in strategy_results:
        sig = strategy_results["signal_cumulative_pnl"]
        ax.plot(sig.index, sig.values, color="#e74c3c", linewidth=1.3, label="Z-Score MR")

    # Variants
    variants = [
        ("Vol-Scaled", lambda s: apply_vol_scaling(s), "#3498db"),
        ("Momentum", lambda s: apply_momentum_signal(s), "#2ecc71"),
        ("Carry+MR", lambda s: apply_carry_overlay(s, curves_data, tenors=tenors), "#f39c12"),
    ]

    for vname, vfunc, color in variants:
        try:
            vres = vfunc(strategy_results)
            vc = vres["variant_cumulative_pnl"]
            ax.plot(vc.index, vc.values, linewidth=1.0, label=vname, color=color)
        except Exception:
            pass

    _add_regime_shading(ax)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative PnL (bps)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(f"Cumulative PnL — All Variants ({label})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    suffix = label.replace(" ", "_").lower()
    fig.savefig(output_dir / f"variant_cumulative_pnl_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_sample_pca(
    loadings: pd.DataFrame,
    explained_variance: list,
    output_dir: Optional[Path] = None,
) -> None:
    """Two-panel figure: (a) loading shapes, (b) explained variance bar chart."""

    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Full-Sample PCA: Loadings and Explained Variance", fontsize=13, fontweight="bold")

    mats = loadings.index.tolist()
    colors = ["#2c3e50", "#e74c3c", "#27ae60"]
    pc_labels = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    for i, col in enumerate(loadings.columns[:3]):
        ax1.plot(mats, loadings[col].values, marker="o", markersize=5, linewidth=2,
                 color=colors[i], label=pc_labels[i])
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Loading")
    ax1.set_title("(a) Factor Loadings")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    cumvar = np.cumsum(explained_variance[:3]) * 100
    bars = [ev * 100 for ev in explained_variance[:3]]
    x = np.arange(3)
    ax2.bar(x, bars, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax2.plot(x, cumvar, "ko-", markersize=6, linewidth=1.5, label="Cumulative")
    for i, (b, c) in enumerate(zip(bars, cumvar)):
        ax2.text(i, b + 1, f"{b:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["PC1\n(Level)", "PC2\n(Slope)", "PC3\n(Curvature)"])
    ax2.set_ylabel("Variance Explained (%)")
    ax2.set_title("(b) Explained Variance")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "full_sample_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Full-sample PCA figure saved.")


def plot_pc_role_tracking(
    all_results: list[dict],
    output_dir: Optional[Path] = None,
) -> None:
    """Show which PC is assigned to curvature over time for each strategy."""

    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_results = [r for r in all_results if r.get("label") != "fwd_1y-10y"]
    n = len(filtered_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("PC Role Assignment: Which PC is Curvature?", fontsize=13, fontweight="bold")

    pc_map = {"PC1": 1, "PC2": 2, "PC3": 3}
    colors_map = {1: "#2c3e50", 2: "#e74c3c", 3: "#27ae60"}

    for idx, r in enumerate(filtered_results):
        ax = axes[idx]
        diag = r["strategy"]["diagnostics"]
        if "curvature_pc" not in diag.columns:
            continue
        vals = diag["curvature_pc"].map(pc_map)
        for pc_val, color in colors_map.items():
            mask = vals == pc_val
            ax.fill_between(diag.index, 0, 1, where=mask,
                            color=color, alpha=0.6, transform=ax.get_xaxis_transform())
        ax.set_ylabel(r["label"], fontsize=10)
        ax.set_yticks([])
        _add_regime_shading(ax)

    legend_elements = [
        Line2D([0], [0], color=colors_map[1], linewidth=8, label="PC1"),
        Line2D([0], [0], color=colors_map[2], linewidth=8, label="PC2"),
        Line2D([0], [0], color=colors_map[3], linewidth=8, label="PC3"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "pc_role_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PC role tracking chart saved.")


def plot_drawdown_comparison(
    all_results: list[dict],
    output_dir: Optional[Path] = None,
) -> None:
    """Drawdown chart for static strategies across all curves."""

    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2c3e50", "#e74c3c", "#27ae60"]

    for idx, r in enumerate(all_results):
        pnl = r["strategy"]["daily_pnl"]
        cum = pnl.cumsum()
        dd = cum - cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color=colors[idx % len(colors)],
                        label=r["label"])

    _add_regime_shading(ax)
    ax.set_ylabel("Drawdown (bps)")
    ax.set_xlabel("Date")
    ax.set_title("Underwater Chart: Static Strategy Drawdowns", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Drawdown comparison chart saved.")


def plot_weight_stability(
    strategy_results: dict,
    label: str = "",
    output_dir: Optional[Path] = None,
) -> None:
    """Show rolling butterfly weights and their stability over time."""

    output_dir = Path(output_dir) if output_dir else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = strategy_results["weights"]
    w_cols = [c for c in weights.columns if c.startswith("w_")]
    if not w_cols:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    suffix = f" ({label})" if label else ""
    fig.suptitle(f"Butterfly Weight Dynamics{suffix}", fontsize=13, fontweight="bold")

    colors = ["#2c3e50", "#e74c3c", "#27ae60"]
    for i, col in enumerate(w_cols):
        lbl = col.replace("w_", "") + "y"
        ax1.plot(weights.index, weights[col], color=colors[i % len(colors)],
                 linewidth=0.8, label=lbl)
    ax1.set_ylabel("Weight")
    ax1.legend(fontsize=9)
    ax1.set_title("(a) Butterfly Weights Over Time")
    _add_regime_shading(ax1)
    ax1.grid(alpha=0.3)

    # Weight change (turnover)
    turnover = weights[w_cols].diff().abs().sum(axis=1)
    ax2.bar(turnover.index, turnover.values, width=1, color="steelblue", alpha=0.6)
    ax2.set_ylabel("Turnover\n(Σ|Δw|)")
    ax2.set_title("(b) Daily Turnover")
    _add_regime_shading(ax2)
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()
    sfx = label.replace(" ", "_").lower()
    fig.savefig(output_dir / f"weight_stability_{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
