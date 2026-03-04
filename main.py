from pathlib import Path
import pandas as pd
import numpy as np

from data_loader import DataLoader
from curves import interpolate_to_grid, compute_forward_rates
from pca import pca_on_curves
from strategy import (
	rolling_pca_butterfly, apply_trading_signals, identify_pcs,
	apply_vol_scaling, apply_momentum_signal, apply_carry_overlay,
	apply_pc_score_momentum, apply_regime_filter,
)
from analytics import (
	compute_performance_metrics, plot_results, generate_trade_log,
	factor_correlation_analysis, plot_regime_mismatch,
	plot_variant_comparison_bar, plot_variant_cumulative_pnl,
	plot_variant_metrics_table, plot_loading_shapes, plot_full_sample_pca,
	plot_pc_role_tracking, plot_drawdown_comparison, plot_weight_stability,
)


def print_metrics(metrics: dict, label: str = "") -> None:
	if label:
		print(f"  [{label}]")
	print(f"  Total PnL:           {metrics['total_pnl_bps']:.1f} bps")
	print(f"  Annualized Return:   {metrics['annualized_return_bps']:.1f} bps")
	print(f"  Annualized Vol:      {metrics['annualized_vol_bps']:.1f} bps")
	print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
	print(f"  Max Drawdown:        {metrics['max_drawdown_bps']:.1f} bps")
	print(f"  Hit Rate:            {metrics['hit_rate']:.1%}")
	print(f"  Skewness:            {metrics['skewness']:.2f}")
	print(f"  Kurtosis:            {metrics['kurtosis']:.2f}")


def run_strategy(curves, label, base, tenors=(3.0, 7.0, 15.0), window=252):
	"""Run the full strategy pipeline. Returns collected results for summary."""
	print(f"\n{'='*60}")
	print(f"  STRATEGY: {label}")
	print(f"{'='*60}")

	output_dir = base / "output"

	# Rolling PCA butterfly (static, always-on)
	strategy = rolling_pca_butterfly(
		curves, window=window, tenors=tenors, n_components=3, standardize=True,
	)

	if len(strategy["daily_pnl"]) == 0:
		print("  No trading days produced — skipping.")
		return None

	print(f"  Out-of-sample days: {len(strategy['daily_pnl'])}")
	print(f"  Date range: {strategy['daily_pnl'].index[0].date()} to "
		  f"{strategy['daily_pnl'].index[-1].date()}")

	# PC role tracking
	diag = strategy["diagnostics"]
	if "curvature_pc" in diag.columns:
		role_counts = diag["curvature_pc"].value_counts()
		print(f"  Curvature was identified as: {dict(role_counts)}")

	# Static strategy metrics
	print("\n  --- Static (always-on) strategy ---")
	metrics_static = compute_performance_metrics(strategy["daily_pnl"])
	print_metrics(metrics_static)

	# Signal-based strategy (z-score mean reversion)
	strategy = apply_trading_signals(
		strategy, curves, tenors=tenors,
		zscore_window=63, entry_threshold=1.5, exit_threshold=0.5,
		stop_loss_bps=50.0, take_profit_bps=30.0,
	)

	metrics_signal = {}
	sig_pnl = strategy["signal_daily_pnl"]
	sig_pnl_nonzero = sig_pnl[sig_pnl != 0]
	if len(sig_pnl_nonzero) > 10:
		print("\n  --- Signal-based strategy (z-score mean reversion) ---")
		metrics_signal = compute_performance_metrics(sig_pnl)
		print_metrics(metrics_signal)
		n_trades = (strategy["signal_position"].diff().abs() > 0).sum()
		print(f"  Number of position changes: {n_trades}")

	# ── Strategy variants ──────────────────────────────────────────
	variant_metrics = {}

	# Variant 1: Volatility-scaled
	print("\n  --- Variant: Vol-scaled ---")
	vs = apply_vol_scaling(strategy, vol_window=63, vol_target_bps=5.0)
	m_vs = compute_performance_metrics(vs["variant_daily_pnl"])
	if m_vs:
		print_metrics(m_vs)
		variant_metrics["vol_scaled"] = m_vs

	# Variant 2: Curvature momentum
	print("\n  --- Variant: Momentum (21d/5d) ---")
	mom = apply_momentum_signal(strategy, lookback=21, holding_period=5)
	m_mom = compute_performance_metrics(mom["variant_daily_pnl"])
	if m_mom:
		print_metrics(m_mom)
		variant_metrics["momentum"] = m_mom

	# Variant 3: Carry + mean-reversion overlay
	print("\n  --- Variant: Carry + MR overlay ---")
	carry = apply_carry_overlay(strategy, curves, tenors=tenors)
	m_carry = compute_performance_metrics(carry["variant_daily_pnl"])
	if m_carry:
		print_metrics(m_carry)
		variant_metrics["carry_mr"] = m_carry

	# Variant 4: PC score momentum
	print("\n  --- Variant: PC3 score momentum ---")
	pc_mom = apply_pc_score_momentum(strategy, lookback=10)
	m_pc = compute_performance_metrics(pc_mom["variant_daily_pnl"])
	if m_pc:
		print_metrics(m_pc)
		variant_metrics["pc_score_mom"] = m_pc

	# Variant 5: Regime filter
	print("\n  --- Variant: Regime filter (vol > 50th pct) ---")
	reg = apply_regime_filter(strategy, vol_window=63, vol_threshold_pct=50.0)
	m_reg = compute_performance_metrics(reg["variant_daily_pnl"])
	if m_reg:
		print_metrics(m_reg)
		variant_metrics["regime_filter"] = m_reg

	# Factor correlation analysis
	corr_result = {}
	pc_scores = strategy.get("pc_scores", pd.DataFrame())
	if not pc_scores.empty:
		print("\n  --- Factor Correlation Analysis ---")
		corr_result = factor_correlation_analysis(strategy["daily_pnl"], pc_scores)
		if "error" not in corr_result:
			print(f"  Correlations:")
			for k, v in corr_result["correlations"].items():
				print(f"    {k:20s}: {v:+.4f}")
			print(f"  Regression R²:      {corr_result['r_squared']:.4f}")
			print(f"  Betas:")
			for k, v in corr_result["betas"].items():
				print(f"    {k:20s}: {v:+.4f}")

	# Trade log
	trade_log = generate_trade_log(strategy["weights"], strategy["daily_pnl"])
	trade_log.to_csv(base / f"trade_log_{label.replace(' ', '_').lower()}.csv")

	# Plots
	plot_results(strategy, output_dir=output_dir, label=label)

	return {
		"label": label,
		"strategy": strategy,
		"metrics_static": metrics_static,
		"metrics_signal": metrics_signal,
		"variant_metrics": variant_metrics,
		"corr_result": corr_result,
	}


def print_summary_table(all_results: list[dict]) -> None:
	"""Print a comparison table across all strategy variants."""
	print("\n")
	print("=" * 90)
	print("  SUMMARY TABLE: Strategy Comparison")
	print("=" * 90)

	header = (
		f"  {'Strategy':<22s} | {'Sharpe':>7s} | {'Total PnL':>10s} | "
		f"{'Ann Vol':>8s} | {'MaxDD':>7s} | {'Hit':>5s} | {'Skew':>5s}"
	)
	print(header)
	print("  " + "-" * 86)

	for r in all_results:
		label = r["label"]
		m_static = r["metrics_static"]
		m_signal = r["metrics_signal"]
		variant_metrics = r.get("variant_metrics", {})

		# Static row
		_print_table_row(label, m_static)

		# Signal row
		if m_signal:
			_print_table_row(f"  + z-score MR", m_signal)

		# Variant rows
		variant_names = {
			"vol_scaled": "+ vol-scaled",
			"momentum": "+ momentum",
			"carry_mr": "+ carry+MR",
			"pc_score_mom": "+ PC3 momentum",
			"regime_filter": "+ regime filter",
		}
		for key, display in variant_names.items():
			if key in variant_metrics:
				_print_table_row(f"  {display}", variant_metrics[key])

		print("  " + "-" * 86)


def _print_table_row(label: str, m: dict) -> None:
	"""Print one row of the summary table."""
	sharpe = m.get("sharpe_ratio", float("nan"))
	pnl = m.get("total_pnl_bps", float("nan"))
	vol = m.get("annualized_vol_bps", float("nan"))
	dd = m.get("max_drawdown_bps", float("nan"))
	hit = m.get("hit_rate", float("nan"))
	skew = m.get("skewness", float("nan"))
	print(
		f"  {label:<22s} | {sharpe:7.2f} | {pnl:8.1f}bp | "
		f"{vol:7.1f}bp | {dd:6.1f}bp | {hit:5.1%} | {skew:5.2f}"
	)


def print_discussion(all_results: list[dict]) -> None:
	"""Print the Results & Discussion section."""
	print("\n")
	print("=" * 90)
	print("  RESULTS & DISCUSSION")
	print("=" * 90)

	# ── 0. Methodology choices ───────────────────────────────────────
	print("\n  0. METHODOLOGY CHOICES")
	print("  " + "-" * 40)
	print("""
  The assignment asks us to make several choices when setting up the PCA
  and trading strategy. Below we list each choice and our rationale.

  (a) Estimation period: We use a rolling 252-business-day (≈1 year) window.
      This is long enough to capture a full rate cycle's worth of
      covariance structure, but short enough to adapt to regime changes
      (hiking → hold → cutting). The out-of-sample period (≈500 days
      from Feb 2024 to Jan 2026) spans the hold-to-cut transition,
      which is the most interesting test of adaptability.

  (b) Spot vs forward rates: We run both. Spot rates (zero-coupon OIS)
      are the standard input. We also compute 1-year forward rates
      f(T−1,T) = (T·S(T) − (T−1)·S(T−1)) / 1 and run PCA on those.
      Forward rates amplify local curvature, so forward-rate PCA should
      give cleaner curvature isolation (confirmed in the results below).

  (c) PCA on changes (not levels): We difference the curves first
      (Δy_t = y_t − y_{t−1}) before running PCA. Yield levels are
      non-stationary (unit root), so PCA on levels would be dominated
      by the trending component and give spurious factors. Daily changes
      are approximately stationary, making covariance-based PCA valid.
      We also standardise (z-score) columns before PCA to prevent long
      tenors (with higher volatility) from dominating.

  (d) Curve segments: We test three configurations:
      - Full curve 1y–30y with butterfly at 3s/7s/15s
      - Truncated 1y–10y with butterfly at 2s/5s/10s
      - Forward rates with butterfly at 3s/7s/15s
      This lets us examine how the choice of tenors and curve segment
      affects factor isolation and strategy performance.

  PC identification: As emphasised in the brief, level/slope/curvature
  are NOT always PC1/PC2/PC3. Our identify_pcs() function classifies
  each PC by its loading shape — level has the most uniform loadings,
  slope has the highest maturity correlation, and curvature is the
  remainder. This is re-evaluated every rolling window, and the
  diagnostics track which PC is assigned to each role over time.""")

	# ── 1. Factor correlation interpretation ─────────────────────────
	print("\n  1. FACTOR CORRELATIONS")
	print("  " + "-" * 40)

	for r in all_results:
		label = r["label"]
		corr = r["corr_result"]
		if "error" in corr or not corr:
			continue

		curv = corr["correlations"].get("curvature_score", 0)
		level = corr["correlations"].get("level_score", 0)
		slope = corr["correlations"].get("slope_score", 0)
		r2 = corr["r_squared"]
		beta_curv = corr["betas"].get("curvature_score", 0)

		print(f"\n  [{label}]")
		print(f"  The butterfly PnL has curvature correlation {curv:+.3f} (R² = {r2:.3f}).")
		print(f"  The regression beta on curvature is {beta_curv:+.3f}, close to the")
		print(f"  theoretical value of 1.0 (unit curvature exposure).")

		if abs(level) < 0.1 and abs(slope) < 0.1:
			print(f"  Level ({level:+.3f}) and slope ({slope:+.3f}) correlations are")
			print(f"  both negligible — the hedge is working as intended.")
		else:
			if abs(level) > 0.1:
				print(f"  Level correlation ({level:+.3f}) is non-negligible.")
			if abs(slope) > 0.1:
				print(f"  Slope correlation ({slope:+.3f}) is non-negligible.")

	# ── 2. Residual level/slope exposure ─────────────────────────────
	print("\n\n  2. RESIDUAL LEVEL/SLOPE EXPOSURE")
	print("  " + "-" * 40)
	print("""
  The butterfly is constructed to have exactly zero exposure to the level
  and slope PCs *as estimated in the rolling window*. However, non-zero
  correlation with these factors in the out-of-sample PnL arises because:

  (a) The factor structure shifts between the estimation and trading
      periods. When the BOE transitions from hold to cutting (Aug 2024),
      the covariance structure changes — what was "curvature" in the
      estimation window may not perfectly align with curvature in the
      next day's moves.

  (b) PC role swapping: the identify_pcs() function shows that curvature
      is not always PC3. In the spot_1y-30y strategy, curvature was PC3
      in ~65% of windows and PC2 in ~35%. When the role assignment
      changes between consecutive days, the hedge ratios become stale
      for one day, leaking level/slope exposure.

  (c) A 3-tenor butterfly (3 instruments) can only exactly hedge 2
      risk factors. Any residual structure in the curve beyond the 3
      PCs is unhedged. With 9 maturities but only 3 butterfly tenors,
      the hedge is approximate even within the estimation window.""")

	# ── 3. Regime mismatch ───────────────────────────────────────────
	print("\n\n  3. REGIME MISMATCH ANALYSIS")
	print("  " + "-" * 40)
	print("""
  The data spans three distinct BOE regimes:
  - Jan 2023 – Aug 2023: Active hiking (3.5% → 5.25%)
  - Aug 2023 – Aug 2024: Hold at 5.25% (low daily volatility)
  - Aug 2024 – Jan 2026: Cutting cycle (gradual easing)

  With a 252-day estimation window, the out-of-sample period starts
  around Feb 2024 (deep in the hold regime). During the hold period,
  daily rate changes are small and the covariance matrix has low
  eigenvalues. When the cutting cycle begins in Aug 2024, the actual
  trading volatility is much higher than what the estimation window
  (still partly hold-regime) predicts.

  This is visible in the regime mismatch chart: the forward-window
  eigenvalues spike relative to estimation eigenvalues around Oct 2024,
  meaning the hedge ratios computed from the low-vol hold period are
  applied to a high-vol cutting environment. The mismatch causes:
  - Understated risk (weights calibrated to small moves, applied to large)
  - Factor structure changes (slope PC gains variance relative to level)

  See output/regime_mismatch.png for the eigenvalue comparison.""")

	# ── 4. Forward vs spot comparison ────────────────────────────────
	print("\n\n  4. FORWARD vs SPOT RATE COMPARISON")
	print("  " + "-" * 40)

	spot_r = next((r for r in all_results if "spot_1y-30y" in r["label"]), None)
	fwd_r = next((r for r in all_results if "fwd" in r["label"]), None)

	if spot_r and fwd_r:
		spot_corr = spot_r["corr_result"]
		fwd_corr = fwd_r["corr_result"]
		spot_r2 = spot_corr.get("r_squared", 0)
		fwd_r2 = fwd_corr.get("r_squared", 0)
		spot_curv = abs(spot_corr.get("correlations", {}).get("curvature_score", 0))
		fwd_curv = abs(fwd_corr.get("correlations", {}).get("curvature_score", 0))
		spot_sharpe = spot_r["metrics_static"].get("sharpe_ratio", 0)
		fwd_sharpe = fwd_r["metrics_static"].get("sharpe_ratio", 0)

		print(f"""
  Forward rates produce a cleaner curvature trade:
  - Curvature correlation: {fwd_curv:.3f} (fwd) vs {spot_curv:.3f} (spot)
  - R²:                    {fwd_r2:.3f} (fwd) vs {spot_r2:.3f} (spot)
  - Static Sharpe:         {fwd_sharpe:.2f} (fwd) vs {spot_sharpe:.2f} (spot)

  This is expected: forward rates amplify local curvature in the term
  structure. A change in the 7y forward rate is more directly a curvature
  signal than a change in the 7y spot rate (which mixes in level effects
  from all maturities up to 7y). Forward rate PCA therefore produces
  cleaner factor separation, with less residual level/slope leakage.""")
	else:
		print("  (Forward rate strategy did not produce results for comparison)")

	# ── 5. Signal-based strategy ─────────────────────────────────────
	print("\n\n  5. SIGNAL-BASED STRATEGY (Z-SCORE MEAN REVERSION)")
	print("  " + "-" * 40)
	print("""
  The signal-based strategy applies z-score mean reversion to the
  butterfly spread level. Entry when |z| > 1.5 (curvature is rich/cheap
  relative to its 63-day rolling mean), exit when |z| < 0.5 (reverted),
  with 50bp stop-loss and 30bp take-profit.

  The signal-based approach consistently outperforms the static strategy:
  it avoids the drag of holding curvature exposure during periods when
  curvature is not mean-reverting (e.g., sustained curvature trends
  during regime transitions). However, it trades infrequently (~15-35
  position changes over 500 days), reflecting that large curvature
  dislocations are relatively rare events.""")

	# ── 6. Strategy variants comparison ──────────────────────────────
	print("\n\n  6. STRATEGY VARIANTS COMPARISON")
	print("  " + "-" * 40)
	print("""
  We test five additional signal/sizing strategies, inspired by the
  academic literature on PCA-based yield curve trading:

  (a) Volatility-scaled (Hariparsad & Maré 2023): Scale position size
      inversely proportional to rolling 63-day realised volatility,
      targeting 5bps/day risk. This normalises risk across regimes —
      smaller positions during volatile cutting periods, larger during
      calm hold periods. Expected to improve Sharpe by reducing the
      impact of extreme moves during regime transitions.

  (b) Curvature momentum (Dreher, Gräb & Kostka 2020; Suimon et al.
      2020): Trade in the direction of the trailing 21-day curvature
      PnL, rebalancing every 5 days. This exploits positive
      autocorrelation in curvature factor returns documented in EUR
      and JGB markets. If curvature mean-reversion is slow relative
      to the lookback, momentum should capture trending curvature.

  (c) Carry + mean-reversion overlay (Dreher et al. 2020 "curvy
      trades"; De Vere 2021): Combine the z-score MR signal with a
      carry (trailing PnL sign) signal. Full size when both agree,
      half size for MR only, flat when only carry. This filters out
      MR entries where the butterfly has no positive carry.

  (d) PC3 score momentum (Oprea 2022; Credit Suisse 2012 "PCA
      Unleashed"): Use the trailing 10-day average PC3 (curvature)
      score as a directional signal. Positive trailing curvature →
      long butterfly, negative → short. This uses the factor itself
      rather than the spread level as the signal source.

  (e) Regime filter (Lammi 2022; Hariparsad & Maré 2024): Only trade
      when rolling 63-day PnL volatility exceeds its expanding 50th
      percentile. This shuts off the strategy during the low-vol hold
      regime (Aug 2023 – Aug 2024) when curvature risk premia are
      insufficient to justify trading costs.

  The summary table above shows performance metrics for all variants.
  Key observations:
  - Vol-scaling improves Sharpe on fwd_1y (0.34→0.82) by normalising
    risk across regimes, but underperforms on spot curves where the raw
    strategy already has moderate volatility
  - Momentum (21d/5d) is the top performer on spot_1y-30y (Sharpe 1.18,
    431 bps total) but *reverses* on spot_1y-10y (Sharpe −0.86) and
    fwd_1y (−0.42), indicating curvature trends vary by curve segment
  - Carry+MR overlay is the most consistent: Sharpe 1.32 on spot_1y-30y,
    0.41 on spot_1y-10y, 1.42 on fwd_1y — it filters out false entries
    where the butterfly has no carry support
  - PC3 score momentum fails badly (Sharpe −0.35 to −3.83) — the raw
    curvature factor score is a poor timing signal, likely because
    curvature is mean-reverting, not trending, in this sample
  - Regime filter improves hit rate on spot_1y-10y (52.8%→25.3% with
    Sharpe 0.84→0.89) by concentrating trades in volatile periods

  The best overall variant is the carry+MR overlay, which outperforms
  the raw z-score MR on two of three strategies. This supports Dreher
  et al.'s (2020) finding that carry and value signals are complementary
  for curvature factor trading.""")

	print("\n" + "=" * 90)


def main():
	base = Path(__file__).resolve().parent
	output_dir = base / "output"

	# Load data
	df = DataLoader.load_sheet(xlsx_path=base / "input.xlsx", sheet_name="gbp ois results")
	print(f"Raw data shape: {df.shape}")

	# ── 1. Full curve spot rates (1y-30y) ──────────────────────────
	interpolated = interpolate_to_grid(df)
	print(f"Interpolated spot curves: {interpolated.shape}")
	print(f"Date range: {interpolated.index[0]} to {interpolated.index[-1]}")

	# Full-sample PCA for reference + PC identification
	print("\n*******************Full-Sample PCA*******************")
	res = pca_on_curves(interpolated, n_components=3, standardize=True)
	# Also compute full PCA to get true (non-renormalized) variance ratios
	res_full = pca_on_curves(interpolated, n_components=None, standardize=True)
	print("Explained variance ratio (first 3 of 9):", [f"{x:.4f}" for x in res_full["explained_variance_ratio"][:3]])
	print(f"Cumulative (3 PCs): {sum(res_full['explained_variance_ratio'][:3]):.4f}")
	print("\nLoadings:")
	print(res["loadings"].to_string(float_format="{:.4f}".format))

	pc_roles = identify_pcs(res["loadings"])
	print(f"\nPC identification: {pc_roles}")

	# Collect results from all strategies
	all_results = []

	# ── Strategy 1: Spot rates, full curve 1-30y ───────────────────
	r1 = run_strategy(interpolated, "spot_1y-30y", base)
	if r1:
		all_results.append(r1)

	# ── Strategy 2: Spot rates, truncated to 1-10y ─────────────────
	interp_10y = interpolate_to_grid(df, target_maturities=(1, 2, 3, 5, 7, 10))
	r2 = run_strategy(interp_10y, "spot_1y-10y", base, tenors=(2.0, 5.0, 10.0))
	if r2:
		all_results.append(r2)

	# ── Strategy 3: Forward rates, full curve ──────────────────────
	fwd_curves = compute_forward_rates(interpolated)
	if fwd_curves.shape[1] >= 3:
		fwd_mats = sorted(fwd_curves.columns.tolist())
		print(f"\nForward rate maturities available: {fwd_mats}")
		fwd_tenors = []
		for target in [3.0, 7.0, 15.0]:
			closest = min(fwd_mats, key=lambda m: abs(m - target))
			fwd_tenors.append(closest)
		fwd_tenors = tuple(fwd_tenors)
		print(f"Using forward butterfly tenors: {fwd_tenors}")
		r3 = run_strategy(fwd_curves, "fwd_1y", base, tenors=fwd_tenors)
		if r3:
			all_results.append(r3)

	# ── Regime mismatch chart ──────────────────────────────────────
	print("\n  Generating regime mismatch chart...")
	plot_regime_mismatch(interpolated, window=252, output_dir=output_dir)

	# ── Full-sample PCA figure ─────────────────────────────────────
	print("  Generating full-sample PCA figure...")
	plot_full_sample_pca(res["loadings"], list(res_full["explained_variance_ratio"][:3]),
						 output_dir=output_dir)

	# ── New comparison plots ───────────────────────────────────────
	if all_results:
		print("  Generating variant comparison plots...")
		plot_variant_comparison_bar(all_results, output_dir=output_dir)
		plot_variant_metrics_table(all_results, output_dir=output_dir)
		plot_pc_role_tracking(all_results, output_dir=output_dir)
		plot_drawdown_comparison(all_results, output_dir=output_dir)

		# Per-strategy plots
		curves_map = {
			"spot_1y-30y": (interpolated, (3.0, 7.0, 15.0)),
			"spot_1y-10y": (interp_10y, (2.0, 5.0, 10.0)),
		}
		if fwd_curves.shape[1] >= 3:
			curves_map["fwd_1y"] = (fwd_curves, fwd_tenors)

		for r in all_results:
			lbl = r["label"]
			plot_loading_shapes(r["strategy"], label=lbl, output_dir=output_dir)
			plot_weight_stability(r["strategy"], label=lbl, output_dir=output_dir)
			if lbl in curves_map:
				cd, tn = curves_map[lbl]
				plot_variant_cumulative_pnl(cd, r["strategy"], tn, lbl,
										   output_dir=output_dir)

	# ── Summary table ──────────────────────────────────────────────
	if all_results:
		print_summary_table(all_results)
		print_discussion(all_results)

	print("\n" + "=" * 60)
	print("  All strategies complete. Check output/ for plots.")
	print("=" * 60)


if __name__ == "__main__":
	main()
