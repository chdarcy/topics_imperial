from pathlib import Path
import pandas as pd

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


def run_strategy(curves, label, base, tenors, window=252):
	"""Run the full strategy pipeline. Returns collected results for summary."""
	print(f"\n{'='*60}")
	print(f"  STRATEGY: {label}")
	print(f"{'='*60}")

	output_dir = base / "output"

	# --- Rolling PCA butterfly (static, always-on) ---
	strategy = rolling_pca_butterfly(curves, window=window, tenors=tenors, n_components=3)

	print(f"  Out-of-sample days: {len(strategy['daily_pnl'])}")
	print(f"  Date range: {strategy['daily_pnl'].index[0].date()} to "
		  f"{strategy['daily_pnl'].index[-1].date()}")

	# --- PC role tracking ---
	diag = strategy["diagnostics"]
	role_counts = diag["curvature_pc"].value_counts()
	print(f"  Curvature was identified as: {dict(role_counts)}")

	# --- Static strategy metrics ---
	print("\n  --- Static (always-on) strategy ---")
	metrics_static = compute_performance_metrics(strategy["daily_pnl"])
	print_metrics(metrics_static)

	# --- Signal-based strategy (z-score mean reversion) ---
	strategy = apply_trading_signals(
		strategy, curves, tenors=tenors,
		zscore_window=63, entry_threshold=1.5, exit_threshold=0.5,
		stop_loss_bps=50.0, take_profit_bps=30.0,
	)
	sig_pnl = strategy["signal_daily_pnl"]
	print("\n  --- Signal-based strategy (z-score mean reversion) ---")
	metrics_signal = compute_performance_metrics(sig_pnl)
	print_metrics(metrics_signal)
	n_trades = (strategy["signal_position"].diff().abs() > 0).sum()
	print(f"  Number of position changes: {n_trades}")

	# --- Strategy variants ---
	variant_metrics = {}
	variant_defs = [
		("vol_scaled", "Vol-scaled", apply_vol_scaling, {"vol_window": 63, "vol_target_bps": 5.0}),
		("momentum", "Momentum (21d/5d)", apply_momentum_signal, {"lookback": 21, "holding_period": 5}),
		("carry_mr", "Carry + MR overlay", apply_carry_overlay, {"curves": curves, "tenors": tenors}),
		("pc_score_mom", "PC3 score momentum", apply_pc_score_momentum, {"lookback": 10}),
		("regime_filter", "Regime filter (vol > 50th pct)", apply_regime_filter, {"vol_window": 63, "vol_threshold_pct": 50.0}),
	]
	for key, display, func, kwargs in variant_defs:
		print(f"\n  --- Variant: {display} ---")
		vres = func(strategy, **kwargs)
		m = compute_performance_metrics(vres["variant_daily_pnl"])
		print_metrics(m)
		variant_metrics[key] = m

	# --- Factor correlation analysis ---
	pc_scores = strategy["pc_scores"]
	print("\n  --- Factor Correlation Analysis ---")
	corr_result = factor_correlation_analysis(strategy["daily_pnl"], pc_scores)
	print(f"  Correlations:")
	for k, v in corr_result["correlations"].items():
		print(f"    {k:20s}: {v:+.4f}")
	print(f"  Regression R²:      {corr_result['r_squared']:.4f}")
	print(f"  Betas:")
	for k, v in corr_result["betas"].items():
		print(f"    {k:20s}: {v:+.4f}")

	# --- Trade log and plots ---
	trade_log = generate_trade_log(strategy["weights"], strategy["daily_pnl"])
	trade_log.to_csv(base / f"trade_log_{label.replace(' ', '_').lower()}.csv")
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

def main():
	base = Path(__file__).resolve().parent
	output_dir = base / "output"

	# Load data
	df = DataLoader.load_sheet(xlsx_path=base / "input.xlsx", sheet_name="gbp ois results")
	print(f"Raw data shape: {df.shape}")

	# Interpolate to grid
	interpolated = interpolate_to_grid(df, target_maturities=(1, 2, 3, 5, 7, 10, 15, 20, 30))
	print(f"Interpolated spot curves: {interpolated.shape}")
	print(f"Date range: {interpolated.index[0]} to {interpolated.index[-1]}")

	# Full-sample PCA for reference + PC identification
	print("\n*******************Full-Sample PCA*******************")
	res = pca_on_curves(interpolated, n_components=3)

	# Also compute full PCA to get true (non-renormalized) variance ratios
	res_full = pca_on_curves(interpolated, n_components=None)
	print("Explained variance ratio (first 3 of 9):", [f"{x:.4f}" for x in res_full["explained_variance_ratio"][:3]])
	print(f"Cumulative (3 PCs): {sum(res_full['explained_variance_ratio'][:3]):.4f}")
	print("\nLoadings:")
	print(res["loadings"].to_string(float_format="{:.4f}".format))

	pc_roles = identify_pcs(res["loadings"])
	print(f"\nPC identification: {pc_roles}")

	# Collect results from all strategies
	all_results = []

	# ── Strategy 1: Spot rates, full curve 1-30y ───────────────────
	r1 = run_strategy(interpolated, "spot_1y-30y", base, tenors=(3.0, 7.0, 15.0))
	all_results.append(r1)

	# ── Strategy 2: Spot rates, truncated to 1-10y ─────────────────
	interp_10y = interpolate_to_grid(df, target_maturities=(1, 2, 3, 5, 7, 10))
	r2 = run_strategy(interp_10y, "spot_1y-10y", base, tenors=(2.0, 5.0, 10.0))
	all_results.append(r2)

	# ── Strategy 3: Forward rates, full curve ──────────────────────
	fwd_curves = compute_forward_rates(interpolated)
	r3 = run_strategy(fwd_curves, "fwd_1y", base, tenors=(3.0, 7.0, 15.0))
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
			curves_map["fwd_1y"] = (fwd_curves, (3.0, 7.0, 15.0))

		for r in all_results:
			lbl = r["label"]
			plot_loading_shapes(r["strategy"], label=lbl, output_dir=output_dir)
			plot_weight_stability(r["strategy"], label=lbl, output_dir=output_dir)
			if lbl in curves_map:
				cd, tn = curves_map[lbl]
				plot_variant_cumulative_pnl(cd, r["strategy"], tn, lbl,
										   output_dir=output_dir)

	# ── Summary table ──────────────────────────────────────────────
	print_summary_table(all_results)

	print("\n" + "=" * 60)
	print("  All strategies complete. Check output/ for plots.")
	print("=" * 60)


if __name__ == "__main__":
	main()
