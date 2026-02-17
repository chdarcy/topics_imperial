"""PCA utilities for yield curve daily changes.

This module computes principal components from daily changes of yield curves.
Input: DataFrame indexed by date with maturities (years) as float columns.

API:
 - pca_on_curves(curves_df, n_components=None, standardize=True, min_obs=2)
     returns a dict with keys: loadings (DataFrame), scores (DataFrame),
     eigenvalues (np.ndarray), explained_variance_ratio (np.ndarray), means, stds

Implementation notes:
 - Uses numpy eigendecomposition on the sample covariance matrix (no sklearn).
 - Computes daily differences first to ensure stationarity.
 - Optionally standardizes columns (z-score) before PCA.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def pca_on_curves(
    curves: pd.DataFrame,
    n_components: Optional[int] = None,
    *,
    standardize: bool = True,
    min_obs: int = 2,
) -> dict:
    """Run PCA on daily changes of yield curves.

    Parameters
    - curves: DataFrame indexed by date, columns are maturities (floats).
    - n_components: number of PCs to keep (defaults to all).
    - standardize: if True, z-score columns (zero mean, unit std) before PCA.
    - min_obs: minimum number of observations (after differencing and dropping NaNs).

    Returns a dict with:
    - loadings: DataFrame (maturities x PCs) with eigenvectors
    - scores: DataFrame (dates x PCs) with principal component time series
    - eigenvalues: ndarray of eigenvalues (descending)
    - explained_variance_ratio: ndarray
    - means: Series of column means used (on diffs)
    - stds: Series of column stds used (or None if not standardizing)
    """
    if curves.shape[1] == 0:
        raise ValueError("Input curves DataFrame has no maturity columns")

    # ensure columns are sorted floats
    try:
        cols = sorted(float(c) for c in curves.columns)
    except Exception:
        cols = list(curves.columns)
    curves = curves.reindex(columns=cols)

    # daily changes
    diffs = curves.diff().dropna(how="all")
    # drop any rows with NaN to build a clean matrix
    diffs = diffs.dropna(axis=0, how="any")

    if diffs.shape[0] < min_obs:
        raise ValueError("Not enough observations after differencing to run PCA")

    # handle zero-variance (non-positive or NaN) columns after differencing
    dropped_columns = []
    stds = None
    if standardize:
        stds = diffs.std(axis=0, ddof=1)
        zero_std = stds[(stds <= 0) | stds.isna()].index.tolist()
        if zero_std:
            dropped_columns = zero_std
            diffs = diffs.drop(columns=zero_std)

    # store the (possibly reduced) columns used for PCA
    cols_used = list(diffs.columns)

    # recompute means/stds on the reduced diffs
    means = diffs.mean(axis=0)
    if standardize:
        stds = diffs.std(axis=0, ddof=1)

    X = diffs.values  # shape (T, M)
    X_centered = X - means.values
    if standardize:
        # divide directly by remaining stds (we already dropped non-positive/NaN)
        X_centered = X_centered / stds.values

    # covariance matrix (variables in columns)
    cov = np.cov(X_centered, rowvar=False, bias=False)

    # eigendecomposition (symmetric matrix)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    M = eigvecs.shape[1]
    if n_components is None:
        n_components = M
    n_components = min(int(n_components), M)

    eigvals = eigvals[:n_components]
    eigvecs = eigvecs[:, :n_components]

    explained_variance_ratio = eigvals / np.sum(eigvals)

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    # align loadings index with the columns actually used for PCA
    try:
        loadings_index = [float(c) for c in cols_used]
    except Exception:
        loadings_index = cols_used
    loadings = pd.DataFrame(eigvecs, index=loadings_index, columns=pc_names)

    scores = pd.DataFrame(X_centered @ eigvecs, index=diffs.index, columns=pc_names)

    result = {
        "loadings": loadings,
        "scores": scores,
        "eigenvalues": eigvals,
        "explained_variance_ratio": explained_variance_ratio,
        "means": means,
        "stds": (stds if standardize else None),
    }
    if dropped_columns:
        result["dropped_columns"] = dropped_columns
    return result


__all__ = ["pca_on_curves"]
