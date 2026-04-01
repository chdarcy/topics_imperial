from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def pca_on_curves(
    curves: pd.DataFrame,
    n_components: Optional[int] = None,
) -> dict:
    
    cols = sorted(float(c) for c in curves.columns)
    curves = curves.reindex(columns=cols)

    diffs = curves.diff().dropna(how="all")
    diffs = diffs.dropna(axis=0, how="any")

    X = diffs.values
    X_centered = (X - diffs.mean(axis=0).values) / diffs.std(axis=0, ddof=1).values

    cov = np.cov(X_centered, rowvar=False, bias=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
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
    loadings = pd.DataFrame(eigvecs, index=cols, columns=pc_names)
    scores = pd.DataFrame(X_centered @ eigvecs, index=diffs.index, columns=pc_names)

    return {
        "loadings": loadings,
        "scores": scores,
        "eigenvalues": eigvals,
        "explained_variance_ratio": explained_variance_ratio,
    }


__all__ = ["pca_on_curves"]
