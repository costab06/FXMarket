"""Hierarchical Risk Parity (HRP) portfolio construction.

Reference: López de Prado (2018), Chapter 16: "Machine Learning Asset Allocation"

HRP builds a diversified portfolio without inverting the covariance matrix
(which is numerically unstable for large universes and causes Markowitz's
"error maximisation" problem).  Instead, HRP:

1. **Clusters** assets hierarchically using a correlation distance matrix.
2. **Quasi-diagonalises** the covariance matrix by reordering assets along
   the dendrogram leaves.
3. **Allocates** risk recursively via bisection: at each level, the two
   sub-clusters receive weights inversely proportional to their variance.

HRP produces more stable, diversified allocations than minimum-variance
or maximum-Sharpe portfolios, especially out-of-sample.

Reference: López de Prado (2018), Sections 16.1–16.4, Snippets 16.1–16.5.
         López de Prado (2016), "Building Diversified Portfolios that Outperform
         Out-of-Sample", Journal of Portfolio Management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Correlation distance ─────────────────────────────────────────────────────


def _corr_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation distance matrix.

    d(i, j) = sqrt(0.5 * (1 - ρ_{ij}))

    This metric satisfies the triangle inequality and maps correlation
    in [-1, 1] to distance in [0, 1].

    Reference: López de Prado (2018), Snippet 16.3.
    """
    return ((1 - corr) / 2) ** 0.5


# ── Hierarchical clustering ──────────────────────────────────────────────────


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Sort clustered assets to place similar assets adjacent (quasi-diagonalisation).

    Recursively extracts the leaf order from a scipy linkage matrix so that
    the covariance matrix, when reordered, is approximately block-diagonal.

    Reference: López de Prado (2018), Snippet 16.2.

    Parameters
    ----------
    link : np.ndarray
        Scipy linkage matrix (shape n-1 × 4).

    Returns
    -------
    list[int]
        Ordered list of original asset indices (leaves of the dendrogram).
    """
    n = int(link[-1, 3])  # total number of original observations
    n_leaves = int(link[-1, 3] - link.shape[0])

    # Index: leaf nodes are 0..n_leaves-1, internal nodes are n_leaves..n-1
    cluster_map: dict[int, list[int]] = {i: [i] for i in range(n_leaves)}

    for i, row in enumerate(link):
        left, right = int(row[0]), int(row[1])
        node_id = n_leaves + i
        cluster_map[node_id] = cluster_map.pop(left, [left]) + cluster_map.pop(right, [right])

    # The last merge gives the full ordered list
    return list(cluster_map[n_leaves + len(link) - 1])


# ── Recursive bisection allocation ──────────────────────────────────────────


def _get_cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Compute the variance of a cluster's minimum-variance portfolio.

    The minimum-variance portfolio of a cluster (for internal allocation)
    uses only the sub-covariance matrix of the cluster's assets.

    Reference: López de Prado (2018), Snippet 16.4.
    """
    cov_sub = cov.loc[items, items]
    n = len(items)
    # Inverse-variance weights
    inv_var = 1.0 / np.diag(cov_sub.values)
    w = inv_var / inv_var.sum()
    return float(w @ cov_sub.values @ w)


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_items: list,
) -> pd.Series:
    """Allocate weights by recursive bisection.

    At each level, the portfolio is split into two sub-clusters.  Each
    sub-cluster receives weight proportional to the inverse of its variance,
    ensuring equal risk contribution at every level of the hierarchy.

    Reference: López de Prado (2018), Snippet 16.5.

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix of all assets.
    sorted_items : list
        Quasi-diagonalised asset order (output of :func:`_get_quasi_diag`).

    Returns
    -------
    pd.Series
        Portfolio weights indexed by asset name.
    """
    w = pd.Series(1.0, index=sorted_items)
    clusters = [sorted_items]

    while clusters:
        clusters = [
            c[i:j]
            for c in clusters
            for i, j in [(0, len(c) // 2), (len(c) // 2, len(c))]
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            left, right = clusters[i], clusters[i + 1]
            v_left  = _get_cluster_var(cov, left)
            v_right = _get_cluster_var(cov, right)
            alpha = 1 - v_left / (v_left + v_right)
            w[left]  *= alpha
            w[right] *= 1 - alpha

    return w


# ── HRP main function ────────────────────────────────────────────────────────


def hrp(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute Hierarchical Risk Parity portfolio weights.

    1. Computes the correlation-based distance matrix.
    2. Clusters assets using hierarchical agglomerative clustering.
    3. Quasi-diagonalises the covariance matrix using the cluster leaf order.
    4. Allocates weights via recursive bisection of the variance.

    Requires ``scipy`` (part of the [ml] extra).

    Reference: López de Prado (2018), Section 16.4, Snippets 16.1–16.5.
             López de Prado (2016), Journal of Portfolio Management.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return matrix: rows are dates, columns are assets.
        Log-returns or simple returns accepted.
    linkage_method : str, default "single"
        scipy hierarchical clustering linkage method.
        Options: "single", "complete", "average", "ward".
        "single" (minimum linkage) is used in the original paper.

    Returns
    -------
    pd.Series
        Portfolio weights (summing to 1) indexed by asset name.

    Examples
    --------
    >>> returns = pd.DataFrame(np.random.randn(252, 5), columns=list("ABCDE"))
    >>> weights = hrp(returns)
    >>> weights.sum()  # approximately 1.0
    """
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError as e:
        raise ImportError(
            "hrp requires scipy.  Install with: pip install scipy"
        ) from e

    cov  = returns.cov()
    corr = returns.corr()
    dist = _corr_dist(corr)

    # Convert to condensed distance matrix for scipy
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method=linkage_method)

    # Quasi-diagonalise: reorder assets by cluster leaf order
    sort_ix = _get_quasi_diag(link)
    assets = list(returns.columns)
    sorted_assets = [assets[i] for i in sort_ix]

    # Allocate
    weights = _recursive_bisection(cov, sorted_assets)
    return weights.reindex(assets)
