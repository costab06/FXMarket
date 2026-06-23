"""Purged K-Fold cross-validation with embargo.

Reference: López de Prado (2018), Chapter 7: "Cross-Validation in Finance"

Standard K-Fold CV leaks information when labels overlap across train/test
splits: a test observation whose label was computed using bars that also
appear in the training set will produce inflated (optimistic) scores.

**Purging**: remove training observations whose label spans (t0, t1) overlap
with any test observation's label span.

**Embargo**: after the test fold, drop a small buffer of training observations
to prevent look-ahead leakage caused by features that use trailing windows
(e.g., EMA, rolling volatility).

This implementation follows the scikit-learn ``BaseCrossValidator`` interface
so it can be used directly with ``sklearn.model_selection.cross_val_score``
and friends.

Reference: López de Prado (2018), Sections 7.1–7.4, Snippet 7.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-Fold CV with purging and embargo for financial time-series.

    Splits the sample into ``n_splits`` folds while:

    1. **Purging**: removing training samples whose label period overlaps
       with any label in the test fold.
    2. **Embargo**: removing a fraction of training samples immediately
       after the test fold to prevent feature leakage.

    Conforms to the scikit-learn ``BaseCrossValidator`` interface (implements
    ``split``, ``get_n_splits``).  Compatible with sklearn utilities such as
    ``cross_val_score``, ``GridSearchCV``, etc.

    Reference: López de Prado (2018), Snippet 7.1.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    t1 : pd.Series, optional
        Maps event start times to exit/label end times.  Required to
        determine which training samples overlap with each test fold.
        If None, no purging is applied (equivalent to standard K-Fold).
    pct_embargo : float, default 0.0
        Fraction of the sample to use as the post-test embargo buffer.
        E.g. 0.01 embargoes 1% of the total observations after the test set.

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, t1=t1_series, pct_embargo=0.01)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.0,
    ) -> None:
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate (train_indices, test_indices) pairs.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Feature matrix.  If a DataFrame, its index is used as the
            timestamp reference.
        y : ignored
        groups : ignored

        Yields
        ------
        (np.ndarray, np.ndarray)
            Integer index arrays for train and test sets.
        """
        if isinstance(X, pd.DataFrame):
            idx = X.index
        else:
            idx = pd.RangeIndex(len(X))

        indices = np.arange(len(idx))
        embargo_size = int(len(idx) * self.pct_embargo)

        fold_size = len(idx) // self.n_splits
        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else len(idx)
            test_idx = indices[test_start:test_end]

            # Training indices: everything outside the test fold
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            if self.t1 is not None:
                # Purge: remove training samples whose label overlaps with test
                test_times = idx[test_idx]
                t1 = self.t1.reindex(idx)

                # Test fold time range
                test_t0 = test_times[0]
                test_t1 = test_times[-1]

                # Label end times for test samples
                if hasattr(test_times, '__iter__') and hasattr(t1, 'loc'):
                    try:
                        test_label_end = t1.loc[test_times].dropna()
                        if not test_label_end.empty:
                            test_t1 = test_label_end.max()
                    except KeyError:
                        pass

                purge_mask = []
                for ti in train_idx:
                    t = idx[ti]
                    t1_i = t1.iloc[ti] if ti < len(t1) else None
                    # Overlap: training label overlaps [test_t0, test_t1]
                    if t1_i is None or pd.isna(t1_i):
                        purge_mask.append(False)
                        continue
                    overlaps = (t1_i >= test_t0) and (t >= test_t0 or t1_i <= test_t1)
                    purge_mask.append(overlaps and t < test_t0)

                if purge_mask:
                    train_idx = train_idx[~np.array(purge_mask)]

            # Embargo: remove samples immediately after the test fold
            if embargo_size > 0:
                embargo_end = min(test_end + embargo_size, len(idx))
                embargo_idx = set(range(test_end, embargo_end))
                train_idx = np.array([i for i in train_idx if i not in embargo_idx])

            yield train_idx, test_idx
