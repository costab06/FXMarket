"""Feature importance: MDI, MDA, and SFI.

Reference: López de Prado (2018), Chapter 8: "Feature Importance"

Standard feature importance measures (e.g. sklearn's impurity-based
importance) are biased toward high-cardinality features and correlated
features, and do not measure out-of-sample predictive power.

Three complementary measures are implemented:

1. **MDI** (Mean Decrease in Impurity): fast, in-bag estimate.
   Computes the weighted average decrease in node impurity caused by each
   feature across all trees.  Prone to substitution effects when features
   are correlated.

2. **MDA** (Mean Decrease in Accuracy): out-of-bag estimate.
   Shuffles each feature one at a time and measures the resulting drop
   in cross-validated accuracy.  Detects substitution effects (correlated
   features both drop less).

3. **SFI** (Single Feature Importance): trains a model using each feature
   individually and reports the cross-validated score.  Immune to
   substitution effects; computationally expensive.

All functions accept a fitted or unfitted sklearn-compatible estimator.

Reference: López de Prado (2018), Sections 8.1–8.3, Snippets 8.1–8.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── MDI ─────────────────────────────────────────────────────────────────────


def feature_importance_mdi(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean Decrease in Impurity (MDI) feature importance.

    Extracts the impurity-based feature importances from a fitted
    tree-ensemble model (e.g. ``RandomForestClassifier``).  Each tree's
    feature importances are averaged, and the standard deviation across
    trees is reported as a stability measure.

    Reference: López de Prado (2018), Section 8.1, Snippet 8.2.

    Parameters
    ----------
    model : fitted sklearn tree ensemble
        A fitted ``RandomForestClassifier``, ``RandomForestRegressor``,
        ``GradientBoostingClassifier``, etc.  Must have ``estimators_``
        attribute (list of decision trees).
    feature_names : list[str]
        Feature names corresponding to the columns of the training set.

    Returns
    -------
    pd.DataFrame
        Columns: [mean, std].  Indexed by feature name.
        Sorted by mean importance (descending).
    """
    importances = np.array([
        tree.feature_importances_ for tree in model.estimators_
    ])
    imp_df = pd.DataFrame(
        {
            "mean": importances.mean(axis=0),
            "std":  importances.std(axis=0) * (importances.shape[0] ** -0.5),
        },
        index=feature_names,
    )
    # Normalise so importances sum to 1 (they should already, but trees may differ)
    imp_df["mean"] /= imp_df["mean"].sum()
    return imp_df.sort_values("mean", ascending=False)


# ── MDA ─────────────────────────────────────────────────────────────────────


def feature_importance_mda(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "neg_log_loss",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Mean Decrease in Accuracy (MDA) feature importance.

    For each feature, shuffles its values across the test fold and measures
    the drop in cross-validated score.  A large drop indicates high importance.

    Algorithm:
    1. Fit model on each training fold, score on test fold → baseline score.
    2. Shuffle feature j in the test fold (leaving others unchanged).
    3. Re-score → shuffled score.
    4. MDA[j] = mean(baseline score - shuffled score) across folds.

    Reference: López de Prado (2018), Section 8.2, Snippet 8.3.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Unfitted estimator (cloned per fold).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    cv : cross-validator
        An iterable or sklearn CV object that yields (train_idx, test_idx).
        Use :class:`PurgedKFold` for financial data.
    scoring : str, default "neg_log_loss"
        sklearn scoring string.
    random_state : int, optional
        Seed for feature shuffling.

    Returns
    -------
    pd.DataFrame
        Columns: [mean, std].  Indexed by feature name.
        Positive mean = feature reduces score when removed (important).
    """
    try:
        from sklearn.base import clone
        from sklearn.metrics import check_scoring
    except ImportError as e:
        raise ImportError(
            "feature_importance_mda requires scikit-learn.  "
            "Install with: pip install scikit-learn"
        ) from e

    rng = np.random.default_rng(random_state)
    scorer = check_scoring(model, scoring=scoring)
    feature_names = list(X.columns)
    n_features = len(feature_names)

    baseline_scores: list[float] = []
    # importance_per_fold[fold][feature] = drop in score
    imp_per_fold: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx]

        clf = clone(model)
        clf.fit(X_train, y_train)
        baseline = scorer(clf, X_test, y_test)
        baseline_scores.append(baseline)

        fold_imp: dict[str, float] = {}
        for feat in feature_names:
            X_shuffled = X_test.copy()
            X_shuffled[feat] = rng.permutation(X_shuffled[feat].values)
            score_shuf = scorer(clf, X_shuffled, y_test)
            fold_imp[feat] = baseline - score_shuf
        imp_per_fold.append(fold_imp)

    imp_df = pd.DataFrame(imp_per_fold, columns=feature_names)
    return pd.DataFrame(
        {"mean": imp_df.mean(), "std": imp_df.std() * (len(imp_per_fold) ** -0.5)},
    ).sort_values("mean", ascending=False)


# ── SFI ─────────────────────────────────────────────────────────────────────


def feature_importance_sfi(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str = "neg_log_loss",
) -> pd.DataFrame:
    """Single Feature Importance (SFI).

    Trains the model on each feature individually and records the
    cross-validated score.  Features that score well in isolation are
    genuinely important (not merely correlated substitutes).

    This is the most computationally expensive estimator: one CV run per
    feature.  Use for feature selection on pre-filtered feature sets.

    Reference: López de Prado (2018), Section 8.3, Snippet 8.4.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Unfitted estimator (cloned per feature).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    cv : cross-validator
        An iterable or sklearn CV object.  Use :class:`PurgedKFold`.
    scoring : str, default "neg_log_loss"
        sklearn scoring string.

    Returns
    -------
    pd.DataFrame
        Columns: [mean, std].  Indexed by feature name.
        Higher mean score = more important feature.
    """
    try:
        from sklearn.base import clone
        from sklearn.metrics import check_scoring
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        raise ImportError(
            "feature_importance_sfi requires scikit-learn.  "
            "Install with: pip install scikit-learn"
        ) from e

    feature_names = list(X.columns)
    results: list[dict] = []

    for feat in feature_names:
        X_single = X[[feat]]
        scores = cross_val_score(
            clone(model), X_single, y, cv=cv, scoring=scoring
        )
        results.append({
            "feature": feat,
            "mean": scores.mean(),
            "std":  scores.std() * (len(scores) ** -0.5),
        })

    return (
        pd.DataFrame(results)
        .set_index("feature")
        .sort_values("mean", ascending=False)
    )
