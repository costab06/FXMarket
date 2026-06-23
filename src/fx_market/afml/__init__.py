"""Advances in Financial Machine Learning (AFML) feature engineering library.

Implements the core techniques from:

    López de Prado, M. (2018).
    "Advances in Financial Machine Learning."
    Wiley, Hoboken NJ.

Modules
-------
bars                Financial data structures: tick, volume, dollar, imbalance bars (Ch. 2)
labeling            Triple-barrier method, fixed-horizon labels, meta-labeling (Ch. 3)
fracdiff            Fractional differentiation for stationarity with memory (Ch. 5)
sampling            CUSUM filter, sample weights, sequential bootstrap (Ch. 2, 4)
cv                  Purged K-Fold and embargo cross-validation (Ch. 7)
entropy             Shannon, Lempel-Ziv, plug-in, and Kontoyiannis entropy (Ch. 18)
structural_breaks   CUSUM and Supremum-ADF structural-break tests (Ch. 17)
hrp                 Hierarchical Risk Parity portfolio construction (Ch. 16)
importance          MDI, MDA, and SFI feature importance measures (Ch. 8)

Optional dependencies
---------------------
Most modules use only numpy and pandas.  Install the [ml] extra for
scikit-learn and scipy, which are required by ``cv``, ``hrp``, and
``importance``:

    pip install "fx-market[ml]"
"""

from .bars import dollar_bars, tick_bars, tick_imbalance_bars, volume_bars, volume_imbalance_bars
from .cv import PurgedKFold
from .entropy import kontoyiannis_entropy, lempel_ziv_complexity, plug_in_entropy, shannon_entropy
from .fracdiff import fracdiff_expanding, fracdiff_ffd, min_ffd
from .hrp import hrp
from .importance import feature_importance_mda, feature_importance_mdi, feature_importance_sfi
from .labeling import (
    daily_vol,
    fixed_horizon_labels,
    get_bins,
    get_events,
    meta_label,
    triple_barrier,
)
from .microstructure import (
    corwin_schultz_spread,
    hasbrouck_lambda,
    kyle_lambda,
    tick_entropy,
    volume_entropy,
    vpin,
)
from .sampling import (
    average_uniqueness,
    cusum_filter,
    ind_matrix,
    sample_weights,
    seq_bootstrap,
    time_decay_weights,
)
from .structural_breaks import chow_type_adf, cusum_structural_break, sadf

__all__ = [
    # bars
    "tick_bars", "volume_bars", "dollar_bars",
    "tick_imbalance_bars", "volume_imbalance_bars",
    # labeling
    "daily_vol", "get_events", "triple_barrier", "get_bins",
    "fixed_horizon_labels", "meta_label",
    # fracdiff
    "fracdiff_ffd", "fracdiff_expanding", "min_ffd",
    # sampling
    "cusum_filter", "ind_matrix", "average_uniqueness",
    "sample_weights", "seq_bootstrap", "time_decay_weights",
    # cv
    "PurgedKFold",
    # entropy
    "shannon_entropy", "lempel_ziv_complexity", "plug_in_entropy", "kontoyiannis_entropy",
    # structural_breaks
    "cusum_structural_break", "chow_type_adf", "sadf",
    # hrp
    "hrp",
    # importance
    "feature_importance_mdi", "feature_importance_mda", "feature_importance_sfi",
    # microstructure (Ch. 19)
    "kyle_lambda", "hasbrouck_lambda", "vpin",
    "corwin_schultz_spread", "tick_entropy", "volume_entropy",
]
