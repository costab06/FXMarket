"""Entropy-based features for market microstructure analysis.

Reference: López de Prado (2018), Chapter 18: "Entropy Features"

Entropy measures the information content of a sequence.  In financial
markets, entropy is useful for:

- Detecting informed trading (low entropy → predictable order flow)
- Characterising market microstructure regimes
- Building features for ML models

Four entropy estimators are implemented:

1. **Shannon entropy**: classical information entropy H = -Σ p log p.
   Requires a discretised sequence; sensitive to bin choice.

2. **Lempel-Ziv complexity** (LZ76): compresses the sequence and measures
   how many distinct sub-words are needed.  Distribution-free.

3. **Plug-in entropy** (maximum likelihood estimator): estimates the
   probability distribution from frequencies and computes Shannon entropy.

4. **Kontoyiannis entropy** (match-length estimator): uses the length of
   the shortest sub-sequence starting at t that has not appeared in the
   preceding n characters.  Consistent and efficient for stationary processes.

Reference: López de Prado (2018), Sections 18.1–18.4, Snippets 18.1–18.4.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd


# ── Discretisation helper ────────────────────────────────────────────────────


def _to_string(x: np.ndarray | pd.Series, n_bins: int = 10) -> str:
    """Quantise a numeric series into a symbolic string.

    Maps each value to a letter (a–z, extended to digits) corresponding to
    its quantile bin.  Used internally by entropy estimators that require
    a discrete alphabet.

    Parameters
    ----------
    x : array-like
        Numeric sequence.
    n_bins : int, default 10
        Number of quantile bins.

    Returns
    -------
    str
        Symbolic string of length len(x).
    """
    x = np.asarray(x, dtype=float)
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    labels = np.digitize(x, bins[:-1]) - 1
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(alphabet[min(l, len(alphabet) - 1)] for l in labels)


# ── Shannon entropy ──────────────────────────────────────────────────────────


def shannon_entropy(
    x: str | np.ndarray | pd.Series,
    n_bins: int = 10,
    base: float = 2.0,
) -> float:
    """Compute Shannon entropy of a sequence.

    H(X) = -Σ_x p(x) log_b p(x)

    If ``x`` is a string, it is treated directly as a symbolic sequence.
    If numeric, it is quantised into ``n_bins`` quantile bins first.

    Reference: López de Prado (2018), Section 18.1, Snippet 18.1.

    Parameters
    ----------
    x : str or array-like
        Symbolic string or numeric sequence to measure.
    n_bins : int, default 10
        Number of quantile bins (used only for numeric input).
    base : float, default 2.0
        Logarithm base.  2 → bits, e → nats, 10 → dits.

    Returns
    -------
    float
        Shannon entropy in the specified base.
    """
    if not isinstance(x, str):
        x = _to_string(x, n_bins)
    counts = Counter(x)
    n = len(x)
    return -sum((c / n) * math.log(c / n, base) for c in counts.values())


# ── Lempel-Ziv complexity ────────────────────────────────────────────────────


def lempel_ziv_complexity(
    x: str | np.ndarray | pd.Series,
    n_bins: int = 10,
    normalise: bool = True,
) -> float:
    """Compute the Lempel-Ziv (LZ76) complexity of a sequence.

    LZ76 builds a dictionary of sub-words incrementally: starting from the
    beginning, it finds the longest sub-word in the current dictionary and
    adds the next character.  The complexity c(n) is the number of distinct
    sub-words needed to describe the full sequence.

    Normalised: c(n) / (n / log₂(|Σ|))  where |Σ| is the alphabet size.
    Ranges in (0, 1]: 1 = maximally complex (random), < 1 = predictable.

    Reference: López de Prado (2018), Section 18.2, Snippet 18.2.

    Parameters
    ----------
    x : str or array-like
        Symbolic string or numeric sequence.
    n_bins : int, default 10
        Number of quantile bins (used only for numeric input).
    normalise : bool, default True
        Whether to normalise by the theoretical maximum complexity.

    Returns
    -------
    float
        LZ76 complexity (normalised if ``normalise=True``).
    """
    if not isinstance(x, str):
        x = _to_string(x, n_bins)
    n = len(x)
    if n == 0:
        return 0.0

    # LZ76 parsing
    i, c = 0, 1
    dictionary = {x[0]}
    while i < n - 1:
        for j in range(i + 1, n):
            sub = x[i + 1: j + 1]
            if sub not in dictionary:
                dictionary.add(sub)
                c += 1
                i = j
                break
        else:
            break

    if not normalise:
        return float(c)

    alphabet_size = len(set(x))
    if alphabet_size <= 1:
        return 0.0
    max_c = n / math.log2(n) if n > 1 else 1.0
    return c / max_c


# ── Plug-in entropy ──────────────────────────────────────────────────────────


def plug_in_entropy(
    x: str | np.ndarray | pd.Series,
    word_length: int = 1,
    n_bins: int = 10,
    base: float = 2.0,
) -> float:
    """Compute plug-in (maximum-likelihood) entropy using word frequencies.

    Counts all overlapping sub-words of length ``word_length`` in the
    sequence, estimates their probabilities by frequency, and computes
    Shannon entropy.  For word_length=1 this is equivalent to :func:`shannon_entropy`.

    Provides a richer estimate of entropy by capturing word-level structure.

    Reference: López de Prado (2018), Section 18.3, Snippet 18.3.

    Parameters
    ----------
    x : str or array-like
        Symbolic string or numeric sequence.
    word_length : int, default 1
        Length of sub-words (n-grams) to count.
    n_bins : int, default 10
        Number of quantile bins (used only for numeric input).
    base : float, default 2.0
        Logarithm base.

    Returns
    -------
    float
        Plug-in entropy estimate.
    """
    if not isinstance(x, str):
        x = _to_string(x, n_bins)

    n = len(x)
    if n < word_length:
        return 0.0

    words = [x[i: i + word_length] for i in range(n - word_length + 1)]
    counts = Counter(words)
    total = len(words)
    entropy = -sum((c / total) * math.log(c / total, base) for c in counts.values())
    return entropy / word_length  # bits per character


# ── Kontoyiannis entropy ─────────────────────────────────────────────────────


def kontoyiannis_entropy(
    x: str | np.ndarray | pd.Series,
    n_bins: int = 10,
    base: float = 2.0,
) -> float:
    """Compute the Kontoyiannis (match-length) entropy estimator.

    For each position t, finds the length Λ_t of the shortest sub-sequence
    starting at t that has not appeared in x[0:t].  The entropy estimate is:

        H ≈ (1/n) Σ log(t) / Λ_t

    This estimator is consistent for stationary ergodic processes and does
    not require specifying a word length or number of bins.

    Reference: López de Prado (2018), Section 18.4, Snippet 18.4.

    Parameters
    ----------
    x : str or array-like
        Symbolic string or numeric sequence.
    n_bins : int, default 10
        Number of quantile bins (used only for numeric input).
    base : float, default 2.0
        Logarithm base.

    Returns
    -------
    float
        Kontoyiannis entropy estimate.
    """
    if not isinstance(x, str):
        x = _to_string(x, n_bins)

    n = len(x)
    if n < 2:
        return 0.0

    log_sum = 0.0
    count = 0

    for t in range(1, n):
        seen = set()
        # Build all sub-strings in x[0:t]
        for i in range(t):
            for j in range(i + 1, t + 1):
                seen.add(x[i:j])

        # Find shortest sub-string starting at t not in seen
        for length in range(1, n - t + 1):
            sub = x[t: t + length]
            if sub not in seen:
                if t > 0:
                    log_sum += math.log(t, base) / length
                    count += 1
                break

    if count == 0:
        return 0.0
    return log_sum / count
