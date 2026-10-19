"""Group-aware data splitting for the ExoVet pipeline.

Every split groups rows by host star (``kepid``) so that sibling KOIs of the
same star never straddle train and test. This is the difference between a
believable number and a leaked one: about 1.2 KOIs per host star in the
cumulative table, so a random split puts related objects on both sides.
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import GROUP_COLUMN, TARGET


class Fold(NamedTuple):
    """One grouped cross-validation fold."""

    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def grouped_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = GROUP_COLUMN,
    target_column: str = TARGET,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Fold]:
    """Return stratified, group-aware folds over *df*.

    Stratification keeps the class balance close across folds; the grouping
    constraint keeps every host star entirely inside one fold.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    folds: List[Fold] = []
    groups = df[group_column].to_numpy()
    y = df[target_column].to_numpy()
    for index, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups=groups)):
        folds.append(Fold(index=index, train_idx=train_idx, test_idx=test_idx))
    return folds


def random_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_column: str = TARGET,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Fold]:
    """Return ordinary stratified folds, ignoring the host-star grouping.

    Used only as the leaky baseline of the split ablation.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y = df[target_column].to_numpy()
    folds: List[Fold] = []
    for index, (train_idx, test_idx) in enumerate(splitter.split(df, y)):
        folds.append(Fold(index=index, train_idx=train_idx, test_idx=test_idx))
    return folds


def train_calibration_test(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_fold: int = 0,
    calibration_fold: int = 1,
    group_column: str = GROUP_COLUMN,
    target_column: str = TARGET,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split *df* into train / calibration / test index arrays.

    The calibration fold is kept separate from train so that isotonic or
    Platt calibration never sees the training data, and the test fold is
    touched by neither the model nor the calibrator.
    """
    if test_fold == calibration_fold:
        raise ValueError("test_fold and calibration_fold must differ")
    if not (0 <= test_fold < n_splits) or not (0 <= calibration_fold < n_splits):
        raise ValueError("fold indices must be within range(n_splits)")

    folds = grouped_kfold(
        df,
        n_splits=n_splits,
        group_column=group_column,
        target_column=target_column,
        seed=seed,
    )
    test_idx = folds[test_fold].test_idx
    calibration_idx = folds[calibration_fold].test_idx
    used = np.concatenate([test_idx, calibration_idx])
    train_idx = np.setdiff1d(np.arange(len(df)), used, assume_unique=False)
    return train_idx, calibration_idx, test_idx


def random_train_calibration_test(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_fold: int = 0,
    calibration_fold: int = 1,
    target_column: str = TARGET,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same three-way split as :func:`train_calibration_test` but ungrouped.

    This is the leaky baseline of the split ablation: sibling KOIs of the
    same host star can end up on both sides of the split.
    """
    if test_fold == calibration_fold:
        raise ValueError("test_fold and calibration_fold must differ")
    folds = random_kfold(df, n_splits=n_splits, target_column=target_column, seed=seed)
    test_idx = folds[test_fold].test_idx
    calibration_idx = folds[calibration_fold].test_idx
    used = np.concatenate([test_idx, calibration_idx])
    train_idx = np.setdiff1d(np.arange(len(df)), used, assume_unique=False)
    return train_idx, calibration_idx, test_idx


def sibling_leak_train_calibration_test(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    calibration_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and calibration folds that leak siblings of the test fold.

    The *test* fold is kept exactly as the grouped split produced it, and
    only the training pool is chosen at random. Because the training pool
    is drawn from every remaining row, a host star whose KOIs are in the
    test fold will usually have another KOI in training.

    Comparing this against the grouped split isolates the sibling leak:
    same test rows, same model, only the composition of the training data
    differs.
    """
    test_idx = np.asarray(test_idx)
    remaining = np.setdiff1d(np.arange(len(df)), test_idx, assume_unique=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(remaining)
    n_calibration = int(round(len(remaining) * calibration_fraction))
    calibration_idx = np.sort(order[:n_calibration])
    train_idx = np.sort(order[n_calibration:])
    return train_idx, calibration_idx, test_idx


def assert_disjoint_groups(
    df: pd.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    group_column: str = GROUP_COLUMN,
) -> None:
    """Raise if any host star appears in both index arrays."""
    left = set(df.iloc[left_idx][group_column])
    right = set(df.iloc[right_idx][group_column])
    overlap = left & right
    if overlap:
        raise AssertionError(
            f"{len(overlap)} host star(s) present in both splits, e.g. {sorted(overlap)[:5]}"
        )


def iter_folds(folds: List[Fold]) -> Iterator[Fold]:
    """Iterate over folds (convenience wrapper)."""
    yield from folds
