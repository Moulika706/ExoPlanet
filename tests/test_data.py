"""Tests for src/exovet/data.py: cleaning, feature sets, grouped splits."""

from __future__ import annotations

import pandas as pd

from exovet.data import (
    CORE_FEATURES,
    LABELS,
    clean_labels,
    feature_set,
    load_raw,
    missing_value_rates,
    split_by_group,
)


def test_clean_labels_keeps_only_three_classes() -> None:
    df = pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE", "OTHER", None],
            "kepid": [1, 2, 3, 4, 5],
            "koi_period": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = clean_labels(df)
    assert set(out["koi_disposition"]) == set(LABELS)
    assert len(out) == 3


def test_clean_labels_resets_index() -> None:
    df = pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED", "FALSE POSITIVE"],
            "kepid": [1, 2],
            "koi_period": [1.0, 2.0],
        },
        index=[10, 20],
    )
    out = clean_labels(df)
    assert list(out.index) == [0, 1]


def test_feature_sets() -> None:
    assert "koi_score" not in feature_set(False)
    assert "koi_score" in feature_set(True)
    assert set(feature_set(False)) == set(CORE_FEATURES)


def test_missing_value_rates_in_unit_interval() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
    rates = missing_value_rates(df, ["a", "b"])
    assert rates["a"] == 1 / 3
    assert rates["b"] == 0.0
    assert (rates >= 0).all() and (rates <= 1).all()


def test_split_has_no_group_overlap() -> None:
    df = pd.DataFrame(
        {
            "kepid": [1] * 10 + [2] * 10 + [3] * 10,
            "x": range(30),
            "koi_disposition": ["CONFIRMED"] * 30,
        }
    )
    train, test = split_by_group(df, test_size=1 / 3, seed=42)
    assert set(train["kepid"]) & set(test["kepid"]) == set()
    assert len(train) + len(test) == len(df)


def test_split_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "kepid": list(range(20)) * 2,
            "x": range(40),
            "koi_disposition": ["CONFIRMED"] * 40,
        }
    )
    t1 = split_by_group(df, seed=7)
    t2 = split_by_group(df, seed=7)
    pd.testing.assert_frame_equal(t1[0], t2[0])
    pd.testing.assert_frame_equal(t1[1], t2[1])


def test_load_raw_reads_fetched_csv(tmp_path) -> None:
    csv = tmp_path / "koi.csv"
    pd.DataFrame({"koi_disposition": ["CONFIRMED"], "kepid": [1]}).to_csv(csv, index=False)
    df = load_raw(str(csv))
    assert len(df) == 1
