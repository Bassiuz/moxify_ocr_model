"""Tests for :mod:`moxify_ocr.data.splits`."""

from __future__ import annotations

import pytest

from moxify_ocr.data.splits import assign_split


def test_same_set_same_split_deterministic() -> None:
    """Same set code, same seed → same split every time."""
    first = assign_split("clu", seed=0)
    for _ in range(5):
        assert assign_split("clu", seed=0) == first


def test_different_seeds_different_splits_for_same_set() -> None:
    """At least one set must land in different splits for different seeds."""
    # Try many set codes; at least one must differ across seeds.
    candidates = [f"s{i:03d}" for i in range(200)]
    differences = 0
    for code in candidates:
        if assign_split(code, seed=0) != assign_split(code, seed=1):
            differences += 1
    assert differences > 0


def test_holdout_goes_to_test() -> None:
    """Set in holdout_sets → always 'test', even if hash would say train."""
    # Pick a set that under seed=0 lands in train (to prove holdout overrides it).
    train_candidate: str | None = None
    for code in (f"s{i:04d}" for i in range(500)):
        if assign_split(code, seed=0) == "train":
            train_candidate = code
            break
    assert train_candidate is not None
    assert assign_split(train_candidate, seed=0) == "train"
    assert (
        assign_split(train_candidate, seed=0, holdout_sets=frozenset({train_candidate}))
        == "test"
    )


def test_rough_distribution() -> None:
    """Against ~100 synthetic set codes, distribution ≈ 85/10/5 (±10 tolerance)."""
    codes = [f"set{i:04d}" for i in range(100)]
    counts = {"train": 0, "val": 0, "test": 0}
    for code in codes:
        counts[assign_split(code, seed=0)] += 1
    assert abs(counts["train"] - 85) <= 10
    assert abs(counts["val"] - 10) <= 10
    assert abs(counts["test"] - 5) <= 10


def test_case_insensitive() -> None:
    """'CLU' and 'clu' land in the same split."""
    assert assign_split("CLU", seed=0) == assign_split("clu", seed=0)
    assert assign_split("MkM", seed=7) == assign_split("mkm", seed=7)


def test_valid_splits_only() -> None:
    """Output is always in {'train', 'val', 'test'}."""
    for i in range(500):
        split = assign_split(f"s{i}", seed=i)
        assert split in {"train", "val", "test"}


def test_seed_0_default() -> None:
    """Default seed is 0."""
    assert assign_split("clu") == assign_split("clu", seed=0)


def test_holdout_empty_default() -> None:
    """Empty frozenset default works normally; set not in holdout routes via hash."""
    # Calling without holdout_sets should equal calling with empty frozenset.
    assert assign_split("clu", seed=0) == assign_split(
        "clu", seed=0, holdout_sets=frozenset()
    )


def test_holdout_case_insensitive() -> None:
    """Holdout comparison must match the lowercase form."""
    assert assign_split("CLU", seed=0, holdout_sets=frozenset({"clu"})) == "test"


@pytest.mark.parametrize("set_code", ["clu", "mkm", "dmu", "woe", "lci"])
def test_parametrized_determinism(set_code: str) -> None:
    """Same inputs, same outputs across calls."""
    assert assign_split(set_code, seed=42) == assign_split(set_code, seed=42)
