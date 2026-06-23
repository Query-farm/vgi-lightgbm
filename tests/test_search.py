"""Unit tests for grid_search helpers."""

from __future__ import annotations

import pytest

from vgi_lightgbm.search import _combos, _parse_grid


class TestParseGrid:
    def test_lists(self) -> None:
        assert _parse_grid('{"num_leaves": [15, 31], "learning_rate": [0.1]}') == {
            "num_leaves": [15, 31],
            "learning_rate": [0.1],
        }

    def test_scalar_wrapped(self) -> None:
        assert _parse_grid('{"num_leaves": 31}') == {"num_leaves": [31]}

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _parse_grid("")

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty JSON object"):
            _parse_grid("[1, 2]")


class TestCombos:
    def test_cartesian_product(self) -> None:
        combos = _combos({"a": [1, 2], "b": [10, 20]})
        assert {tuple(sorted(c.items())) for c in combos} == {
            (("a", 1), ("b", 10)),
            (("a", 1), ("b", 20)),
            (("a", 2), ("b", 10)),
            (("a", 2), ("b", 20)),
        }

    def test_single(self) -> None:
        assert _combos({"a": [5]}) == [{"a": 5}]
