"""Unit tests for the discriminated-union grid/randomized search helpers."""

from __future__ import annotations

import pyarrow as pa

from vgi_lightgbm.search import (
    _GRID_UNION,
    SEARCH_FUNCTIONS,
    _combos,
    _grid_size,
    _member_struct,
    _param_grid,
    _select_combos,
)
from vgi_lightgbm.typed_models import _HPARAMS


class TestGridUnion:
    def test_one_member_per_estimator(self) -> None:
        members = {f.name for f in _GRID_UNION}
        assert members == set(_HPARAMS)

    def test_member_fields_match_hparams(self) -> None:
        for spec in _HPARAMS.values():
            struct = _member_struct(spec)
            assert [f.name for f in struct] == [hp.name for hp in spec]

    def test_member_field_types_are_lists_of_scalar(self) -> None:
        spec = _HPARAMS["lgbm_classifier"]
        struct = _member_struct(spec)
        by_name = {f.name: f.type for f in struct}
        # num_leaves is int -> list<int64>; learning_rate is float -> list<double>
        assert by_name["num_leaves"] == pa.list_(pa.int64())
        assert by_name["learning_rate"] == pa.list_(pa.float64())
        assert by_name["boosting_type"] == pa.list_(pa.string())


class TestParamGrid:
    def test_only_set_params_searched(self) -> None:
        grid = _param_grid("lgbm_classifier", {"num_leaves": [15, 31]})
        assert grid == {"num_leaves": [15, 31]}

    def test_max_depth_zero_maps_to_unlimited(self) -> None:
        grid = _param_grid("lgbm_classifier", {"max_depth": [0, 3, 5]})
        assert grid == {"max_depth": [-1, 3, 5]}

    def test_empty_objective_maps_to_none(self) -> None:
        grid = _param_grid("lgbm_classifier", {"objective": ["", "multiclass"]})
        assert grid == {"objective": [None, "multiclass"]}

    def test_null_member_value_yields_empty_grid(self) -> None:
        assert _param_grid("lgbm_classifier", None) == {}


class TestGridSize:
    def test_product(self) -> None:
        assert _grid_size({"a": [1, 2], "b": [10, 20, 30]}) == 6

    def test_single(self) -> None:
        assert _grid_size({"a": [5]}) == 1


class TestCombos:
    def test_cartesian_product(self) -> None:
        combos = _combos({"a": [1, 2], "b": [10, 20]})
        assert {tuple(sorted(c.items())) for c in combos} == {
            (("a", 1), ("b", 10)),
            (("a", 1), ("b", 20)),
            (("a", 2), ("b", 10)),
            (("a", 2), ("b", 20)),
        }


class TestSelectCombos:
    def test_caps_at_grid_size(self) -> None:
        combos = _combos({"a": [1, 2, 3]})
        assert len(_select_combos(combos, n_iter=100, random_state=0)) == 3

    def test_samples_n_iter(self) -> None:
        combos = _combos({"a": list(range(10))})
        picked = _select_combos(combos, n_iter=4, random_state=0)
        assert len(picked) == 4

    def test_deterministic_for_seed(self) -> None:
        combos = _combos({"a": list(range(10))})
        assert _select_combos(combos, 4, 0) == _select_combos(combos, 4, 0)


def test_search_functions_registered() -> None:
    names = {f.Meta.name for f in SEARCH_FUNCTIONS}
    assert names == {"grid_search", "randomized_search"}
