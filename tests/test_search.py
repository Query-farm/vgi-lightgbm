"""Unit tests for the grid/randomized search helpers."""

from __future__ import annotations

import pyarrow as pa

from vgi_lightgbm.search import (
    SEARCH_FUNCTIONS,
    GridSearchArgs,
    RandomizedSearchArgs,
    _combos,
    _grid_size,
    _param_grid,
    _search_param_fields,
    _select_combos,
)
from vgi_lightgbm.typed_models import _HPARAMS


class _Args:
    """Stand-in for the generated args dataclass: attributes are value lists."""

    def __init__(self, **kw: object) -> None:
        self.__dict__.update(kw)


class TestSearchParamArguments:
    def test_one_argument_per_hyperparameter(self) -> None:
        names = [name for name, _type, _default in _search_param_fields()]
        assert set(names) == {hp.name for spec in _HPARAMS.values() for hp in spec}

    def test_each_argument_is_a_list_of_its_scalar_type(self) -> None:
        by_name = {name: annotated for name, annotated, _default in _search_param_fields()}

        # The Arg metadata carries the explicit Arrow type for the list argument.
        def arrow_of(name: str) -> pa.DataType:
            return by_name[name].__metadata__[0].arrow_type

        assert arrow_of("num_leaves") == pa.list_(pa.int64())
        assert arrow_of("learning_rate") == pa.list_(pa.float64())
        assert arrow_of("boosting_type") == pa.list_(pa.string())

    def test_estimator_is_a_plain_string_on_both_functions(self) -> None:
        for cls in (GridSearchArgs, RandomizedSearchArgs):
            annotated = cls.__annotations__["estimator"]
            assert annotated.__origin__ is str
            assert annotated.__metadata__[0].choices == sorted(_HPARAMS)


class TestParamGrid:
    def test_only_set_params_searched(self) -> None:
        grid = _param_grid("lgbm_classifier", _Args(num_leaves=[15, 31]))
        assert grid == {"num_leaves": [15, 31]}

    def test_max_depth_zero_maps_to_unlimited(self) -> None:
        grid = _param_grid("lgbm_classifier", _Args(max_depth=[0, 3, 5]))
        assert grid == {"max_depth": [-1, 3, 5]}

    def test_empty_objective_maps_to_none(self) -> None:
        grid = _param_grid("lgbm_classifier", _Args(objective=["", "multiclass"]))
        assert grid == {"objective": [None, "multiclass"]}

    def test_no_values_yields_empty_grid(self) -> None:
        assert _param_grid("lgbm_classifier", _Args()) == {}
        assert _param_grid("lgbm_classifier", _Args(num_leaves=[])) == {}


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
