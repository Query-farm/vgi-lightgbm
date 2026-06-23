"""Unit tests for the generated typed ``fit_lgbm_*`` functions."""

from __future__ import annotations

import pytest

from vgi_lightgbm.models import _ESTIMATORS
from vgi_lightgbm.typed_models import _HPARAMS, TYPED_FIT_FUNCTIONS, _estimator_kwargs, _make_args_class


def test_one_typed_function_per_estimator() -> None:
    names = {f.Meta.name for f in TYPED_FIT_FUNCTIONS}
    assert names == {"fit_lgbm_classifier", "fit_lgbm_regressor"}


@pytest.mark.parametrize("est_name", list(_HPARAMS))
def test_typed_params_are_valid_for_estimator(est_name: str) -> None:
    """Every exposed typed hyperparameter must be a real estimator param."""
    _task, cls, _defaults = _ESTIMATORS[est_name]
    valid = set(cls().get_params().keys())
    for hp in _HPARAMS[est_name]:
        assert hp.name in valid, f"{hp.name} is not a valid {est_name} param"


def test_max_depth_zero_maps_to_unlimited() -> None:
    spec = _HPARAMS["lgbm_classifier"]
    args_cls = _make_args_class("lgbm_classifier", spec)
    # build an args object with all defaults except max_depth := 0
    kwargs = {hp.name: hp.default for hp in spec}
    kwargs.update({"data": None, "model_name": "", "target": "y", "id": ""})
    args = args_cls(**kwargs)
    est_kwargs = _estimator_kwargs(spec, args)
    assert est_kwargs["max_depth"] == -1  # 0 -> unlimited


def test_empty_objective_drops_to_default() -> None:
    spec = _HPARAMS["lgbm_classifier"]
    args_cls = _make_args_class("lgbm_classifier", spec)
    kwargs = {hp.name: hp.default for hp in spec}
    kwargs.update({"data": None, "model_name": "", "target": "y", "id": ""})
    args = args_cls(**kwargs)
    est_kwargs = _estimator_kwargs(spec, args)
    assert est_kwargs["objective"] is None  # '' -> None (LightGBM default)
