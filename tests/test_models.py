"""Unit tests for the model registry, estimator catalog, and BLOB pack/unpack.

The full fit -> predict -> list -> drop lifecycle is covered end-to-end by
test/sql/lightgbm_models.test; here we test the storage backend and helpers.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from lightgbm import LGBMClassifier

from vgi_lightgbm.models import _label_dtype, _parse_params, build_estimator, encode_labels
from vgi_lightgbm.registry import (
    LocalDiskStore,
    ModelBlobError,
    ModelMetadata,
    ModelNameError,
    ModelNotFoundError,
    pack_model,
    unpack_meta,
    unpack_model,
    validate_name,
)

_TRAIN_X = np.array([[0.0], [1.0], [0.0], [1.0], [0.2], [0.9], [0.1], [0.8]])
_TRAIN_Y = np.array([0, 1, 0, 1, 0, 1, 0, 1])


def _fitted() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=5, min_child_samples=1, verbosity=-1, random_state=0).fit(_TRAIN_X, _TRAIN_Y)


def _meta(name: str = "m") -> ModelMetadata:
    return ModelMetadata(
        name=name,
        estimator="lgbm_classifier",
        task="classification",
        target="y",
        feature_names=["a"],
        classes=[0, 1],
        categorical=[False],
        n_samples=8,
        n_features=1,
        train_score=1.0,
        lightgbm_version="x",
        created_at="now",
    )


class TestLocalDiskStore:
    def test_roundtrip(self, tmp_path) -> None:
        original = _fitted()
        store = LocalDiskStore(tmp_path)
        store.save(original, _meta())
        assert store.exists("m")
        booster, meta = store.load("m")
        assert meta.name == "m"
        assert meta.feature_names == ["a"]
        assert meta.classes == [0, 1]
        # the reloaded booster's raw output matches the original estimator's P(class=1)
        reloaded = np.asarray(booster.predict(_TRAIN_X)).reshape(-1)
        original_p1 = original.predict_proba(_TRAIN_X)[:, 1]
        assert np.allclose(reloaded, original_p1, atol=1e-9)

    def test_list(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta("a"))
        store.save(_fitted(), _meta("b"))
        assert sorted(m.name for m in store.list()) == ["a", "b"]

    def test_delete(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta())
        assert store.delete("m") is True
        assert store.delete("m") is False
        assert not store.exists("m")

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(ModelNotFoundError):
            LocalDiskStore(tmp_path).load("nope")


class TestModelBlob:
    def test_pack_unpack_roundtrip(self) -> None:
        est = _fitted()
        blob = pack_model(est, _meta("blobby"))
        meta = unpack_meta(blob)
        assert meta.name == "blobby"
        assert meta.feature_names == ["a"]
        booster, meta2 = unpack_model(blob)
        assert meta2.task == "classification"
        reloaded = np.asarray(booster.predict(_TRAIN_X)).reshape(-1)
        assert np.allclose(reloaded, est.predict_proba(_TRAIN_X)[:, 1], atol=1e-9)

    def test_bad_blob_raises(self) -> None:
        with pytest.raises(ModelBlobError):
            unpack_meta(b"not a model blob")


class TestValidateName:
    def test_accepts_reasonable(self) -> None:
        assert validate_name("iris_clf-1.2") == "iris_clf-1.2"

    @pytest.mark.parametrize("bad", ["", "../etc", "a/b", ".hidden", "with space"])
    def test_rejects_unsafe(self, bad: str) -> None:
        with pytest.raises(ModelNameError):
            validate_name(bad)


class TestEstimatorCatalog:
    def test_build_with_params(self) -> None:
        task, est = build_estimator("lgbm_classifier", {"n_estimators": 7})
        assert task == "classification"
        assert est.n_estimators == 7

    def test_regression_task(self) -> None:
        task, est = build_estimator("lgbm_regressor", {})
        assert task == "regression"

    def test_unknown_estimator(self) -> None:
        with pytest.raises(ValueError, match="unknown estimator"):
            build_estimator("does_not_exist", {})

    def test_unknown_hyperparameter(self) -> None:
        with pytest.raises(ValueError, match="unknown hyperparameter"):
            build_estimator("lgbm_classifier", {"nonsense": 5})


class TestEncodeLabels:
    def test_int_labels_roundtrip(self) -> None:
        codes, classes = encode_labels([2, 0, 1, 2, 0])
        assert classes == [0, 1, 2]
        assert list(codes) == [2, 0, 1, 2, 0]
        # decode is classes[code]
        assert [classes[c] for c in codes] == [2, 0, 1, 2, 0]

    def test_string_labels_sorted_and_decode(self) -> None:
        codes, classes = encode_labels(["b", "a", "c", "a"])
        assert classes == ["a", "b", "c"]
        assert [classes[c] for c in codes] == ["b", "a", "c", "a"]

    def test_string_labels_give_varchar_prediction(self) -> None:
        _codes, classes = encode_labels(["setosa", "versicolor"])
        assert _label_dtype(classes) == pa.string()

    def test_int_labels_give_bigint_prediction(self) -> None:
        _codes, classes = encode_labels([0, 1, 2])
        assert _label_dtype(classes) == pa.int64()

    def test_no_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="no non-null labels"):
            encode_labels([None, None])


class TestParseParams:
    def test_empty(self) -> None:
        assert _parse_params("") == {}
        assert _parse_params("   ") == {}

    def test_json_object(self) -> None:
        assert _parse_params('{"n_estimators": 200, "num_leaves": 15}') == {"n_estimators": 200, "num_leaves": 15}

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _parse_params("[1, 2, 3]")
