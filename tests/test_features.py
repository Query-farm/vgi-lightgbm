"""Unit tests for categorical-feature detection and encoding."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from vgi_lightgbm.features import (
    categorical_indices,
    detect_categoricals,
    encode_matrix,
    fit_categories,
    validate_features,
)


def _table() -> pa.Table:
    return pa.table(
        {
            "num": [1.0, 2.0, 3.0, None],
            "color": ["red", "blue", "red", None],
            "flag": [True, False, True, False],
        }
    )


class TestCategoricalDetection:
    def test_detect(self) -> None:
        t = _table()
        feats = ["num", "color", "flag"]
        mask = detect_categoricals(t, feats)
        assert mask == [False, True, False]
        assert categorical_indices(mask) == [1]

    def test_fit_categories_sorted_and_distinct(self) -> None:
        t = _table()
        cats = fit_categories(t, ["num", "color", "flag"], [False, True, False])
        assert cats == {"color": ["blue", "red"]}


class TestEncodeMatrix:
    def test_encodes_categoricals_and_preserves_nan(self) -> None:
        t = _table()
        feats = ["num", "color", "flag"]
        mask = detect_categoricals(t, feats)
        cats = fit_categories(t, feats, mask)
        x = encode_matrix(t, feats, mask, cats)
        # color: red->1, blue->0 (sorted ['blue','red']); NULL -> -1
        assert x[:, 1].tolist() == [1.0, 0.0, 1.0, -1.0]
        # numeric NULL becomes NaN
        assert np.isnan(x[3, 0])
        # bool becomes float
        assert x[:, 2].tolist() == [1.0, 0.0, 1.0, 0.0]

    def test_unseen_category_maps_to_minus_one(self) -> None:
        feats = ["color"]
        mask = [True]
        cats = {"color": ["blue", "red"]}
        t = pa.table({"color": ["green", "red"]})
        x = encode_matrix(t, feats, mask, cats)
        assert x[:, 0].tolist() == [-1.0, 1.0]


class TestValidateFeatures:
    def test_accepts_supported_types(self) -> None:
        validate_features(_table(), ["num", "color", "flag"])

    def test_missing_column(self) -> None:
        with pytest.raises(ValueError, match="missing required"):
            validate_features(_table(), ["nope"])

    def test_unsupported_type(self) -> None:
        t = pa.table({"ts": pa.array([[1, 2]], type=pa.list_(pa.int64()))})
        with pytest.raises(ValueError, match="unsupported types"):
            validate_features(t, ["ts"])
