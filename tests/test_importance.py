"""Unit tests for the LightGBM-specific extras (feature_importance).

``explain`` and the full streaming paths are covered end-to-end by
test/sql/lightgbm_importance.test; here we exercise the importance helper through
the in-process harness against a model saved in a temp registry.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from lightgbm import LGBMClassifier

from tests.harness import invoke_table_function
from vgi_lightgbm.importance import FeatureImportance
from vgi_lightgbm.registry import LocalDiskStore, ModelMetadata, set_store


@pytest.fixture()
def registry(tmp_path):
    store = LocalDiskStore(tmp_path)
    set_store(store)
    yield store
    set_store(None)


def _save_model(store: LocalDiskStore) -> None:
    rng = np.random.default_rng(0)
    # feature 0 perfectly separates the classes; features 1-2 are noise.
    x = np.column_stack([np.r_[np.zeros(40), np.ones(40)], rng.normal(size=80), rng.normal(size=80)])
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    est = LGBMClassifier(n_estimators=20, min_child_samples=1, verbosity=-1, random_state=0).fit(x, y)
    store.save(
        est,
        ModelMetadata(
            name="m",
            estimator="lgbm_classifier",
            task="classification",
            target="y",
            feature_names=["signal", "noise_a", "noise_b"],
            classes=[0, 1],
            categorical=[False, False, False],
            n_samples=80,
            n_features=3,
            lightgbm_version="x",
        ),
    )


class TestFeatureImportance:
    def test_ranks_signal_first(self, registry) -> None:
        _save_model(registry)
        table = invoke_table_function(FeatureImportance, positional=(pa.scalar("m"),))
        assert table.column("feature").to_pylist()[0] == "signal"
        assert table.column("rank").to_pylist() == [1, 2, 3]
        assert table.num_rows == 3

    def test_split_importance(self, registry) -> None:
        _save_model(registry)
        table = invoke_table_function(
            FeatureImportance,
            positional=(pa.scalar("m"),),
            named={"importance_type": pa.scalar("split")},
        )
        assert table.num_rows == 3

    def test_invalid_importance_type(self, registry) -> None:
        _save_model(registry)
        with pytest.raises(ValueError, match="invalid importance_type"):
            invoke_table_function(
                FeatureImportance,
                positional=(pa.scalar("m"),),
                named={"importance_type": pa.scalar("bogus")},
            )

    def test_unknown_model(self, registry) -> None:
        with pytest.raises(ValueError, match="not found"):
            invoke_table_function(FeatureImportance, positional=(pa.scalar("nope"),))
