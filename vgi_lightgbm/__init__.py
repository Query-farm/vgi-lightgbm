"""LightGBM as a VGI worker: a supervised train/predict model registry for DuckDB/SQL.

LightGBM's value is fast, accurate gradient-boosted train/predict (with first-class
categorical-feature support), so the worker is built around a model registry rather
than the broad datasets/metrics/transforms surface of a general ML library. The
implementation is split by area:

- ``datasets``      -- a few reference datasets + generators, so demos and the SQL
  tests are self-contained (reuses scikit-learn's bundled data)
- ``models``        -- supervised ``fit`` / ``predict`` / ``cross_val_predict`` /
  ``cross_val_score`` and the model registry, with LightGBM estimators. ``fit``
  always returns the model as a BLOB and persists it when ``model_name`` is given;
  ``predict`` accepts either ``model_name`` or a ``model`` BLOB.
- ``typed_models``  -- generated ``fit_lgbm_classifier`` / ``fit_lgbm_regressor``
  exposing LightGBM's real hyperparameters as native typed SQL named arguments
- ``search``        -- ``grid_search`` over a single estimator (JSON parameter grid)
- ``importance``    -- LightGBM-specific extras: ``feature_importance`` (split/gain)
  and SHAP ``explain`` (per-row, long-format feature contributions)
- ``registry``      -- pluggable model store (local disk now, S3/R2 later); models
  are serialized with LightGBM's native text format, not pickle

``lightgbm_worker.py`` at the repo root assembles these into the ``lightgbm``
catalog and runs the worker.
"""

from __future__ import annotations

__version__ = "0.1.0"
