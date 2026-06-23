# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-23

### Added
- Initial `vgi-lightgbm` VGI worker exposing LightGBM to DuckDB/SQL.
- Model registry surface: `fit` (returns the model as a BLOB and persists it when
  `model_name` is given), `predict` (by `model_name` or inline `model` BLOB, with
  `with_proba`), `cross_val_predict`, `cross_val_score`, `list_models`,
  `model_info`, `drop_model`. Estimators: `lgbm_classifier`, `lgbm_regressor`.
- Typed `fit_lgbm_classifier` / `fit_lgbm_regressor` exposing LightGBM's common
  hyperparameters (`n_estimators`, `num_leaves`, `max_depth`, `learning_rate`,
  `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`,
  `boosting_type`, `objective`) as native typed SQL named arguments.
- First-class **categorical features**: string feature columns are detected and
  passed to LightGBM as native `categorical_feature`s; NULLs are kept as missing
  values.
- `grid_search`: cross-validated grid search over a JSON parameter grid, returning
  the leaderboard plus the refit best model as a BLOB on the best row.
- LightGBM-specific interpretation: `feature_importance` (split / gain) and SHAP
  `explain` in long format (regression, binary, and multiclass).
- Self-contained datasets (via scikit-learn): `iris`, `wine`, `breast_cancer`,
  `diabetes`, `california_housing`, `make_classification`, `make_regression`.
- Native LightGBM text serialization (no pickle) behind a swappable `ModelStore`
  (local disk now, S3/R2 seam).
- Fly.io deployment (Dockerfile installs from PyPI, no vendoring).
- Quality gate: ruff + mypy, pytest unit tests, and the `test/sql/*.test`
  integration suite over three transports (stdio, HTTP, in-process haybarn).
- GitHub Actions CI and Dependabot (pip / actions / docker).

[Unreleased]: https://github.com/rustyconover/vgi-lightgbm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rustyconover/vgi-lightgbm/releases/tag/v0.1.0
