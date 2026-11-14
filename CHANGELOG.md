# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Composed example queries that combine the worker's own functions: an
  out-of-fold confusion matrix (`cross_val_predict` joined back to `iris`), gain
  vs permutation importance side by side (`feature_importance` +
  `permutation_importance`), and per-species SHAP drivers (`explain` joined to
  `iris`). Published on the driving function, the schema example list and (the
  offline one) `vgi.executable_examples`, plus a README snippet and a new
  `breast_cancer_oof_missed_malignant` agent task.

### Changed
- **Ported to VGI protocol 3.2** (`vgi-python` 0.30.0 -> 0.50.0, `vgi-rpc`
  0.44.1 -> 0.53.1) so the worker attaches with the current community `vgi`
  extension on haybarn 2.0. Table keys are now declared as canonical vgi.v3
  constraint records.
- Agent-task grader SQL moved out of the published `vgi.agent_test_tasks` tag
  into `vgi-agent-tests.yaml`, so the worker passes `vgi-lint-check` 2.x
  (haybarn 2.0 / vgi.v3).

### Fixed
- `permutation_importance`'s example selected a non-existent `importance`
  column (the output is `importance_mean`), so it failed to bind; it was hidden
  by the VGI901/902 lint exemption for stored-model functions. A new haybarn
  test now runs every published example against stored models.
- Examples that fit on `SELECT * FROM iris` (the schema/executable `fit`
  examples, the typed `fit_lgbm_*` examples, README `cross_val_*`) leaked the
  label into the features through the `target_name` string column; they now
  `EXCLUDE (target_name)`.
- **The worker could no longer be attached.** `vgi-python` 0.8.5 predates the
  `attach_catalogs` field the current community `vgi` extension requires, so
  every `ATTACH` failed with *"Worker returned an out-of-date Apache Arrow
  schema. RPC response schema mismatch for 'catalog_attach' (field count
  differs: expected 15, actual 14)"*. Bumped to `vgi-python>=0.17.0` /
  `vgi-rpc>=0.26.1`.

### Added
- `vgi.agent_test_tasks`: an 18-task agent-suitability suite covering every
  function the worker exposes, for `vgi-lint simulate`.
- Per-schema `vgi.categories` navigation registry and a `vgi.category` on every
  table and function.
- Per-column units/definitions on the wine, breast-cancer, diabetes and
  California-housing feature columns.
- `randomized_search`: cross-validated random hyperparameter search (`n_iter`,
  `random_state`, capped at the grid size).
- `permutation_importance`: model-agnostic ranked feature importance.
- `partial_dependence`: average prediction vs. one numeric feature (multiclass ->
  one curve per class).
- `predict` output modes: `output_margin` (raw margin score) and `pred_leaf`
  (per-tree leaf indices), mutually exclusive.

### Changed
- **Breaking:** `grid_search` / `randomized_search` now take a plain
  `estimator := 'lgbm_classifier'` plus one typed, list-valued argument per swept
  hyperparameter (`num_leaves := [15, 31]`), replacing the
  `union_value(<estimator> := {...})` tagged union. `randomized_search`'s sampler
  seed is now `search_seed` (`random_state` is itself a searchable
  hyperparameter).
- **Breaking:** the regression datasets name their target for what it is —
  `diabetes.progression` and `california_housing.median_house_value` — so the
  catalog-wide `target` column is one type (the classification class label).
- **Breaking:** `list_models` is now `models`, and its `target` column is
  `target_column` (it holds a column *name*).
- Retired `vgi.result_columns_md` tags migrated to the structured
  `vgi.result_columns_schema` / `vgi.result_dynamic_columns_md`.
- (superseded) `grid_search` took the estimator + grid as a typed
  discriminated-union argument (`estimator := union_value(<estimator> := {param:
  [values]})`) instead of the JSON `grid := '...'` string; only the listed
  hyperparameters are searched.
- Classification targets accept **any label dtype** (integer codes or strings):
  labels are encoded internally and `predict` / `cross_val_predict` decode back to
  the original labels (the `prediction` column is `VARCHAR` for string labels,
  `BIGINT` otherwise; `with_proba` names columns `proba_<label>`).
- `feature_importance` and `permutation_importance` are both ranked.
- `model_info` integer columns are `BIGINT` and `task` is a plain string (matching
  the `fit` summary schema).
- Normalized SQL examples / docstrings to the `:=` named-argument syntax.

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

[Unreleased]: https://github.com/Query-farm/vgi-lightgbm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Query-farm/vgi-lightgbm/releases/tag/v0.1.0
