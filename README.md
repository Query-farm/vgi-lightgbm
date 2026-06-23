<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-lightgbm/main/assets/vgi-logo.png" alt="Vector Gateway Interface" height="104">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-lightgbm/main/assets/lightgbm-logo.png" alt="LightGBM" height="52">
</p>

# vgi-lightgbm

[![CI](https://github.com/Query-farm/vgi-lightgbm/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-lightgbm/actions/workflows/ci.yml)

A [VGI](https://github.com/query-farm/vgi-python) worker that brings
[LightGBM](https://lightgbm.readthedocs.io/) into DuckDB/SQL: train gradient-boosted
models, persist them in a registry, predict over SQL tables, and interpret them
(feature importance + SHAP contributions) — all as SQL functions. LightGBM's
first-class **categorical-feature** support is exposed directly: string columns
become native categorical splits, no one-hot encoding required.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'lightgbm' (TYPE vgi, LOCATION 'uv run lightgbm_worker.py');

-- train + persist a model
SELECT model_name, task FROM lightgbm.fit(
  (SELECT * FROM lightgbm.iris()),
  model_name := 'iris_clf', estimator := 'lgbm_classifier', target := 'target', id := 'sample_id');

-- predict later
SELECT * FROM lightgbm.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id');
```

## How it maps LightGBM onto SQL

LightGBM is built around stateful *fit / predict* estimators; SQL is set-oriented.
Each piece is mapped to the VGI primitive that fits its data flow:

| Area | SQL surface | VGI primitive |
| --- | --- | --- |
| **Datasets** | `SELECT * FROM lightgbm.iris()` | table function (source) |
| **Fit** | `lightgbm.fit((SELECT ...), model_name := 'm', ...)` | table-buffering → BLOB + registry |
| **Typed fit** | `lightgbm.fit_lgbm_classifier((SELECT ...), num_leaves := 63, ...)` | table-buffering |
| **Predict** | `lightgbm.predict((SELECT ...), model_name := 'm')` | streaming table-in-out |
| **Cross-val** | `lightgbm.cross_val_predict(...)` / `lightgbm.cross_val_score(...)` | table-buffering |
| **Search** | `lightgbm.grid_search((SELECT ...), grid := '{...}')` | table-buffering |
| **Importance** | `lightgbm.feature_importance('m')` | table function (reads the registry) |
| **Explain (SHAP)** | `lightgbm.explain((SELECT ...), model_name := 'm')` | streaming table-in-out |

**Conventions** for the fit / predict / explain functions:

- The input relation **is** the feature matrix `X`, passed as a `(SELECT ...)`
  subquery. Named arguments use DuckDB's `name := value` (or `=>`) syntax.
- **`id`** names a passthrough column: it is *excluded from the features* and
  copied unchanged onto each output row, so you can join results back to the
  source. It is optional.
- **`target`** (required for `fit` / cross-val) names the label column, also
  excluded from features. Classification targets are integer class labels encoded
  `0..n_classes-1`; the bundled datasets already are.
- **Every remaining column is a feature.** Numeric and boolean columns are used as
  numeric features; **string columns become native LightGBM categorical features**
  (detected automatically). NULLs are kept as missing values, which LightGBM
  handles natively. Other column types raise a clear error — `SELECT` only the
  columns you want as features.
- Hyperparameters are passed as a JSON string: `params := '{"n_estimators": 300, "num_leaves": 63}'`.
  Unknown hyperparameters are rejected with the list of valid ones. The typed
  `fit_lgbm_*` functions expose the common ones as native named arguments instead.
- **`fit`/`predict` align features by name**, not position: `predict` selects the
  model's fitted feature columns by name (input order is irrelevant, extra columns
  are ignored) and errors if a required feature column is missing.

## Model BLOBs and the registry

`fit` (and the typed `fit_lgbm_*` functions) **always return the trained model as
a `model` BLOB**, and *additionally* persist it to the registry when you pass a
`model_name`. `predict`, `explain`, and `feature_importance` take **either** a
`model_name :=` (registry lookup) **or** a `model :=` BLOB (inline).

Because a DuckDB table function may only take one subquery argument (the input
table), pass a BLOB through a session variable:

```sql
-- fit without persisting; capture the BLOB
SET VARIABLE m = (SELECT model FROM lightgbm.fit(
  (SELECT * FROM lightgbm.diabetes()), estimator := 'lgbm_regressor', target := 'target', id := 'sample_id'));

-- predict with the inline model
SELECT * FROM lightgbm.predict(
  (SELECT sample_id, age, sex, bmi, bp, s1, s2, s3, s4, s5, s6 FROM lightgbm.diabetes()),
  model := getvariable('m'), id := 'sample_id');
```

## Function catalog

### Datasets (`lightgbm.<name>()`)
Bundled (via scikit-learn) so demos and tests are self-contained: `iris`, `wine`,
`breast_cancer` (classification), `diabetes`, `california_housing` (regression),
and generators `make_classification`, `make_regression`.

```sql
SELECT target_name, avg(petal_length_cm) FROM lightgbm.iris() GROUP BY target_name;
SELECT * FROM lightgbm.make_classification(n_samples := 500, n_features := 8, n_classes := 3);
```

### Models (registry-backed)
`fit`, `fit_lgbm_classifier`, `fit_lgbm_regressor`, `predict`, `cross_val_predict`,
`cross_val_score`, `grid_search`, `list_models`, `model_info`, `drop_model`.

Estimators: `lgbm_classifier`, `lgbm_regressor`.

```sql
-- train + persist
SELECT model_name, task, n_features FROM lightgbm.fit(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM lightgbm.iris()),
  model_name := 'iris_clf', estimator := 'lgbm_classifier', target := 'target', id := 'sample_id',
  params := '{"n_estimators": 200, "num_leaves": 31}');

-- typed fit: native hyperparameters as SQL named args
SELECT model_name, task FROM lightgbm.fit_lgbm_classifier(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM lightgbm.iris()),
  model_name := 'iris_typed', target := 'target', id := 'sample_id',
  n_estimators := 300, num_leaves := 63, learning_rate := 0.05, boosting_type := 'gbdt');

-- predict later (optionally with per-class probabilities)
SELECT * FROM lightgbm.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id', with_proba := true);

-- evaluate without persisting
SELECT count(*) FROM lightgbm.cross_val_predict(
  (SELECT * FROM lightgbm.iris()), estimator := 'lgbm_classifier', target := 'target', id := 'sample_id', cv := 5);

SELECT fold, score FROM lightgbm.cross_val_score(
  (SELECT * FROM lightgbm.iris()), estimator := 'lgbm_classifier', target := 'target', cv := 5);

-- grid search: leaderboard + the refit best model BLOB on the best row
SELECT params, mean_score, rank FROM lightgbm.grid_search(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM lightgbm.iris()),
  estimator := 'lgbm_classifier', target := 'target',
  grid := '{"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1]}', cv := 4)
ORDER BY rank;

SELECT * FROM lightgbm.list_models();
SELECT * FROM lightgbm.drop_model('iris_clf');
```

### Categorical features
String feature columns are passed to LightGBM as native categorical features —
no manual encoding. The encoding is stored with the model so `predict` replays it.

```sql
SELECT model_name, n_categorical FROM lightgbm.fit(
  (SELECT id, size, color, target FROM widgets),  -- `color`/`size` may be strings
  model_name := 'widget_clf', target := 'target', id := 'id');
```

### Interpretation (LightGBM-specific)
`feature_importance` and `explain`.

```sql
-- ranked per-feature importance for a model (split count or total gain)
SELECT * FROM lightgbm.feature_importance('iris_clf', importance_type := 'gain');

-- SHAP contributions in long format: (id, [class], feature, shap_value, base_value).
-- base_value + sum(shap_value) == the model's raw-margin prediction (per row, per class).
SELECT * FROM lightgbm.explain((SELECT * FROM lightgbm.diabetes()), model_name := 'diab_reg', id := 'sample_id');
```

## Model registry storage

Fitted models are serialized with **LightGBM's native text format (not pickle)**
plus a JSON metadata sidecar. The store is chosen behind the `ModelStore`
interface in `vgi_lightgbm/registry.py`:

- **Local disk** (default): `LIGHTGBM_MODELS_DIR` (default `./models`).
- **S3 / Cloudflare R2**: not yet implemented — `get_store()` is the single seam
  where an `S3Store` drops in.

On Fly.io the local store is backed by a mounted volume (see `fly.toml`) so models
survive machine restarts. `predict` records the LightGBM version used to fit and
logs a warning (visible in `duckdb_logs()`) if the worker's version differs.

## Local development

```sh
make venv          # create .venv with vgi + lightgbm + scikit-learn (from PyPI)
make lint          # ruff + mypy
make pytest        # unit tests
make test-sql      # SQL tests in-process via haybarn (no custom DuckDB build needed)
make test-stdio    # SQL tests with the worker as a subprocess (custom unittest runner)
make test-http     # SQL tests against a local HTTP server
```

The `test/sql/*.test` files are the integration suite. `test-stdio`/`test-http`
run them with DuckDB's `unittest` runner built with the VGI extension
(`VGI_BUILD_DIR`) and are the local authority. `test-sql` replays the **same**
files in-process against the `haybarn` DuckDB distribution (which can
`INSTALL vgi FROM community`), so they also run on a stock CI runner.

### Continuous integration
`.github/workflows/ci.yml` runs ruff, mypy, the unit tests, the haybarn SQL suite,
and a Docker build + `/health` smoke test on every push and PR. Dependabot
(`.github/dependabot.yml`) keeps the Python deps, GitHub Actions, and the Docker
base image up to date weekly.

