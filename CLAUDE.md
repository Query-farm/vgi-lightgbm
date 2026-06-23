# CLAUDE.md — vgi-lightgbm

Contributor/agent notes for this repo. User-facing docs live in `README.md`;
this file is the "how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://github.com/query-farm/vgi-python) worker exposing LightGBM to
DuckDB/SQL. `lightgbm_worker.py` assembles every function into one `lightgbm`
catalog (single `main` schema) and runs it over stdio (local) or HTTP (Fly.io).
Depends on the published `vgi-python` / `vgi-rpc` from PyPI; modeled on the
sibling `~/Development/vgi-xgboost` worker, with the depth (typed fit functions,
grid search, model BLOBs, categorical features) of `~/Development/vgi-scikit-learn`.

LightGBM's value is fast, accurate gradient-boosted train/predict with native
categorical-feature support, so this worker is focused on a **model registry**
(fit/predict/cross-val), **typed fit + discriminated-union hyperparameter search**
(grid + randomized), and interpretation (feature_importance, SHAP `explain`,
permutation_importance, partial_dependence). It deliberately does *not* mirror
vgi-sklearn's
metrics/transforms surface — those are scikit-learn's. A small datasets module
(reusing scikit-learn's bundled data) keeps demos and the SQL tests self-contained.

## Layout

```
lightgbm_worker.py    entry point: builds the `lightgbm` Catalog, LightGBMWorker, main()
serve.py              HTTP entry point (injects --http into Worker.main())
vgi_lightgbm/
  datasets.py         dataset table functions (toy sets + make_* generators)
  models.py           fit / predict / cross_val_predict / cross_val_score + registry mgmt + _fit_and_emit
  typed_models.py     generated fit_lgbm_<task> functions with typed hyperparams
  search.py           grid_search / randomized_search (discriminated-union estimator arg via _GRID_UNION/_HPARAMS)
  importance.py       feature_importance + explain (SHAP, long) + permutation_importance + partial_dependence
  features.py         categorical (string) detection + integer encoding for LightGBM
  registry.py         ModelStore + LocalDiskStore (S3/R2 seam) + native-text (de)serialize + model BLOB pack/unpack
  buffering.py        shared sink/combine/serialize/matrix helpers
  schema_utils.py     pa.Field comment helper, name sanitisation, NoArgs
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_lightgbm/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_FUNCTIONS` in `lightgbm_worker.py`.

## Which VGI primitive for which job

| Need | Primitive | Example here |
| --- | --- | --- |
| Emit rows, no input | `TableFunctionGenerator` (`@bind_fixed_schema` / `@init_single_worker`, or custom `on_bind` for schema-from-args) | `datasets.py`, `importance.FeatureImportance` |
| `fit` / buffer whole input | `TableBufferingFunction` via `buffering.SinkBuffer` | `models.FitModel`, `CrossValPredict`, `search.GridSearch`/`RandomizedSearch`, `importance.PermutationImportance`/`PartialDependence` |
| Score/explain a stream with an already-fit model | `TableInOutGenerator` | `models.PredictModel`, `importance.ExplainModel` |

Conventions for fit/predict/explain: input relation is X via a `(SELECT ...)`
subquery (Arg(0)); name `target` (features = the rest) and an optional `id`
passthrough; hyperparameters as a JSON-string arg on the generic `fit` (the typed
`fit_lgbm_*` expose them as named args instead). `grid_search`/`randomized_search`
instead take a typed **discriminated-union** estimator arg (see below), not JSON.

## Models: registry + BLOB + typed functions + categoricals

- **fit always returns a `model` BLOB** (booster text + metadata packed by
  `registry.pack_model`) and persists to the registry only if `model_name` is
  given (so `model_name` is optional). `predict`/`explain`/`feature_importance`
  take **either** `model_name :=` or `model :=` (a BLOB); `registry.unpack_meta`
  reads metadata at bind, `unpack_model` rebuilds the `Booster` at process.
- **Serialization is LightGBM's native text, not pickle**
  (`registry.booster_to_text`/`booster_from_text`, `.lgb` files + text in the
  BLOB). No arbitrary-code-execution risk; portable across LightGBM patch
  versions. The fitted sklearn wrapper is reduced to its `Booster`; classes and
  task live in metadata so prediction (label / proba / argmax) is reproduced from
  the raw booster (`models._proba`). `predict` warns via `duckdb_logs()` if the
  worker's LightGBM version differs from the one a model was fit with.
- **Typed `fit_lgbm_<task>` functions** are generated in `typed_models.py` from
  the `_HPARAMS` spec via `types.new_class(name, (SinkBuffer[args, DrainState],),
  ...)` — plain `type()` can't resolve the subscripted-generic base. Both share
  `models._fit_and_emit`. To add/adjust hyperparameters, edit `_HPARAMS`; the
  `test_typed_params_are_valid_for_estimator` test guards that every exposed param
  is real for its estimator. `max_depth := 0` maps to `-1` (LightGBM's
  "unlimited"); `objective := ''` keeps LightGBM's task default.
- **predict aligns features by name** (reorder-safe, extra columns ignored);
  missing feature columns raise clear errors at bind.
- **Classification labels can be any dtype** (string, int, bool). `models.encode_labels`
  builds a stable sorted `classes` list and label-encodes the target to `0..n-1`
  codes; `ModelMetadata.classes` stores the *original* labels. `predict` /
  `cross_val_predict` decode `code → classes[code]` and type the `prediction`
  column from the label dtype (VARCHAR for strings, BIGINT for ints); `with_proba`
  emits `proba_<original_label>` columns. (Replaces the old "round to int" path.)
- **predict output modes:** default label/value; `with_proba` (per-class probs,
  classifiers only); `output_margin` (raw margin via `raw_score=True`, a `margin`
  float64 column); `pred_leaf` (one leaf index per tree as a `list<int32>` column).
  `output_margin`/`pred_leaf` are mutually exclusive and incompatible with
  `with_proba` (validated at bind). Same arg names as vgi-xgboost.
- **Native categorical (string) features** (`features.py`). At fit, string columns
  are detected (`detect_categoricals`), the sorted distinct values per column are
  learned (`fit_categories`), and `encode_matrix` integer-encodes them; the
  categorical column indices are passed to LightGBM as `categorical_feature` so it
  learns native categorical splits. The per-feature `categorical` mask and the
  `categories` orderings are stored in `ModelMetadata`, so `predict` re-encodes new
  rows identically (unseen/NULL category -> `-1`, LightGBM's unknown-category
  sentinel). Uniform across `fit`, `fit_lgbm_*`, `grid_search`/`randomized_search`,
  `permutation_importance`, and `partial_dependence`. `n_features` is the *original*
  feature count (LightGBM never one-hot-expands, so there's no width blow-up — that's
  the whole point). Numeric NULLs are kept as NaN (LightGBM handles missing values
  natively).
- **`search.grid_search` / `randomized_search` are a discriminated union** (same
  design as vgi-sklearn's `search.py`). The `estimator` arg is a sparse Arrow union
  `_GRID_UNION` (one member per estimator built from `_HPARAMS`, each field a
  `list<scalar>`); SQL calls it `union_value(<estimator> := {param: [values]})`.
  The worker reads it as a `vgi.TaggedUnion` (`.tag` = estimator, `.value` = grid
  dict); `_param_grid` translates a member into a param grid, applying the same
  per-param mapping as typed fit element-wise (`max_depth` 0→-1, `objective` ""→
  default); omitted params stay at default. Returns the CV leaderboard (one row per
  combo, ranked) with the refit best model BLOB on the single best row — grab it
  with `WHERE model IS NOT NULL`. `randomized_search` adds `n_iter` (sampled,
  capped at the grid size via `_select_combos`) + `random_state`. **Requires
  `vgi-python >=0.8.3`** (ships `vgi.TaggedUnion` / union-tag-preserving decode) —
  already the pin. Dense unions are unsupported by the C++ extension; `union_value`
  produces sparse, which works.
- **`feature_importance` and `permutation_importance` are ranked** (a `rank` int32
  column, sorted by importance desc). `partial_dependence` (`importance.py`,
  buffering) shows how a feature moves the prediction over a grid: numeric-only
  (categorical → clear error), output `(feature_value, class, partial_dependence)`,
  one curve per class for multiclass / NULL `class` for regression+binary. Both
  `permutation_importance` and `partial_dependence` reconstruct a small
  sklearn-compatible wrapper from the stored booster text (the registry holds no
  pickled estimator) to run `sklearn.inspection.*`.
- **Schema consistency:** `_FIT_SCHEMA` and `_MODEL_INFO_SCHEMA` agree —
  `n_samples`/`n_features`/`n_classes`/`n_categorical` are `int64` and `task` is a
  plain `string`, so `fit` output joins cleanly to `model_info` (the `rank`
  columns are intentionally `int32`).

## Sharp edges (read before debugging)

1. **Don't name the worker module `lightgbm.py`.** It would shadow the real
   `lightgbm` package import. The entry point is `lightgbm_worker.py`; the package
   is `vgi_lightgbm`.
2. **ATTACH's first argument is the catalog name, which must be `lightgbm`.**
   `ATTACH 'lightgbm' AS lgb (...)` works; `ATTACH 'lgb' ...` fails with "Unknown
   catalog: 'lgb'. Available: lightgbm". The `AS <alias>` is how you rename it.
3. **Labels are label-encoded internally; you no longer pre-encode.** `encode_labels`
   builds a stable sorted `classes` and maps the target to `0..n-1` codes; `predict`
   decodes back. So string/float/non-contiguous labels just work — don't add a
   manual re-encoding step (it would double-encode). The `prediction` column dtype
   follows the original label dtype (VARCHAR/BIGINT), and `with_proba` keys columns
   by the original label (`proba_<label>`).
4. **SHAP `explain` is long format.** `booster.predict(x, pred_contrib=True)`
   returns `(n_rows, n_features+1)` for regression/binary and
   `(n_rows, (n_features+1)*n_classes)` for multiclass (per-class blocks, base
   value last in each block). We emit `(id, [class], feature, shap_value,
   base_value)` long rows so the output width is fixed (edge #9) and multiclass is
   supported. `base_value + sum(shap_value)` per (row[, class]) == the raw margin.
5. **Quiet LightGBM.** Always pass `verbosity=-1` (in `_QUIET` defaults) or the
   booster spews training logs over the RPC stream. Categorical fits also warn
   about overriding `categorical_feature`; that's expected.
6. **A table function gets at most ONE subquery parameter** — the table input
   (`Arg(0)`). To pass a runtime model BLOB you cannot use `model := (SELECT
   model FROM ...)`; stash it in a session variable and read it back as a scalar:
   `SET VARIABLE m = (SELECT ...)` then `predict(..., model := getvariable('m'))`.
7. **`pa.Float64Array` does not exist** — the class is `pa.DoubleArray`. A bad
   `Param`/`Arg` type hint does NOT error; the framework warns and registers the
   function with **zero input columns**. Watch for `UserWarning: ... type hints
   could not be resolved`.
8. **Table argument syntax is `(SELECT ...)`, not `TABLE(...)`.**
9. **`Arg(0)` = positional, `Arg("name")` = named-only.** A positional `Arg(0)`
   **cannot have a default** (DuckDB always requires it) — the framework raises at
   import time. So `feature_importance`'s `model_name` is a required positional;
   to use a BLOB instead, pass `''` positionally and `model :=`. The table input
   is always `Arg(0)`.
10. **Buffering / in-out state classes must extend `ArrowSerializableDataclass`**
    (e.g. `buffering.DrainState`).
11. **Output schema is fixed at bind.** Fine here: predict/explain widths come
    from the model's metadata (known at bind via `unpack_meta`/`load_meta`), and
    `explain` uses long format so it never depends on the feature count.
12. **HTTP entry point:** current vgi-python has **no `main_http`**. Serve HTTP
    via `Worker.main()` with `--http`; `serve.py` injects that flag.
13. **Generating VGI function classes dynamically:** use `types.new_class(name,
    (SinkBuffer[Args, State],), {}, lambda ns: ns.update(namespace))`. Plain
    `type(name, (Base[...],), ns)` raises "type() doesn't support MRO entry
    resolution" for subscripted-generic bases. Build the args dataclass with
    `dataclasses.make_dataclass` using `Annotated[t, Arg(...)]` field types; set
    `FunctionArguments` in the namespace. mypy can't follow this (the dynamic base
    and `cls.buffered_table`), so `typed_models.py` carries two targeted
    `# type: ignore`s.

## Testing

```sh
make venv          # .venv with vgi + lightgbm + scikit-learn (from PyPI) + ruff/mypy
make lint          # ruff + mypy (config in pyproject.toml; both run clean)
make pytest        # in-process unit tests (fast; uses tests/harness.py)
make test-sql      # SQL tests in-process via haybarn (CI-portable; no custom DuckDB build)
make test-stdio    # SQL tests, worker as a subprocess  (authoritative)
make test-http     # SQL tests against a local HTTP server
```

- **SQL tests are authoritative.** Unit tests call classmethods directly and can
  pass while the real RPC path is broken. Always run `test-stdio` (or `test-sql`).
- The same `test/sql/*.test` files run over **three transports**: stdio and HTTP
  (via DuckDB's `unittest` runner at `$(VGI_BUILD_DIR)/test/unittest`, the local
  authority) and **in-process via haybarn** (`make test-sql`). The haybarn path is
  what CI uses — it `INSTALL vgi FROM community` on Query Farm's DuckDB build, so
  it needs no custom binary. `tests/sqllogic.py` is a small subset sqllogictest
  runner; if you use a directive it doesn't support, extend it.
- For fast local probing with *real* error messages, drive haybarn from Python:
  `con.execute("INSTALL vgi FROM community; LOAD vgi")`,
  `con.execute("ATTACH 'lightgbm' AS lgb (TYPE vgi, LOCATION '<venv-python> lightgbm_worker.py')")`,
  then run SQL — far better than reading sqllogictest diffs while iterating.
- `make test-stdio` / `test-http` point `LIGHTGBM_MODELS_DIR` at an isolated
  `.test-models/` so the registry tests don't pollute `./models`.
- **CI:** `.github/workflows/ci.yml` runs ruff + mypy + unit + haybarn SQL +
  Docker smoke. Dependabot watches pip / actions / docker. Keep all steps green;
  the haybarn SQL step needs network (community extension fetch).

## Deployment (Fly.io)

The Docker image `pip install`s `vgi-python` / `vgi-rpc` straight from PyPI — no
vendoring — and adds `libgomp1` (LightGBM's OpenMP runtime).

```sh
make deploy        # build (linux/amd64) -> smoke-test -> push -> fly deploy
fly volumes create lightgbm_models --size 1 --region iad   # one-time, registry
```

`fly.toml` bumps VM memory to 1gb (lightgbm/scipy are heavy) and mounts a volume
at `/data` for the model registry (`LIGHTGBM_MODELS_DIR=/data/models`). The Docker
smoke test verifies imports + `/health`.

## Model registry

`registry.get_store()` is the single seam selecting the backend. `LocalDiskStore`
(native LightGBM text `.lgb` + JSON metadata, root from `LIGHTGBM_MODELS_DIR`,
default `./models`) is the only impl today; an `S3Store` for S3/R2 drops in here
without touching `models.py`.
