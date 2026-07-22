"""Hyperparameter search exposed as discriminated-union SQL functions.

``lightgbm.grid_search`` runs an exhaustive cross-validated grid search and
``lightgbm.randomized_search`` samples ``n_iter`` random combinations; both
return the cross-validation leaderboard (one row per combination, ranked by mean
held-out score) with the **refit best model packed as a BLOB on the single best
row** (grab it with ``WHERE model IS NOT NULL``).

The estimator is named with the same ``estimator := '<name>'`` VARCHAR argument
as ``fit`` / ``cross_val_*``, and the search grid is a set of **typed,
list-valued named arguments** — one per curated hyperparameter (the same set the
typed ``fit_lgbm_*`` functions expose):

    SELECT params, mean_score, rank FROM lightgbm.main.grid_search(
      (SELECT * FROM training), target := 'y', estimator := 'lgbm_classifier',
      num_leaves := [15, 31, 63], learning_rate := [0.05, 0.1], cv := 4)
    ORDER BY rank;

Only the hyperparameters you list are searched; the rest stay at the estimator's
defaults. No model is persisted unless you feed the best-row BLOB back into
``predict(model := ...)``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, make_dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import lightgbm as lgb
import numpy as np
import pyarrow as pa
from sklearn.model_selection import KFold, StratifiedKFold
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import (
    categorical_indices,
    detect_categoricals,
    encode_matrix,
    fit_categories,
    validate_features,
)
from .models import (
    _ESTIMATORS,
    CLASSIFICATION,
    _features_excluding,
    _target_array,
    _target_codes,
    build_estimator,
)
from .registry import ModelMetadata, now_iso, pack_model, validate_name
from .schema_utils import field as sfield
from .schema_utils import result_columns_schema
from .typed_models import _HPARAMS, _UNSET

_PYTYPE_TO_ARROW: dict[type, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
}


# Every searchable hyperparameter, keyed by name. Both estimators expose the same
# LightGBM knobs (they differ only in their default objective), so one flat set of
# typed list-valued SQL arguments covers both — see ``_search_args_class``.
_SEARCH_HPARAMS = {hp.name: hp for spec in _HPARAMS.values() for hp in spec}


def _param_grid(tag: str, args: Any) -> dict[str, list[Any]]:
    """Translate the list-valued SQL search arguments into a LightGBM param grid.

    Applies the same per-hyperparameter translations as the typed ``fit_<estimator>``
    functions, element-wise (e.g. ``max_depth`` 0 -> -1; ``objective`` '' -> None).
    Hyperparameters left empty are omitted, so they stay at the estimator default
    rather than being searched.
    """
    grid: dict[str, list[Any]] = {}
    for hp in _HPARAMS[tag]:
        vals = getattr(args, hp.name, None)
        if not vals:
            continue
        items: list[Any] = []
        for v in vals:
            if hp.none_if is not _UNSET and v == hp.none_if:
                v = None if hp.map_value is None else hp.map_value(v)
            items.append(v)
        grid[hp.name] = items
    return grid


def _search_param_fields() -> list[Any]:
    """One typed, list-valued SQL argument per searchable hyperparameter.

    A grid is a set of *candidate values per hyperparameter*, so each argument is a
    list of that hyperparameter's scalar type (``num_leaves := [15, 31]``). Keeping
    them as separate top-level arguments — rather than one struct/union blob —
    means DuckDB type-checks each one, autocomplete lists them, and the catalog
    advertises `estimator` as the same VARCHAR it is on `fit`/`cross_val_*`.
    """
    fields: list[Any] = []
    for name, hp in _SEARCH_HPARAMS.items():
        arrow = pa.list_(_PYTYPE_TO_ARROW[hp.type])
        doc = f"Candidate values for {name} (empty = leave at the estimator default). {hp.doc}"
        fields.append(
            (
                name,
                Annotated[list, Arg(name, default=(), doc=doc, arrow_type=arrow)],
                dc_field(default=()),
            )
        )
    return fields


def _search_args_class(cls_name: str, extra: list[Any]) -> type:
    """Build a search function's argument dataclass: input + estimator + grid + knobs."""
    fields: list[Any] = [
        ("data", Annotated[TableInput, Arg(0, doc="Training rows: features + target [+ id].")]),
        # No default: the estimator is required, and a placeholder default would
        # not be one of the declared choices (VGI318).
        (
            "estimator",
            Annotated[str, Arg("estimator", doc="Estimator to search.", choices=sorted(_ESTIMATORS))],
        ),
        (
            "target",
            Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")],
            dc_field(default=""),
        ),
        (
            "id",
            Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")],
            dc_field(default=""),
        ),
        *extra,
        *_search_param_fields(),
    ]
    return make_dataclass(cls_name, fields, frozen=True, slots=True)


def _grid_size(space: dict[str, list[Any]]) -> int:
    """Total number of combinations in a (list-valued) parameter grid."""
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


def _combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


def _select_combos(combos: list[dict[str, Any]], n_iter: int, random_state: int) -> list[dict[str, Any]]:
    """Pick up to ``n_iter`` combinations at random (capped at the grid size)."""
    n = min(n_iter, len(combos))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(combos), size=n, replace=False)
    return [combos[i] for i in sorted(int(i) for i in idx)]


_SEARCH_SCHEMA = pa.schema(
    [
        sfield("params", pa.string(), "JSON of this combination's hyperparameters.", nullable=False),
        sfield("mean_score", pa.float64(), "Mean held-out CV score (accuracy or R^2).", nullable=False),
        sfield("std_score", pa.float64(), "Std-dev of the held-out CV scores.", nullable=False),
        sfield("rank", pa.int32(), "1-based rank by mean_score (1 = best).", nullable=False),
        sfield("metric", pa.string(), "accuracy (classification) or r2 (regression).", nullable=False),
        sfield("model", pa.large_binary(), "Refit best model as a `BLOB` (non-NULL only on the best row)."),
    ]
)


def _validate_search_bind(name: str, params: BindParams[Any]) -> BindResponse:
    """Shared bind validation for grid_search / randomized_search."""
    a = params.args
    if not a.target:
        raise ValueError(f"{name} requires 'target' (the label column name, e.g. target := 'label')")
    if not a.estimator:
        raise ValueError(f"{name} requires 'estimator' (one of: {', '.join(sorted(_ESTIMATORS))})")
    if a.estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {a.estimator!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
    if getattr(a, "model_name", ""):
        validate_name(a.model_name)
    input_schema = params.bind_call.input_schema
    assert input_schema is not None
    if a.target not in input_schema.names:
        raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
    return BindResponse(output_schema=_SEARCH_SCHEMA)


def _run_search(cls: Any, params: Any, state: DrainState, out: OutputCollector, select: Any) -> None:
    """Shared finalize for the search functions.

    ``select(combos, args)`` returns the subset of combinations to evaluate
    (all of them for grid search; a random sample for randomized search).
    """
    if state.done:
        out.finish()
        return
    state.done = True

    a = params.args
    tag = a.estimator
    if tag not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {tag!r}")
    task = _ESTIMATORS[tag][0]
    grid = _param_grid(tag, a)
    if not grid:
        raise ValueError(f"{cls.Meta.name} requires at least one hyperparameter to search, e.g. num_leaves := [15, 31]")

    input_schema = input_schema_of(params)
    feats = _features_excluding(input_schema, a.target, a.id)

    table = cls.buffered_table(params, input_schema)
    if table is None or table.num_rows == 0:
        raise ValueError(f"{cls.Meta.name} received no training rows")

    validate_features(table, feats)
    cat_mask = detect_categoricals(table, feats)
    categories = fit_categories(table, feats, cat_mask)
    cat_idx = categorical_indices(cat_mask)
    x = encode_matrix(table, feats, cat_mask, categories)
    if task == CLASSIFICATION:
        y, classes = _target_codes(table, a.target)
    else:
        y = _target_array(table, a.target, task)
        classes = None
    metric = "accuracy" if task == CLASSIFICATION else "r2"

    combos = select(_combos(grid), a)
    if task == CLASSIFICATION:
        splits = list(StratifiedKFold(n_splits=a.cv, shuffle=True, random_state=0).split(x, y))
    else:
        splits = list(KFold(n_splits=a.cv, shuffle=True, random_state=0).split(x))

    means: list[float] = []
    stds: list[float] = []
    for combo in combos:
        fold_scores: list[float] = []
        for train_idx, test_idx in splits:
            _task, est = build_estimator(tag, combo)
            fit_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}
            est.fit(x[train_idx], y[train_idx], **fit_kwargs)
            fold_scores.append(float(est.score(x[test_idx], y[test_idx])))
        means.append(float(np.mean(fold_scores)))
        stds.append(float(np.std(fold_scores)))

    order = sorted(range(len(combos)), key=lambda i: means[i], reverse=True)
    ranks = [0] * len(combos)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    best = order[0]

    # Refit the best combo on all data and pack it as a BLOB.
    _task, best_est = build_estimator(tag, combos[best])
    fit_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}
    best_est.fit(x, y, **fit_kwargs)
    best_meta = ModelMetadata(
        name=getattr(a, "model_name", "") or "",
        estimator=tag,
        task=task,
        target=a.target,
        feature_names=feats,
        params=combos[best],
        classes=classes,
        categorical=cat_mask,
        categories=categories,
        n_samples=int(table.num_rows),
        n_features=len(feats),
        train_score=float(best_est.score(x, y)),
        lightgbm_version=lgb.__version__,
        created_at=now_iso(),
    )
    best_blob = pack_model(best_est, best_meta)

    out.emit(
        pa.RecordBatch.from_pydict(
            {
                "params": [json.dumps(c, sort_keys=True, default=str) for c in combos],
                "mean_score": means,
                "std_score": stds,
                "rank": ranks,
                "metric": [metric] * len(combos),
                "model": [best_blob if i == best else None for i in range(len(combos))],
            },
            schema=params.output_schema,
        )
    )


_GRID_SEARCH_ARGS = _search_args_class(
    "GridSearchArgs",
    [
        (
            "cv",
            Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")],
            dc_field(default=5),
        )
    ],
)

if TYPE_CHECKING:
    # mypy cannot follow ``dataclasses.make_dataclass``, so it gets a static
    # stand-in carrying the fields the code below reads. The per-hyperparameter
    # list arguments are reached only via ``getattr`` in ``_param_grid``.
    @dataclass(slots=True, frozen=True)
    class GridSearchArgs:  # noqa: D101 - mirror of the generated dataclass
        data: TableInput
        estimator: str
        target: str
        id: str
        cv: int

else:
    GridSearchArgs = _GRID_SEARCH_ARGS


class GridSearch(SinkBuffer[GridSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = GridSearchArgs

    class Meta:
        name = "grid_search"
        description = "Cross-validated grid search; returns the leaderboard + refit best model `BLOB` on the best row"
        categories = ["models", "supervised", "search"]
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_SEARCH_SCHEMA),
            "vgi.doc_llm": (
                "Runs an exhaustive cross-validated grid search over a LightGBM estimator's hyperparameters "
                "and returns the leaderboard — one row per combination tried, with its `params` (JSON), "
                "`mean_score`, `std_score`, `rank` (1 = best), and `metric` (accuracy or r2). Pick the "
                "estimator with `estimator := 'lgbm_classifier'` or `'lgbm_regressor'`, then give each "
                "hyperparameter you want swept its own list of candidate values (`num_leaves := [15, 31]`, "
                "`learning_rate := [0.05, 0.1]`, `n_estimators := [100, 300]`, ...); anything you leave out "
                "stays at the estimator default. Set `target :=`, `cv :=` folds, optional `id :=`. The refit "
                "best model rides as a `model` `BLOB` on the single best row — grab it with `WHERE model IS "
                "NOT NULL`."
            ),
            "vgi.doc_md": (
                "**Cross-validated grid search** — exhaustive hyperparameter sweep.\n\n"
                "- `estimator := 'lgbm_classifier'` | `'lgbm_regressor'`, then one list per swept "
                "hyperparameter (`num_leaves := [15, 31]`, `learning_rate := [0.05, 0.1]`, ...)\n"
                "- `target :=`, `cv :=` folds, optional `id :=`\n"
                "- Returns the leaderboard: `params`, `mean_score`, `std_score`, `rank`, `metric`\n\n"
                "The refit best model is a `model` `BLOB` on the best row — `WHERE model IS NOT NULL`."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_score, rank FROM lightgbm.main.grid_search("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.main.iris()), target := 'target', id := 'sample_id', cv := 2, "
                    "estimator := 'lgbm_classifier', num_leaves := [15, 31], learning_rate := [0.1]) "
                    "ORDER BY rank"
                ),
                description="Grid-search num_leaves x learning_rate on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GridSearchArgs]) -> BindResponse:
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[GridSearchArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GridSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        _run_search(cls, params, state, out, lambda combos, a: combos)


_RANDOMIZED_SEARCH_ARGS = _search_args_class(
    "RandomizedSearchArgs",
    [
        (
            "n_iter",
            Annotated[
                int,
                Arg("n_iter", default=10, doc="Number of random combinations to sample (capped at grid size)."),
            ],
            dc_field(default=10),
        ),
        (
            "cv",
            Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")],
            dc_field(default=5),
        ),
        (
            "search_seed",
            Annotated[int, Arg("search_seed", default=0, doc="Random seed for the combination sampler.")],
            dc_field(default=0),
        ),
    ],
)

if TYPE_CHECKING:

    @dataclass(slots=True, frozen=True)
    class RandomizedSearchArgs:  # noqa: D101 - mirror of the generated dataclass
        data: TableInput
        estimator: str
        target: str
        id: str
        n_iter: int
        cv: int
        search_seed: int

else:
    RandomizedSearchArgs = _RANDOMIZED_SEARCH_ARGS


class RandomizedSearch(SinkBuffer[RandomizedSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = RandomizedSearchArgs

    class Meta:
        name = "randomized_search"
        description = "Cross-validated randomized search: sample n_iter hyperparameter combinations"
        categories = ["models", "supervised", "search"]
        tags = {
            "vgi.result_columns_schema": result_columns_schema(_SEARCH_SCHEMA),
            "vgi.doc_llm": (
                "Runs a cross-validated randomized search over a LightGBM estimator, sampling `n_iter :=` "
                "random hyperparameter combinations (capped at the grid size) instead of exhausting the "
                "grid — cheaper than `grid_search` for large spaces. Same arguments: `estimator :=` names "
                "the estimator and each swept hyperparameter takes a list of candidate values; omitted "
                "params stay at their default. Set `target :=`, `cv :=` folds, `search_seed :=` for "
                "reproducible sampling, optional `id :=`. Returns the same leaderboard (`params`, `mean_score`, "
                "`std_score`, `rank`, `metric`) with the refit best model as a `model` `BLOB` on the best row "
                "(`WHERE model IS NOT NULL`)."
            ),
            "vgi.doc_md": (
                "**Cross-validated randomized search** — sample the hyperparameter space.\n\n"
                "- `estimator := 'lgbm_classifier'` | `'lgbm_regressor'`, then one list per swept "
                "hyperparameter\n"
                "- `n_iter :=` combinations to sample (capped at grid size), `search_seed :=` for "
                "reproducibility\n"
                "- `target :=`, `cv :=` folds, optional `id :=`\n\n"
                "Cheaper than `grid_search` on large grids; best model is a `model` `BLOB` on the best row."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_score, rank FROM lightgbm.main.randomized_search("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.main.iris()), target := 'target', id := 'sample_id', n_iter := 2, "
                    "cv := 2, estimator := 'lgbm_classifier', num_leaves := [15, 31, 63], "
                    "learning_rate := [0.05, 0.1, 0.2]) ORDER BY rank"
                ),
                description="Randomized-search num_leaves x learning_rate on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[RandomizedSearchArgs]) -> BindResponse:
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RandomizedSearchArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RandomizedSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        _run_search(
            cls,
            params,
            state,
            out,
            lambda combos, a: _select_combos(combos, a.n_iter, a.search_seed),
        )


SEARCH_FUNCTIONS: list[type] = [GridSearch, RandomizedSearch]
