"""Hyperparameter search: ``grid_search`` over a single LightGBM estimator.

A buffering function that runs an exhaustive cross-validated grid search over a
JSON parameter grid (each key maps to a list of candidate values), then returns
the CV leaderboard — one row per parameter combination, ranked by mean held-out
score — with the **refit best model packed as a BLOB on the single best row**
(grab it with ``WHERE model IS NOT NULL``). No model is persisted unless you feed
that BLOB back into ``predict(model := ...)``.

    SELECT params, mean_score, rank FROM lightgbm.grid_search(
      (SELECT * FROM training), estimator := 'lgbm_classifier', target := 'y',
      grid := '{"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1]}', cv := 4)
    ORDER BY rank;
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

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
    CLASSIFICATION,
    _features_excluding,
    _target_array,
    build_estimator,
)
from .registry import ModelMetadata, now_iso, pack_model
from .schema_utils import field as sfield


def _parse_grid(grid: str) -> dict[str, list[Any]]:
    grid = (grid or "").strip()
    if not grid:
        raise ValueError("grid_search requires a non-empty 'grid' JSON object, e.g. '{\"num_leaves\": [15, 31]}'")
    parsed = json.loads(grid)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError('grid must be a non-empty JSON object mapping param -> list of values')
    out: dict[str, list[Any]] = {}
    for k, v in parsed.items():
        out[k] = list(v) if isinstance(v, list | tuple) else [v]
    return out


def _combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


@dataclass(slots=True, frozen=True)
class GridSearchArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="lgbm_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    grid: Annotated[str, Arg("grid", default="", doc="JSON object mapping hyperparameter -> list of values.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]


_GRID_SCHEMA = pa.schema(
    [
        sfield("params", pa.string(), "JSON of this combination's hyperparameters.", nullable=False),
        sfield("mean_score", pa.float64(), "Mean held-out CV score (accuracy or R^2).", nullable=False),
        sfield("std_score", pa.float64(), "Std-dev of the held-out CV scores.", nullable=False),
        sfield("rank", pa.int32(), "1-based rank by mean_score (1 = best).", nullable=False),
        sfield("metric", pa.string(), "accuracy (classification) or r2 (regression).", nullable=False),
        sfield("model", pa.large_binary(), "Refit best model as a BLOB (non-NULL only on the best row)."),
    ]
)


class GridSearch(SinkBuffer[GridSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = GridSearchArgs

    class Meta:
        name = "grid_search"
        description = "Cross-validated grid search; returns the leaderboard + refit best model BLOB on the best row"
        categories = ["models", "supervised", "search"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_score, rank FROM lightgbm.grid_search("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM lightgbm.iris()), estimator := 'lgbm_classifier', target := 'target', "
                    "grid := '{\"num_leaves\": [15, 31], \"learning_rate\": [0.05, 0.1]}', cv := 4) ORDER BY rank"
                ),
                description="Grid-search num_leaves x learning_rate on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GridSearchArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("grid_search requires 'target' (the label column name, e.g. target := 'label')")
        # Validate the estimator and every grid combination at bind time.
        grid = _parse_grid(a.grid)
        for combo in _combos(grid):
            build_estimator(a.estimator, combo)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(
                f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}"
            )
        return BindResponse(output_schema=_GRID_SCHEMA)

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
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        task, _ = build_estimator(a.estimator, {})

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("grid_search received no training rows")

        validate_features(table, feats)
        cat_mask = detect_categoricals(table, feats)
        categories = fit_categories(table, feats, cat_mask)
        cat_idx = categorical_indices(cat_mask)
        x = encode_matrix(table, feats, cat_mask, categories)
        y = _target_array(table, a.target, task)
        grid = _parse_grid(a.grid)
        combos = _combos(grid)
        metric = "accuracy" if task == CLASSIFICATION else "r2"

        if task == CLASSIFICATION:
            splits = list(StratifiedKFold(n_splits=a.cv, shuffle=True, random_state=0).split(x, y))
        else:
            splits = list(KFold(n_splits=a.cv, shuffle=True, random_state=0).split(x))

        means: list[float] = []
        stds: list[float] = []
        for combo in combos:
            fold_scores: list[float] = []
            for train_idx, test_idx in splits:
                _task, est = build_estimator(a.estimator, combo)
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
        _task, best_est = build_estimator(a.estimator, combos[best])
        fit_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}
        best_est.fit(x, y, **fit_kwargs)
        best_classes = [int(c) for c in best_est.classes_] if task == CLASSIFICATION else None
        best_meta = ModelMetadata(
            name="",
            estimator=a.estimator,
            task=task,
            target=a.target,
            feature_names=feats,
            params=combos[best],
            classes=best_classes,
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
                    "params": [json.dumps(c, sort_keys=True) for c in combos],
                    "mean_score": means,
                    "std_score": stds,
                    "rank": ranks,
                    "metric": [metric] * len(combos),
                    "model": [best_blob if i == best else None for i in range(len(combos))],
                },
                schema=params.output_schema,
            )
        )


SEARCH_FUNCTIONS: list[type] = [GridSearch]
