"""LightGBM-specific extras that read a stored (or inline) model.

* ``feature_importance`` -- the booster's per-feature importance for a model, one
  row per feature, ranked. ``importance_type`` is ``split`` (number of times the
  feature is used in a split) or ``gain`` (total gain of those splits).
* ``explain``            -- SHAP-style per-row prediction contributions via
  LightGBM's ``pred_contrib=True``. Emitted in **long format**
  ``(row, [class], feature, shap_value, base_value)`` so the output width does
  not depend on the feature count (and multiclass models are supported, one row
  per (row, class, feature)). Streams a table through the model like ``predict``.
* ``permutation_importance`` -- model-agnostic feature importance: the drop in
  score when each feature is shuffled, ranked. Buffers an evaluation table.
* ``partial_dependence`` -- how the model's average prediction moves as one
  numeric feature varies over a grid; multiclass -> one curve per class.

All accept either a registry ``model_name`` or an inline ``model`` BLOB.

    SELECT * FROM lightgbm.feature_importance('iris_clf', importance_type := 'gain');
    SELECT * FROM lightgbm.explain((SELECT * FROM new_data), model_name := 'house_reg', id := 'id');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.inspection import partial_dependence as sk_partial_dependence
from sklearn.inspection import permutation_importance as sk_permutation_importance
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector as BufferingOutputCollector
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import OutputCollector as InOutCollector
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import encode_matrix
from .models import CLASSIFICATION, _target_array, _target_codes
from .registry import (
    ModelMetadata,
    ModelNotFoundError,
    get_store,
    unpack_meta,
    unpack_model,
)
from .schema_utils import field as sfield

_IMPORTANCE_TYPES = {"split", "gain"}


# ===========================================================================
# A small scikit-learn-compatible adapter over a loaded LightGBM Booster.
#
# The registry stores only the booster text (no pickled sklearn wrapper), but
# sklearn's partial_dependence / permutation_importance need an estimator with
# the scikit-learn API (predict / predict_proba / score / classes_). This thin
# adapter provides exactly that, fed by the model metadata.
# ===========================================================================


class _BoosterClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a Booster as a fitted sklearn classifier over 0..n-1 class codes."""

    def __init__(self, booster: Any, classes: list[Any]) -> None:
        self.booster = booster
        # classes_ are the integer codes the booster was trained on (0..n-1).
        self.classes_ = np.arange(len(classes))
        # Trailing-underscore attribute so sklearn's check_is_fitted accepts us.
        self.fitted_ = True

    def fit(self, x: Any, y: Any = None) -> _BoosterClassifier:  # pragma: no cover - already fitted
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        raw = np.asarray(self.booster.predict(x))
        if len(self.classes_) <= 2:
            p1 = raw.reshape(-1)
            return np.column_stack([1.0 - p1, p1])
        return raw

    def predict(self, x: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


class _BoosterRegressor(RegressorMixin, BaseEstimator):
    """Wrap a Booster as a fitted sklearn regressor."""

    def __init__(self, booster: Any) -> None:
        self.booster = booster
        # Trailing-underscore attribute so sklearn's check_is_fitted accepts us.
        self.fitted_ = True

    def fit(self, x: Any, y: Any = None) -> _BoosterRegressor:  # pragma: no cover - already fitted
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.booster.predict(x)).reshape(-1)


def _sklearn_adapter(booster: Any, meta: ModelMetadata) -> BaseEstimator:
    """Build a fitted sklearn-compatible estimator from a Booster + metadata."""
    if meta.task == CLASSIFICATION:
        return _BoosterClassifier(booster, meta.classes or [])
    return _BoosterRegressor(booster)


def _resolve_meta(model_name: str, model: bytes) -> ModelMetadata:
    if model:
        return unpack_meta(model)
    if not model_name:
        raise ValueError("requires either a model_name or a model BLOB")
    try:
        return get_store().load_meta(model_name)
    except ModelNotFoundError as exc:
        raise ValueError(f"model {model_name!r} not found in the registry") from exc


def _resolve_model(model_name: str, model: bytes) -> tuple[Any, ModelMetadata]:
    if model:
        return unpack_model(model)
    return get_store().load(model_name)


# ===========================================================================
# feature_importance
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FeatureImportanceArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model (pass '' to use model:= instead).")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (alternative to model_name).")]
    importance_type: Annotated[
        str, Arg("importance_type", default="gain", doc="split (split count) or gain (total gain).")
    ]


_IMPORTANCE_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("importance", pa.float64(), "Importance score for the chosen importance_type.", nullable=False),
        sfield("rank", pa.int32(), "1-based rank by importance (1 = most important).", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class FeatureImportance(TableFunctionGenerator[FeatureImportanceArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _IMPORTANCE_SCHEMA

    class Meta:
        name = "feature_importance"
        description = "Per-feature importance (split or gain) for a model, ranked"
        categories = ["models", "interpretation"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM lightgbm.feature_importance('iris_clf', importance_type := 'gain')",
                description="Gain-based feature importance for a stored model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FeatureImportanceArgs]) -> BindResponse:
        a = params.args
        if a.importance_type not in _IMPORTANCE_TYPES:
            raise ValueError(
                f"invalid importance_type {a.importance_type!r}; choose one of: {', '.join(sorted(_IMPORTANCE_TYPES))}"
            )
        _resolve_meta(a.model_name, a.model)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def cardinality(cls, params: BindParams[FeatureImportanceArgs]) -> TableCardinality:
        return TableCardinality(estimate=20, max=100000)

    @classmethod
    def process(cls, params: ProcessParams[FeatureImportanceArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        booster, meta = _resolve_model(a.model_name, a.model)
        scores = booster.feature_importance(importance_type=a.importance_type)
        rows = list(zip(meta.feature_names, [float(s) for s in scores], strict=True))
        rows.sort(key=lambda r: r[1], reverse=True)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": [name for name, _ in rows],
                    "importance": [imp for _, imp in rows],
                    "rank": [i + 1 for i in range(len(rows))],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


# ===========================================================================
# explain (SHAP prediction contributions, long format)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ExplainArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to explain (must contain the model's feature columns).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Name of a stored model (or pass model:=).")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (alternative to model_name).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]


_EXPLAIN_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


class ExplainModel(TableInOutGenerator[ExplainArgs]):
    FunctionArguments: ClassVar[type] = ExplainArgs

    class Meta:
        name = "explain"
        description = "Per-row SHAP feature contributions, long format (row, [class], feature, shap_value, base_value)"
        categories = ["models", "interpretation", "inference"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM lightgbm.explain((SELECT * FROM lightgbm.diabetes()), "
                    "model_name := 'diab_reg', id := 'sample_id')"
                ),
                description="Explain each row's prediction with per-feature contributions",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ExplainArgs]) -> BindResponse:
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _resolve_meta(a.model_name, a.model)

        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} "
                f"not present in the input; model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )

        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        multiclass = meta.task == CLASSIFICATION and meta.classes is not None and len(meta.classes) > 2
        if multiclass:
            fields.append(sfield("class", pa.int64(), "Class index the contribution applies to.", nullable=False))
        fields.append(sfield("feature", pa.string(), "Feature column name.", nullable=False))
        fields.append(
            sfield("shap_value", pa.float64(), "Contribution of the feature to the raw margin.", nullable=False)
        )
        fields.append(sfield("base_value", pa.float64(), "Model base (expected) raw-margin value.", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[ExplainArgs]) -> tuple[Any, ModelMetadata]:
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _EXPLAIN_CACHE.get(key)
        if cached is None:
            cached = _resolve_model(params.args.model_name, params.args.model)
            _EXPLAIN_CACHE[key] = cached
        return cached

    @classmethod
    def process(
        cls,
        params: ProcessParams[ExplainArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        booster, meta = cls._model(params)
        table = pa.Table.from_batches([batch])
        x = encode_matrix(table, meta.feature_names, meta.categorical, meta.categories)

        contribs = np.asarray(booster.predict(x, pred_contrib=True))
        feats = meta.feature_names
        n_feat = len(feats)
        n_rows = x.shape[0]
        ids = batch.column(a.id).to_pylist() if a.id else None
        n_classes = len(meta.classes) if (meta.classes is not None) else 0
        multiclass = meta.task == CLASSIFICATION and n_classes > 2

        id_out: list[Any] = []
        class_out: list[int] = []
        feature_out: list[str] = []
        shap_out: list[float] = []
        base_out: list[float] = []

        # LightGBM lays out contribs as (n_features+1) per class, concatenated.
        n_blocks = n_classes if multiclass else 1
        for r in range(n_rows):
            for b in range(n_blocks):
                off = b * (n_feat + 1)
                base = float(contribs[r, off + n_feat])
                for j, fname in enumerate(feats):
                    if ids is not None:
                        id_out.append(ids[r])
                    if multiclass:
                        class_out.append(b)
                    feature_out.append(fname)
                    shap_out.append(float(contribs[r, off + j]))
                    base_out.append(base)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = id_out
        if multiclass:
            columns["class"] = class_out
        columns["feature"] = feature_out
        columns["shap_value"] = shap_out
        columns["base_value"] = base_out
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# permutation_importance (model-agnostic feature importance, ranked)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PermImportanceArgs:
    data: Annotated[TableInput, Arg(0, doc="Evaluation table (the model's features + the target column).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB. Provide this OR model_name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    n_repeats: Annotated[int, Arg("n_repeats", default=5, doc="Number of times each feature is shuffled.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


_PERM_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("importance_mean", pa.float64(), "Mean drop in score when the feature is shuffled.", nullable=False),
        sfield("importance_std", pa.float64(), "Std-dev of the importance across repeats.", nullable=False),
        sfield("rank", pa.int32(), "1-based rank by importance_mean (1 = most important).", nullable=False),
    ]
)


class PermutationImportance(SinkBuffer[PermImportanceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = PermImportanceArgs

    class Meta:
        name = "permutation_importance"
        description = "Model-agnostic feature importance: the drop in score when each feature is shuffled, ranked"
        categories = ["models", "interpretation", "evaluation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM lightgbm.permutation_importance((SELECT * FROM lightgbm.iris()), "
                    "model_name := 'iris_clf', target := 'target') ORDER BY rank"
                ),
                description="Rank iris features by permutation importance for a stored model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PermImportanceArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("permutation_importance requires either 'model_name' or 'model' (a model BLOB)")
        if not a.target:
            raise ValueError("permutation_importance requires 'target' (the label column name)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _resolve_meta(a.model_name, a.model)
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} not present in the input; "
                f"model features: {', '.join(meta.feature_names)}"
            )
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_PERM_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PermImportanceArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PermImportanceArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: BufferingOutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        booster, meta = _resolve_model(a.model_name, a.model)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("permutation_importance received no rows")

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = encode_matrix(table, meta.feature_names, cat_mask, meta.categories)
        if meta.task == CLASSIFICATION:
            y, _classes = _target_codes(table, a.target)
        else:
            y = _target_array(table, a.target, meta.task)

        estimator = _sklearn_adapter(booster, meta)
        result = sk_permutation_importance(
            estimator, x, y, n_repeats=a.n_repeats, random_state=a.random_state, scoring=(a.scoring or None)
        )
        rows = list(
            zip(
                meta.feature_names,
                [float(v) for v in result.importances_mean],
                [float(v) for v in result.importances_std],
                strict=True,
            )
        )
        rows.sort(key=lambda r: r[1], reverse=True)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": [name for name, _, _ in rows],
                    "importance_mean": [mean for _, mean, _ in rows],
                    "importance_std": [std for _, _, std in rows],
                    "rank": [i + 1 for i in range(len(rows))],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# partial_dependence (how the model's prediction moves with one feature)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PartialDependenceArgs:
    data: Annotated[TableInput, Arg(0, doc="Background table (the model's feature columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB. Provide this OR model_name.")]
    feature: Annotated[str, Arg("feature", default="", doc="Numeric feature column to vary (required).")]
    grid_resolution: Annotated[int, Arg("grid_resolution", default=100, doc="Number of grid points along the feature.")]


_PD_SCHEMA = pa.schema(
    [
        sfield("feature_value", pa.float64(), "Value the feature was set to.", nullable=False),
        sfield("class", pa.int64(), "Class label (NULL for regression / the single binary curve)."),
        sfield("partial_dependence", pa.float64(), "Average model output at this feature value.", nullable=False),
    ]
)


class PartialDependence(SinkBuffer[PartialDependenceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = PartialDependenceArgs

    class Meta:
        name = "partial_dependence"
        description = "How a stored model's average prediction changes as one feature varies over a grid"
        categories = ["models", "interpretation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM lightgbm.partial_dependence((SELECT * FROM lightgbm.iris()), "
                    "model_name := 'iris_clf', feature := 'petal_length_cm') ORDER BY feature_value"
                ),
                description="Partial dependence of 'iris_clf' on petal length",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PartialDependenceArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("partial_dependence requires either 'model_name' or 'model' (a model BLOB)")
        if not a.feature:
            raise ValueError("partial_dependence requires 'feature' (the column to vary)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _resolve_meta(a.model_name, a.model)
        if a.feature not in meta.feature_names:
            raise ValueError(
                f"feature {a.feature!r} is not one of the model's features: {', '.join(meta.feature_names)}"
            )
        idx = meta.feature_names.index(a.feature)
        if (meta.categorical or [False] * len(meta.feature_names))[idx]:
            raise ValueError(f"partial_dependence supports numeric features only; {a.feature!r} is categorical")
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(f"model requires feature column(s) {', '.join(missing)} not present in the input")
        return BindResponse(output_schema=_PD_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PartialDependenceArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PartialDependenceArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: BufferingOutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        booster, meta = _resolve_model(a.model_name, a.model)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("partial_dependence received no rows")

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = encode_matrix(table, meta.feature_names, cat_mask, meta.categories)
        idx = meta.feature_names.index(a.feature)
        estimator = _sklearn_adapter(booster, meta)
        result = sk_partial_dependence(estimator, x, [idx], grid_resolution=a.grid_resolution, kind="average")
        grid = result["grid_values"][0]
        averages = np.asarray(result["average"])  # shape (n_outputs, n_grid)

        # Label each output's curve: regression -> NULL; binary -> the positive
        # class; multiclass -> one curve per class. classes are the original labels.
        if meta.task == CLASSIFICATION and meta.classes:
            labels: list[Any] = meta.classes if averages.shape[0] > 1 else [meta.classes[-1]]
        else:
            labels = [None] * averages.shape[0]

        # When there's one curve per class, the `class` column is the integer class
        # code (0..n-1) — robust whether the original labels are ints or strings.
        multiclass = averages.shape[0] > 1
        feature_value: list[float] = []
        class_col: list[Any] = []
        pd_col: list[float] = []
        for o in range(averages.shape[0]):
            code = _class_code(labels[o], o) if multiclass else None
            for g in range(len(grid)):
                feature_value.append(float(grid[g]))
                class_col.append(code)
                pd_col.append(float(averages[o, g]))
        out.emit(
            pa.RecordBatch.from_pydict(
                {"feature_value": feature_value, "class": class_col, "partial_dependence": pd_col},
                schema=params.output_schema,
            )
        )


def _class_code(label: Any, code: int) -> int:
    """Integer for the ``class`` column: the original int label if it is one, else the code."""
    if isinstance(label, bool):
        return code
    if isinstance(label, int):
        return label
    return code


IMPORTANCE_FUNCTIONS: list[type] = [
    FeatureImportance,
    ExplainModel,
    PermutationImportance,
    PartialDependence,
]
