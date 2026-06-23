"""LightGBM-specific extras that read a stored (or inline) model.

* ``feature_importance`` -- the booster's per-feature importance for a model, one
  row per feature, ranked. ``importance_type`` is ``split`` (number of times the
  feature is used in a split) or ``gain`` (total gain of those splits).
* ``explain``            -- SHAP-style per-row prediction contributions via
  LightGBM's ``pred_contrib=True``. Emitted in **long format**
  ``(row, [class], feature, shap_value, base_value)`` so the output width does
  not depend on the feature count (and multiclass models are supported, one row
  per (row, class, feature)). Streams a table through the model like ``predict``.

Both accept either a registry ``model_name`` or an inline ``model`` BLOB.

    SELECT * FROM lightgbm.feature_importance('iris_clf', importance_type := 'gain');
    SELECT * FROM lightgbm.explain((SELECT * FROM new_data), model_name := 'house_reg', id := 'id');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
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

from .features import encode_matrix
from .models import CLASSIFICATION
from .registry import (
    ModelMetadata,
    ModelNotFoundError,
    get_store,
    unpack_meta,
    unpack_model,
)
from .schema_utils import field as sfield

_IMPORTANCE_TYPES = {"split", "gain"}


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


IMPORTANCE_FUNCTIONS: list[type] = [FeatureImportance, ExplainModel]
